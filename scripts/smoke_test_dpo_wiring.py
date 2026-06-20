#!/usr/bin/env python3
"""Smoke-test DPO controller wiring via a mock OpenAI-compatible server."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aide.controller import LLMController
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.config import SearchConfig
from aide.utils.metric import MetricValue


MOCK_RESPONSE = json.dumps(
    {
        "action": "improve",
        "hint": "Try feature engineering on temporal columns and validate with cross-validation.",
        "confidence": 0.85,
    }
)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        body = {
            "id": "mock",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": MOCK_RESPONSE},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
        }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    port = 8765
    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}/v1"
    search_cfg = SearchConfig(
        max_debug_depth=3,
        debug_prob=0.5,
        num_drafts=5,
        policy_kind="controller",
        policy_model=None,
        policy_temp=0.7,
        policy_max_obs_nodes=32,
        controller_kind="llm",
        controller_model="aide-dpo-mock",
        controller_temp=0.0,
        controller_base_url=base_url,
        hint_max_chars=600,
        hint_pool_path=None,
    )

    node = Node(
        plan="baseline plan",
        code="print('hello')",
        metric=MetricValue(0.5, maximize=True),
        is_buggy=False,
    )
    node.absorb_exec_result(
        ExecutionResult(term_out=["metric=0.5\n"], exec_time=0.1, exc_type=None, exc_info=None, exc_stack=None)
    )
    journal = Journal()
    journal.append(node)

    controller = LLMController()
    out = controller.decide(node, "Predict churn", journal, search_cfg)
    server.shutdown()

    if out is None:
        raise SystemExit("Controller returned None — wiring failed")
    if out.action != "improve" or not out.hint:
        raise SystemExit(f"Unexpected controller output: {out}")
    print(f"OK: action={out.action} hint={out.hint[:60]}... confidence={out.confidence}")


if __name__ == "__main__":
    main()
