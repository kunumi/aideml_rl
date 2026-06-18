# Guia Polyaxon — Kunumi

Guia de referência para usar o Polyaxon nos experimentos do time.

---

## 1. O que é o Polyaxon?

O Polyaxon é um **orquestrador de jobs de ML**. Pense nele como um gerente de fila de trabalhos que:

1. Recebe o seu pedido ("quero rodar esse script com 1 GPU H200")
2. Aloca a máquina certa no cluster (Nebius ou GCP)
3. Executa o job dentro de um container Docker
4. Desaloca a máquina quando termina

**Por que usar em vez de rodar local?**
- Acesso a GPUs grandes (H200, A100) sem gerenciar infraestrutura
- Jobs rodam em paralelo sem conflitar entre si
- Histórico de experimentos, logs e artefatos centralizados
- Integração com W&B para rastrear métricas

---

## 2. Como acessar

### Instalar a CLI

```bash
pip install polyaxon
```

### Configurar o host

```bash
polyaxon config set --host=https://nb.sc.kunumi.io
```

### Login

```bash
polyaxon login
```

### UIs

| Cluster | URL |
|---|---|
| Nebius | http://nb.sc.kunumi.io |
| GCP | https://sc.kunumi.io |

---

## 3. Estrutura de um YAML

Todo job no Polyaxon é definido por um arquivo YAML. Veja o exemplo mínimo em [plx_exemplo_minimal.yaml](plx_exemplo_minimal.yaml):

```yaml
version: 1.1
kind: component       # "component" = template reutilizável
name: hello-polyaxon

inputs:               # parâmetros do job
  - name: mensagem
    type: str
    isOptional: true
    value: "mundo"

run:
  kind: job           # tipo mais comum: container que roda e termina
  container:
    image: python:3.11-slim
    command:
      - sh
      - "-c"
      - echo "Olá, {{ mensagem }}!"   # {{ }} injeta o valor do input
```

### Blocos principais

| Bloco | O que faz |
|---|---|
| `inputs` | Declara os parâmetros do job (tipo, valor padrão, obrigatório ou não) |
| `run.kind` | Tipo de execução (`job` é o mais comum) |
| `run.connections` | Serviços externos necessários (GCS, Docker registry, W&B) |
| `run.environment.nodeSelector` | Qual tipo de máquina alocar |
| `container.image` | Imagem Docker a usar |
| `container.command` | O que executar dentro do container |
| `container.resources` | Recursos (GPU, CPU, memória) |

### Inputs opcionais com template condicional

Quando um input é opcional, use a sintaxe Jinja2 para não passar o argumento se ele não foi definido:

```yaml
{% if lr is not none %}--lr {{ lr }}{% endif %}
```

---

## 4. Como disparar um job

### Comando básico

```bash
polyaxon run -f plx_exemplo_nebius.yaml --upload -p kunumi/k-space
```

- `-f`: arquivo YAML do job
- `--upload`: envia o código do diretório atual para o cluster antes de rodar
- `-p kunumi/k-space`: projeto Polyaxon onde o run vai aparecer

### Ignorar arquivos no upload (.polyaxonignore)

O `--upload` envia tudo no diretório atual. Para excluir arquivos pesados (dados, checkpoints, caches), crie um arquivo `.polyaxonignore` na raiz do projeto:

```
# dados
data/
*.csv
*.parquet

# checkpoints e modelos
*.pt
*.ckpt
checkpoints/

# outros
__pycache__/
.venv/
```

### Sobrescrever parâmetros na linha de comando

Use `-P nome=valor` para mudar qualquer input sem editar o YAML:

```bash
polyaxon run -f plx_exemplo_nebius.yaml --upload -p kunumi/k-space \
  -P lr=5e-5 \
  -P epochs=20 \
  -P wandb_project=meu-experimento
```

### Rodar múltiplos experimentos (ablation)

Para rodar várias combinações de parâmetros de uma vez, use um arquivo de `operation` com `matrix`:

```bash
polyaxon run -f plx_exemplo_ablation.yaml --upload -p kunumi/k-space
```

Cada combinação definida em `matrix.values` gera um run separado. O campo `concurrency` controla quantos rodam em paralelo.

### Verificar a sintaxe do YAML antes de rodar

```bash
polyaxon check -f plx_exemplo_nebius.yaml
```

---

## 5. GCP vs Nebius

Os dois clusters funcionam da mesma forma, mas têm algumas diferenças de configuração:

| | GCP | Nebius |
|---|---|---|
| `queue` | (não precisa definir) | `nebius/default` |
| `nodeSelector` | `polyaxon: a100-2gpus` | `polyaxon: h200` |
| `imagePullSecrets` | (não precisa) | `["docker-nebius-conf"]` |
| Registry da imagem | `us-central1-docker.pkg.dev/...` | `cr.us-central1.nebius.cloud/...` |

Veja as diferenças lado a lado em [plx_exemplo_gcp.yaml](plx_exemplo_gcp.yaml) e [plx_exemplo_nebius.yaml](plx_exemplo_nebius.yaml).

### Hardware disponível no Nebius

| Node group | Hardware | GPUs | CPU | RAM | Nodes |
|---|---|---:|---:|---:|---:|
| h200 | H200 NVLink | 1 | 16 | 200 GiB | 1–5 |
| b200 | B200 NVLink | 8 | 160 | 1792 GiB | 0–1 (provisionando) |
| cpus | AMD Epyc Genoa | — | 4 | 16 GiB | 0–10 |
| main | AMD Epyc Genoa | — | 4 | 16 GiB | 2 |

---

## 6. Docker images

O código roda dentro de um container Docker. A imagem é definida no campo `container.image` do YAML.

### Onde ficam as imagens?

| Cluster | Registry | Exemplo de imagem |
|---|---|---|
| GCP | `us-central1-docker.pkg.dev/k-supercomputing/k-images/` | `kspace:latest` |
| Nebius | `cr.us-central1.nebius.cloud/u00fa0w6fr7qe1zq5p/` | `kspace:base_1` |

### Como referenciar no YAML

```yaml
# GCP
image: "us-central1-docker.pkg.dev/k-supercomputing/k-images/kspace:latest"

# Nebius
image: "cr.us-central1.nebius.cloud/u00fa0w6fr7qe1zq5p/kspace:base_1"
```

### Acesso ao registry privado (Nebius)

No Nebius é necessário informar o secret de autenticação para o Kubernetes conseguir baixar a imagem:

```yaml
environment:
  imagePullSecrets: ["docker-nebius-conf"]
```

No GCP isso não é necessário.

---

## 7. Como usar a UI

Acesse a UI pelo browser (veja URLs na seção 2) e navegue pelo menu lateral:

- **Projects**: lista de projetos — entre em `kunumi/k-space` para ver os runs do time
- **Runs**: lista de jobs disparados; filtre por status (running, succeeded, failed)
- **Logs**: clique em um run e vá na aba "Logs" para ver o output em tempo real
- **Artifacts**: arquivos salvos pelo job (checkpoints, métricas, outputs)
- **Dashboards**: métricas plotadas se você usar `polyaxon.tracking`

---

## 8. W&B (Weights & Biases)

O W&B é uma ferramenta para rastrear experimentos de ML.

### Como a integração funciona

A integração é feita via variáveis de ambiente no YAML. O seu script Python usa o W&B normalmente (`wandb.init()`, `wandb.log()`), e o Polyaxon garante que as credenciais estejam disponíveis via `wandb-connection`.

```yaml
run:
  connections: [..., wandb-connection]  # disponibiliza as credenciais W&B
  container:
    env:
      - name: WANDB_PROJECT
        value: "{{ wandb_project }}"    # nome do projeto no W&B
      - name: TEAM_WANDB
        value: franciscokunumi-universidade-federal-de-minas-gerais
```

### No seu script Python

```python
import wandb

wandb.init(project="meu-projeto")

# log de métricas a cada época
wandb.log({"loss": 0.5, "accuracy": 0.8, "epoch": 1})

# log de um artefato (ex: modelo salvo)
wandb.save("modelo.pt")
```

### Onde ver os resultados

Acesse [wandb.ai](https://wandb.ai) e navegue até o projeto. Você verá todos os runs com gráficos de métricas, configurações e artefatos.

---

## 9. Agente de configuração (Denis)

O Denis escreveu um agente que consegue gerar e validar YAMLs do Polyaxon automaticamente. Use quando precisar criar um novo arquivo do zero ou adaptar um existente para um caso diferente:

[polyaxon-config-generator](https://github.com/kunumi/autobots/blob/main/agents/polyaxon-config-generator.md)

---

## Referência rápida

```bash
# Configurar CLI
polyaxon config set --host=http://nb.sc.kunumi.io   # Nebius
polyaxon config set --host=https://sc.kunumi.io      # GCP
polyaxon login

# Rodar um job
polyaxon run -f plx_exemplo_nebius.yaml --upload -p kunumi/k-space

# Rodar sobrescrevendo parâmetros
polyaxon run -f plx_exemplo_nebius.yaml --upload -p kunumi/k-space -P lr=1e-4 -P epochs=5

# Rodar ablation (múltiplos runs)
polyaxon run -f plx_exemplo_ablation.yaml --upload -p kunumi/k-space

# Validar sintaxe do YAML
polyaxon check -f plx_exemplo_nebius.yaml

# Ver runs em andamento
polyaxon ops ls -p kunumi/k-space

# Ver logs de um run
polyaxon ops logs <run-uuid> -p kunumi/k-space
```
