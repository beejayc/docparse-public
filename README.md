# docparse

Local, batch-first document parser for RAG. It converts mixed enterprise files into reading-order
markdown plus a block-level JSON sidecar in which every block carries a page number, a bounding box
and a type, so a RAG system can cite back to the exact place on the page. Chunking is out of scope:
docparse stops at markdown + blocks.

- **Engine:** [LightOnOCR-3](https://huggingface.co/lightonai/LightOnOCR-3-0.8B) (Apache-2.0), 0.8B for
  all pages, 4B for pages that fail QA, served by [vLLM](https://github.com/vllm-project/vllm).
- **Batch driver:** [saturate](https://pypi.org/project/saturate/) (Apache-2.0), which keeps the GPU
  saturated with an adaptive in-flight window and gives crash-safe, resumable parquet output.
- **Inputs:** PDF (digital and scanned), images, DOCX/PPTX (via LibreOffice), XLSX/CSV (native),
  EML/MSG (with attachments), ZIP.
- **Languages:** English and German.
- **Scale target:** 50k–1M pages per backfill on one GPU.
- **Licence policy:** permissive dependencies only (Apache, MIT, BSD). Enforced by `make check`.

## Output

```
out/docs/{doc_id}/
  document.md        # reading-order markdown; tables as pipe tables or HTML when cells merge
  blocks.jsonl       # one block per line: type, page, bbox, text, html, quality flags
  tables/{block_id}.json   # expanded cell grid for each table and chart
  assets/{block_id}.png    # figure and chart crops
  meta.json          # source paths, models, profile, QA summary, parser version
```

## Quick start

One command, any machine with Docker and an NVIDIA GPU (details per platform below):

```bash
git clone <this repo> && cd DocumentParsingPipeline
INPUT_DIR=/path/to/documents OUTPUT_DIR=/path/to/results \
  docker compose -f docker/compose.yaml --profile run run --rm parser parse /in /out
```

This builds the parser image, starts vLLM with LightOnOCR-3-0.8B, waits until it is healthy, then
inventories, normalises, OCRs, runs QA and assembles. Results are in `$OUTPUT_DIR/out/docs/<doc_id>/`.
Safe to stop and re-run: every stage resumes. The first start downloads the model (about 2 GB) into
`~/.cache/huggingface`.

Pages that fail QA are re-read by the 4B model. One vLLM server serves one model, so the one-command
run finishes with those pages marked `escalation_pending` and prints how to finish them (serve
`MODEL=accurate`, then `docparse run --pass 2`, `docparse qa --pass 2`, `docparse assemble --force`).
Use `--no-escalate` to skip pass 2 on purpose.

### From Python

```python
from pathlib import Path
from docparse.api import parse_corpus

summary = parse_corpus(Path("corpus"), Path("results"), profile="generic")
print(summary.docs, summary.pages, summary.docs_dir, summary.warnings)
```

### Step by step (native install)

```bash
uv sync                                       # or: pip install .
make up PROFILE=generic MODEL=fast            # vLLM container (Linux/WSL2 + NVIDIA)
docparse --profile generic inventory /data/corpus
docparse --profile generic normalise
docparse --profile generic run --pass 1       # safe to kill and re-run; resumes exactly
docparse --profile generic qa --pass 1
make down && make up PROFILE=generic MODEL=accurate
docparse --profile generic run --pass 2       # 4B model, flagged pages only
docparse --profile generic qa --pass 2
docparse --profile generic assemble
docparse report <run_id>
```

Or all of it: `docparse --profile generic parse /data/corpus /data/results`.

## Platforms

docparse has two halves. The **parser** (render, normalise, QA, assemble) is plain Python and runs
on Linux, macOS and Windows. The **model server** (vLLM with CUDA) needs an NVIDIA GPU on Linux or
WSL2. Any platform without that GPU uses the `remote` profile and points at a server elsewhere.

| Your machine | Model server | Parser | Profile |
| --- | --- | --- | --- |
| Linux x86-64 + NVIDIA GPU | Docker (local) | Docker or native | `generic` (or `consumer24`, `pro6000`) |
| Linux ARM64 + NVIDIA GPU (DGX Spark, Grace) | Docker (local) | Docker or native | `gb10` or `generic` |
| Windows 11 + NVIDIA GPU | Docker Desktop (WSL2 backend) | Docker or WSL2 | `generic` |
| macOS (Intel or Apple Silicon) | none (no CUDA): use a remote server | native or Docker | `remote` |
| Windows without WSL2 | remote server | native (best effort) | `remote` |
| Any host without a GPU | remote server | native or Docker | `remote` |

Tested here: Linux ARM64 (GB10). Windows and macOS are supported by design (OS differences are
isolated in `src/docparse/osutil.py` and unit tested) but have not been run on real machines yet.

### Linux (amd64 or arm64) with an NVIDIA GPU

1. Install the NVIDIA driver, Docker Engine and the
   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
   check with `docker run --rm --gpus all ubuntu nvidia-smi`.
2. Run the one-command quick start above.
3. For the fastest settings, use a hardware profile: `make up PROFILE=pro6000` and
   `--profile pro6000`. Profiles pin vLLM and render workers to CPU cores, so check the core ids in
   `config/profiles/<name>.yaml` match your CPU (`lscpu -e`), or stay on `generic` (no pinning).
4. On ARM64 the vLLM image tag differs: `export VLLM_IMAGE=vllm/vllm-openai:v0.33.0-aarch64`
   (verified on GB10; the `gb10` profile already sets it).

### Windows 11 + NVIDIA GPU

1. Install the current NVIDIA Windows driver (it includes WSL2 GPU support) and
   [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) with the WSL2
   backend enabled.
2. Open PowerShell in the repository and run:

   ```powershell
   $env:INPUT_DIR = "C:\data\documents"; $env:OUTPUT_DIR = "C:\data\results"
   docker compose -f docker/compose.yaml --profile run run --rm parser parse /in /out
   ```
3. For native development, work inside a WSL2 Ubuntu shell and follow the Linux steps.
   Native Windows Python also works for the parser half (`remote` profile): `pip install .`, install
   [LibreOffice](https://www.libreoffice.org/) for DOCX/PPTX (found automatically, or set
   `DOCPARSE_SOFFICE`), and use `docparse --profile remote --endpoint http://<gpu-host>:8000 ...`.

### macOS

No CUDA, so no local model server. Run the parser locally and the model on any NVIDIA machine:

```bash
# on the GPU machine (Linux or WSL2): expose vLLM on your network
make up PROFILE=generic        # edit docker/compose.yaml ports to "0.0.0.0:8000:8000"; firewall it!

# on the Mac
brew install uv libreoffice    # LibreOffice only needed for DOCX/PPTX
uv sync
docparse --profile remote --endpoint http://gpu-host:8000 parse ~/Documents/corpus ~/results
```

Or with Docker Desktop for Mac: set `DOCPARSE_ENDPOINT=http://gpu-host:8000` and run the compose
command with `--no-deps` so it does not try to start a local vLLM:
`docker compose -f docker/compose.yaml --profile run run --rm --no-deps parser parse /in /out`.
The vLLM endpoint has no authentication: only expose it on a trusted network or through an SSH tunnel
(`ssh -L 8000:localhost:8000 gpu-host`, then use the default `http://localhost:8000`).

### CPU-only or no GPU

Same as macOS: use `--profile remote` with `--endpoint` or `DOCPARSE_ENDPOINT`. Without any server
reachable, the run stops early with `lightonai/LightOnOCR-3-0.8B is not served at ...`.

### Choosing and tuning a profile

| Profile | GPU | Notes |
| --- | --- | --- |
| `generic` | any NVIDIA, 16 GB+ | Default. No CPU pinning. Start here on a new machine |
| `remote` | none | Model server elsewhere (`--endpoint`) |
| `consumer24` | 24 GB consumer card (RTX 4090 class) | Core ids assume an 16-thread CPU; image tag unverified |
| `pro6000` | RTX PRO 6000, 96 GB | Production backfills; image tag unverified |
| `gb10` | DGX Spark, 128 GB unified, ARM64 | Measured: 0.945 pages/s |

Profiles live in `config/profiles/*.yaml`. `docparse bench` measures render speed and throughput on
your machine and writes the result back into the profile.

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `... is not served at http://localhost:8000` | Start vLLM (`make up` or the compose command) or pass `--endpoint` |
| `permission denied ... docker.sock` | Add your user to the `docker` group and log in again |
| `could not select device driver "nvidia"` | Install the NVIDIA Container Toolkit (Linux) or enable WSL2 GPU (Windows) |
| vLLM runs out of GPU memory | Lower `gpu_memory_utilization` / `max_num_seqs` in the profile |
| DOCX/PPTX files fail | LibreOffice not found: install it, or set `DOCPARSE_SOFFICE=/path/to/soffice` |
| Pinning warning `cannot pin to cores` | Core ids in the profile exceed your CPU: use `generic` |

## Development

```bash
make setup   # uv sync, pre-commit install
make check   # ruff + mypy --strict + pytest + licence gate
```

Permissive dependencies only (Apache, MIT, BSD); the licence gate runs in `make check` and when building the parser image.
