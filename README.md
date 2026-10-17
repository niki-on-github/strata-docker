# Strata in Docker

[Strata](https://github.com/Niko1221/Strata) (Qwen3.8-Flash-Next MoE on
consumer hardware, MIT) packaged as containers, built for a machine other than
the one they run on: **one CUDA architecture per image**.

| Image tag | GPU | `CUDA_ARCHITECTURES` | Engine |
|---|---|---|---|
| `sm80-<upstream-tag>` | **NVIDIA CMP 170HX / A100** (GA100, compute capability 8.0), CUDA 13 base | `80` | `engine/` |
| `sm61-<upstream-tag>` | **Tesla P40 / P100, GTX 10** (Pascal, 6.1, experimental), CUDA 12 base | `61` | `engine-cuda12/` |

The engine **and** the image encoder are compiled during the image build (a GPU
is not needed to build, only to run), and their source hash is stamped into
`engine/BUILD.json` - which is exactly what `setup.py` checks at start, so a
container start downloads the model and serves; it never compiles.

## CI (GitHub Actions)

A daily cron (this repo, "Build and publish Docker image") resolves upstream's
newest tag and builds and pushes **both** architectures to GHCR as
`ghcr.io/<owner>/<repo>:<arch>-<upstream-tag>` (e.g. `sm80-v0.1.45`) and
`:<arch>-latest`; every run rebuilds both, and each job prints the image's
digest in its summary (pin that digest where you deploy). Manual runs and pushes
to `main` build too.

| Input (`workflow_dispatch`) | Default | Meaning |
|---|---|---|
| `strata_ref` | newest upstream tag | upstream ref/tag to build (must be `v0.1.45` or newer) |

## Run

You need the NVIDIA Container Toolkit and a card matching the image (8.0 for
`sm80`, 6.1 for `sm61`). The model is **not** in the image: a first start picks
one (or takes `MODEL`) and downloads 70-120 GB into `/data`, resumably.

### CMP 170HX (3 cards, `sm80`)

```sh
docker run -d --name strata-170hx \
  --gpus all --ipc host --cap-add IPC_LOCK --ulimit memlock=-1 \
  --stop-timeout 60 \
  -p 8080:8080 \
  -v /nvme/strata:/data \
  -e GPUS=all \
  -e HF_ENDPOINT=https://your-hf-mirror \
  -e API_KEY=changeme \
  ghcr.io/<owner>/strata-docker:sm80-latest
```

### Tesla P40 (`sm61`)

```sh
docker run -d --name strata-p40 \
  --gpus all --ipc host --cap-add IPC_LOCK --ulimit memlock=-1 \
  --stop-timeout 60 \
  -p 8081:8080 \
  -v /nvme/strata:/data \
  -e GPU=0 -e CONTEXT=32768 -e API_KEY=changeme \
  ghcr.io/<owner>/strata-docker:sm61-latest
```

Pascal is upstream-experimental: this image carries the CUDA 12 engine in
`engine-cuda12/` and sets `STRATA_EXPERIMENTAL_SM60=1` itself (a card below
compute capability 7.5 is otherwise refused). DFlash2-style drafting is not
available there. `--ipc host`, `--cap-add IPC_LOCK` and `--ulimit memlock=-1`
are what let the engine lock its tens of GB of expert RAM; `--stop-timeout 60`
gives it time to release them on stop (the default 10 s kills it).

## Environment

Every setting is read at start. A variable that is **not** set is left to
`setup.py`'s own recommendation - including the model, which is why a plain
`docker run` picks and downloads a fitting one.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | *(unset)* | quant to install: `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `UD-Q4_K_XL`, … ([MODELS.md](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)); unset = setup's recommendation |
| `FAMILY` | *(unset)* | model family: `qwen` (default), `swift`, `coder`, `unsloth` |
| `CONTEXT` | *(unset)* | context length; unset = setup's pick for the cards |
| `VISION` | *(unset)* | `no`, `yes` (encoder on the GPU), `cpu` - images |
| `KV` | *(unset)* | KV cache precision: `int8`, `q4_0`, `k8v4` |
| `GPU` / `GPUS` | *(unset)* | one card (`GPU=0`) or several sharing the model (`GPUS=0,2`, `GPUS=all`) - as `nvidia-smi` numbers them |
| `LAYER_SPLIT` | `auto` | first layer of each later card (e.g. `24,36`) |
| `LOW_RAM` | `auto` | `on` maps the experts from the pack instead of keeping them in RAM. `setup.py` reads the **host's** RAM from `/proc/meminfo`, so a memory-limited container has to ask for it |
| `RESIDENT_BUDGET_GIB`, `KV_STREAMING` | *(unset)* | `UD-Q4_K_XL`'s RAM budget, KV streaming (`auto`/`on`/`off`) |
| `GGUF_DIR` | *(unset)* | a mounted folder of GGUF files you already have (no download) |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | listen address and port (the web UI, `/v1`, `/v1/messages`, `/health` all share it) |
| `API_KEY` | *(unset)* | require this key. A first-start choice - change it with `REINSTALL=1` |
| `REINSTALL` | `0` | `1` re-runs the setup pass for an already set-up model (context, vision, KV, host, key) |
| `CONFIG` | *(unset)* | start with this config file instead of the newest on `/data` |
| `STRATA_DATA` | `/data` | where the model files, pack, MTP layer and configs live |
| `HF_ENDPOINT` | `https://huggingface.co` | a Hugging Face mirror (e.g. an in-cluster `olah` cache); upstream reads it itself |
| `STRATA_SOURCE` | *(unset)* | `modelscope` to download from ModelScope instead |
| `STRATA_CA_BUNDLE` | *(unset)* | PEM bundle of extra CA certificates to trust (an internal mirror); also exported as `CURL_CA_BUNDLE`/`SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`. Certificates mounted into `/usr/local/share/ca-certificates/*.crt` are imported at start as well |

Extra arguments after the image name go to the server:
`docker run ... strata:sm80 --gpu 0 --context 65536`.

## Build another arch

```sh
docker build --build-arg STRATA_REF=v0.1.45 --build-arg CUDA_ARCHITECTURES=80 -t strata:sm80 .
docker build --build-arg STRATA_REF=v0.1.45 --build-arg CUDA_ARCHITECTURES=61 \
  --build-arg CUDA_BASE=12.9.1-devel-ubuntu24.04 -t strata:sm61 .
docker build --build-arg CUDA_ARCHITECTURES=89 --build-arg CUDA_BASE=12.9.1-devel-ubuntu24.04 -t strata:ada .
```

Arch map: RTX 20 `75`, RTX 30 `86`, RTX 40 `89`, RTX 50 `120`, A100 / CMP 170HX
`80`, Tesla P40 / P100 `61`, V100 `70` (the last two: CUDA 12 base +
`CUDA_ARCHITECTURES` below 75, which selects the experimental engine). Upstream's
own supported set starts at `75`; `BUILD_VISION=0` skips the image encoder.

Two deliberate differences from upstream's Dockerfile, both in
`tools/docker_build.py`: everything is compiled with `-DSTRATA_PORTABLE=ON`
(upstream builds `native`, i.e. for the machine that *builds* the image - on a
runner with AVX-512 the resulting encoder/engine can stop with an illegal
instruction on an AVX2-only PC, the #411/#412/#419 incident upstream documents),
and a sub-7.5 arch builds the CUDA 12 engine into `engine-cuda12/` with
`-DSTRATA_EXPERIMENTAL_SM60=ON`.

## Files

- `Dockerfile` — clones upstream at `STRATA_REF`, compiles for `CUDA_ARCHITECTURES`
- `tools/docker_build.py` — the compile-only build step (no GPU needed), writes the `BUILD.json` stamp
- `docker-entrypoint.sh` — first-start setup + model download, configs kept on `/data` (derived from upstream's entrypoint: no default model, config discovery, CA bundle)
- `.github/workflows/docker-ghcr.yml` — daily tag-resolve + publish of both architectures

## Credits

All engine/server/dashboard work is upstream [Strata](https://github.com/Niko1221/Strata)
(MIT); this repo only packages it. The model and its quants have their own
licences - see upstream's [MODELS.md](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)
and [HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md#license).
