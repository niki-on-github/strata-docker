#!/bin/sh
# Strata in Docker - entry point.
#
# Derived from upstream's docker-entrypoint.sh (Niko1221/Strata; this image pins
# the ref in STRATA_REF), with three differences:
#   1. no default model or context: anything not given as an env var is left to
#      setup.py's own recommendation, so a plain `docker run` picks the model that
#      fits the machine and downloads it (resumably);
#   2. with MODEL unset the config's name is not known up front, so the config a
#      first start writes is picked up from its folder afterwards and kept on /data;
#   3. STRATA_CA_BUNDLE, and certificates mounted in /usr/local/share/ca-certificates,
#      are trusted - the convenience project-maya-docker offers for an internal
#      mirror.  HF_ENDPOINT needs no help: setup.py reads it itself.
#
# The engine is compiled into the image, so a first start only downloads the model
# (70-120 GB, see docs/MODELS.md) and prepares its pack.
set -e
cd /opt/strata || exit 1

STRATA_DATA="${STRATA_DATA:-/data}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
FAMILY="${FAMILY:-}"
MODEL="${MODEL:-}"
CONTEXT="${CONTEXT:-}"
VISION="${VISION:-}"
KV="${KV:-}"
GPU="${GPU:-}"
GPUS="${GPUS:-}"
LAYER_SPLIT="${LAYER_SPLIT:-}"
LOW_RAM="${LOW_RAM:-auto}"
GGUF_DIR="${GGUF_DIR:-}"
RESIDENT_BUDGET_GIB="${RESIDENT_BUDGET_GIB:-}"
KV_STREAMING="${KV_STREAMING:-}"
API_KEY="${API_KEY:-}"
CONFIG="${CONFIG:-}"
REINSTALL="${REINSTALL:-0}"

# `docker run <image> --flag ...` passes server arguments through.
extra="$(printf ' %s' "$@")"

# Upstream's warning (#1584): the engine in this image is an AVX2 build, and a host
# CPU without AVX2 runs it into an illegal instruction (exit 132).
cpuinfo="${STRATA_CPUINFO:-/proc/cpuinfo}"
if [ -r "$cpuinfo" ] && grep -q '^flags' "$cpuinfo" && ! grep -m1 '^flags' "$cpuinfo" | grep -qw avx2; then
  echo "WARNING: this host CPU has no AVX2 - the engine in this image may stop with an illegal instruction." >&2
  echo "  Build the image on that machine (docs/OLDER_GPUS.md, the older-CPU build)." >&2
fi

# A card below compute capability 7.5 (Pascal, Volta) runs only with the
# experimental CUDA 12 engine this image carries (engine-cuda12/, CUDA_ARCHITECTURES
# < 75 at build time): setup.py needs STRATA_EXPERIMENTAL_SM60=1 to admit it.
arch=$(printf '%s\n' "${CUDA_ARCHITECTURES:-}" | tr ',;' '\n\n' | grep -oE '^[0-9]+' | sort -n | head -1)
if [ -n "$arch" ] && [ "$arch" -lt 75 ] && [ -z "${STRATA_EXPERIMENTAL_SM60:-}" ]; then
  export STRATA_EXPERIMENTAL_SM60=1
fi

# Trust an internal mirror's CA (like project-maya-docker).
if ls /usr/local/share/ca-certificates/*.crt >/dev/null 2>&1; then
  update-ca-certificates >/dev/null 2>&1 || echo "warning: update-ca-certificates failed" >&2
fi
if [ -n "${STRATA_CA_BUNDLE:-}" ]; then
  export CURL_CA_BUNDLE="$STRATA_CA_BUNDLE" \
         SSL_CERT_FILE="$STRATA_CA_BUNDLE" \
         REQUESTS_CA_BUNDLE="$STRATA_CA_BUNDLE"
fi

mkdir -p "$STRATA_DATA/config"

# The newest config kept on the volume = what an earlier start set up.
newest_config() {
  ls -t "$STRATA_DATA"/config/strata-*.json 2>/dev/null \
    | grep -v '\.shared-settings\.json$' | head -1
}

# Make $1 the config the server starts with: a copy on the volume, linked from the
# program folder (setup.py starts the newest strata-*.json it finds there).
use_config() {
  cp -f "$1" "$STRATA_DATA/config/$(basename "$1")"
  ln -sfn "$STRATA_DATA/config/$(basename "$1")" "/opt/strata/$(basename "$1")"
}

if [ -n "$CONFIG" ]; then
  [ -f "$CONFIG" ] || { echo "CONFIG=$CONFIG does not exist." >&2; exit 1; }
  use_config "$CONFIG"
  echo "Config: $CONFIG (from CONFIG)"
elif [ "$REINSTALL" != "1" ] && [ -n "$(newest_config)" ]; then
  cfg="$(newest_config)"
  ln -sfn "$cfg" "/opt/strata/$(basename "$cfg")"
  echo "Config: $cfg (already set up on $STRATA_DATA)"
else
  echo "== Setting up Strata: a stopped model download resumes here =="
  set -- --setup --yes --no-start --data-dir "$STRATA_DATA" \
    --host "$HOST" --port "$PORT" --low-ram "$LOW_RAM"
  [ -n "$FAMILY" ] && set -- "$@" --family "$FAMILY"
  [ -n "$MODEL" ] && set -- "$@" --model "$MODEL"
  [ -n "$CONTEXT" ] && set -- "$@" --context "$CONTEXT"
  [ -n "$VISION" ] && set -- "$@" --vision "$VISION"
  [ -n "$KV" ] && set -- "$@" --kv "$KV"
  [ -n "$GPUS" ] && set -- "$@" --gpus "$GPUS"
  [ -n "$GPU" ] && set -- "$@" --gpu "$GPU"
  [ -n "$LAYER_SPLIT" ] && set -- "$@" --layer-split "$LAYER_SPLIT"
  [ -n "$GGUF_DIR" ] && set -- "$@" --gguf-dir "$GGUF_DIR"
  [ -n "$RESIDENT_BUDGET_GIB" ] && set -- "$@" --resident-budget-gib "$RESIDENT_BUDGET_GIB"
  [ -n "$KV_STREAMING" ] && set -- "$@" --kv-streaming "$KV_STREAMING"
  .venv/bin/python setup.py "$@"
fi

# A setup pass writes its config beside the program (setup.py: strata-<model>.json);
# keep every one of them on the volume and link the newest back, so the next start
# takes the "already set up" branch above.
if [ "$REINSTALL" = "1" ] || [ -z "$(newest_config)" ]; then
  found=0
  for f in /opt/strata/strata-*.json; do
    [ -f "$f" ] && [ ! -L "$f" ] || continue    # a link already points into the volume
    case "$f" in *.shared-settings.json) continue ;; esac
    use_config "$f"
  done
  # A setup pass that wrote through an existing link (REINSTALL=1) leaves no new file
  # behind, so ask the volume rather than the loop.
  cfg="$(newest_config)"
  [ -n "$cfg" ] || { echo "the setup pass installed no model/config - see the log above" >&2; exit 1; }
  echo "Config: $cfg (just set up)"
fi

# Start.  GPUS / GPU / LAYER_SPLIT are repeated on purpose (as upstream does): given
# at a start they pin the cards for this model, so a config that names one card is
# not offered a pair again.
set -- --port "$PORT"
[ -n "$GPUS" ] && set -- "$@" --gpus "$GPUS"
[ -n "$GPU" ] && set -- "$@" --gpu "$GPU"
[ -n "$LAYER_SPLIT" ] && set -- "$@" --layer-split "$LAYER_SPLIT"
# API_KEY is a first-start choice (it goes into the config); changing it later
# needs REINSTALL=1, exactly as upstream has it.
# shellcheck disable=SC2086   # the extra args of `docker run <image> ...`, split on purpose
exec .venv/bin/python setup.py "$@" $extra
