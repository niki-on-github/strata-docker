# syntax=docker/dockerfile:1
#
# Strata (https://github.com/Niko1221/Strata) in a container - one CUDA
# architecture per image, built for a machine other than the one it runs on
# (tools/docker_build.py builds it "portable", i.e. AVX2 baseline):
#
#   ghcr.io/niki-on-github/strata-docker:sm80-<tag>  CMP 170HX / A100 (GA100, sm_80), CUDA 13
#   ghcr.io/niki-on-github/strata-docker:sm61-<tag>  Tesla P40 / P100, GTX 10 (Pascal, sm_61, experimental), CUDA 12
#
# The engine AND the image encoder are compiled during the image build, so the
# build needs a CUDA *devel* base but NO GPU.  Their source hash is stamped into
# engine/BUILD.json, which setup.py checks at start: a container start downloads
# the model and serves, it never compiles.
#
#   docker build --build-arg STRATA_REF=v0.2.5 -t strata:sm80 .
#   docker build --build-arg CUDA_ARCHITECTURES=61 \
#                --build-arg CUDA_BASE=12.9.1-devel-ubuntu24.04 -t strata:sm61 .
#
# CUDA 13 dropped Pascal and Volta: a card below compute capability 7.5 needs the
# CUDA 12 base image, and its engine is the experimental one
# (-DSTRATA_EXPERIMENTAL_SM60=ON) in engine-cuda12/ - docs/OLDER_GPUS.md.

ARG CUDA_BASE=13.0.0-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_BASE}

ARG STRATA_REF=v0.2.5
ARG CUDA_ARCHITECTURES=80
ARG BUILD_VISION=1

# STRATA_EXECV=1: setup.py replaces itself with the server, so the server is PID 1
# and a stop signal reaches it.  It releases tens of GB of page-locked RAM then,
# which takes longer than the default 10 s - run with --stop-timeout 60.
# CUDA_ARCHITECTURES stays in the image: docker-entrypoint.sh reads it to decide
# whether the experimental CUDA 12 engine and a pre-7.5 card are admitted.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    STRATA_EXECV=1 \
    CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}

# Upstream's exact build list (its own Dockerfile), plus python3-pip for the venv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "$STRATA_REF" --depth 1 \
        https://github.com/Niko1221/Strata.git /opt/strata \
    && printf '%s\n' "$STRATA_REF" > /opt/strata/.reference

WORKDIR /opt/strata

# cmake and ninja come from requirements-build.txt (pinned, into .venv; upstream's
# Dockerfile does the same), so no apt build tools beyond the compiler are needed.
# Older refs (before 0.2) kept them in requirements.txt instead.
RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && if [ -f requirements-build.txt ]; then .venv/bin/pip install --no-cache-dir -r requirements-build.txt; fi \
    && chmod +x setup.sh

COPY tools/docker_build.py tools/docker_build.py

# Compile the engine (and the image encoder) for CUDA_ARCHITECTURES and write the
# BUILD.json stamp.  The build folders are build-time only; third_party/ MUST stay
# (it is part of the source hash setup.py recomputes at container start).
RUN .venv/bin/python tools/docker_build.py \
    && rm -rf build build-vision build-cuda12 build-vision-cuda12

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint \
    && mkdir -p /data/config /data/home

# HOME on the volume keeps anything the setup writes in $HOME out of the image
# layer; STRATA_DATA is where the model files, the pack, the MTP layer and the
# configs go (docker-entrypoint.sh defaults its --data-dir to it).
ENV HOME=/data/home \
    STRATA_DATA=/data

VOLUME ["/data"]

EXPOSE 8080

ENTRYPOINT ["docker-entrypoint"]
