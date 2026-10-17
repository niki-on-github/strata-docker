#!/usr/bin/env python3
"""Compile the Strata engine (and the image encoder) inside the Docker build.

Same flow as upstream's Dockerfile build step, driven by the CUDA_ARCHITECTURES
build arg, with three differences:

  * the arch is always explicit and pinned (the engine is a fat binary; this repo
    publishes one architecture per image, like project-maya-docker);

  * -DSTRATA_PORTABLE=ON for the engine AND the encoder.  Without it ggml-cpu is
    compiled for the machine that BUILDS the image (GGML_NATIVE): GitHub's runners
    have AVX-512, so a "native" image dies with `illegal instruction` on a PC that
    has only AVX2.  Upstream records that incident in tools/vision/CMakeLists.txt
    (#411 #412 #419) - and upstream's Dockerfile still passes -DSTRATA_PORTABLE=OFF
    for the encoder, so we override it here;

  * a card below compute capability 7.5 (Pascal, Volta) builds the experimental
    CUDA 12 engine into engine-cuda12/ with -DSTRATA_EXPERIMENTAL_SM60=ON
    (docs/OLDER_GPUS.md) - that build needs a CUDA 12.x base image, since CUDA 13
    dropped sm_60 and sm_70.

The BUILD.json written here is the stamp setup.py's engine_ok/vision_ok check at
container start, so the first start reuses this build instead of compiling it.
"""
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup as S  # noqa: E402


def env(name, default=""):
    return (os.environ.get(name) or default).strip()


def archs() -> list:
    raw = env("CUDA_ARCHITECTURES", "80").replace(",", ";")
    try:
        out = sorted({int(x) for x in raw.split(";") if x.strip()})
    except ValueError:
        sys.exit(f"CUDA_ARCHITECTURES {raw!r} is not a list of compute capabilities (e.g. 80, or 61)")
    if not out:
        sys.exit("CUDA_ARCHITECTURES must name at least one architecture (e.g. 80)")
    if out[0] < 60:
        sys.exit(f"CUDA_ARCHITECTURES {out}: even the experimental build needs compute capability 6.0 (sm_60)")
    return out


def main():
    arch = archs()
    toolkit = 12 if arch[0] < 75 else 13          # below sm_75: the experimental CUDA 12 engine
    vision = "gpu" if env("BUILD_VISION", "1") == "1" else "none"

    if toolkit == 12 and not hasattr(S, "ENGINE12_DIR"):
        sys.exit(f"this source has no experimental CUDA 12 engine folder (setup.ENGINE12_DIR), so it cannot "
                 f"build for sm_{arch[0]}: pin STRATA_REF to a newer release (docs/OLDER_GPUS.md)")

    nvcc = None
    try:
        nvcc, _ = S.find_nvcc()
    except Exception:                             # noqa: BLE001 - an older/newer setup.py: fall back to PATH
        pass
    nvcc = nvcc or shutil.which("nvcc")
    if not nvcc:
        sys.exit("nvcc not found: the base image must be a CUDA *devel* image (CUDA_BASE)")

    # llama.cpp at upstream's pinned commit, into third_party/ - part of the source
    # hash below, so it has to come first (upstream's Dockerfile does the same).
    llama = S.get_llama_cpp()
    cuda_archs = ";".join(str(x) for x in arch)

    engine = S.engine_dir(toolkit)                # engine/ (CUDA 13) or engine-cuda12/
    bdir = ROOT / ("build-cuda12" if toolkit == 12 else "build")
    vdir = ROOT / ("build-vision-cuda12" if toolkit == 12 else "build-vision")

    print(f"  Compiling the Strata engine for " + ", ".join(f"sm_{x}" for x in arch) +
          f" (CUDA {toolkit}, portable, vision: {vision}) ...")
    S.cmake_build(ROOT, bdir, "strata",
                  ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
                   "-DSTRATA_Q6K_EXPERTS=ON", "-DSTRATA_MMQ_KQUANTS=ON",
                   "-DSTRATA_PORTABLE=ON",
                   f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
                   f"-DCMAKE_CUDA_COMPILER={nvcc}",
                   f"-DSTRATA_GGML_DIR={llama}",
                   *S.engine_defs(arch, toolkit)],
                  None, "build-strata-cuda12.bat" if toolkit == 12 else "build-strata.bat")

    engine.mkdir(exist_ok=True)
    shutil.copy2(bdir / S.EXE, engine / S.EXE)
    if (bdir / S.KQ_LIB).exists():                # K-quant MMQ prompt library (Unsloth UD-Q4/Q5/Q6_K_XL)
        shutil.copy2(bdir / S.KQ_LIB, engine / S.KQ_LIB)

    if vision != "none":
        S.cmake_build(ROOT / "tools" / "vision", vdir, "strata-vision",
                      [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON",
                       "-DSTRATA_PORTABLE=ON",
                       f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
                       f"-DCMAKE_CUDA_COMPILER={nvcc}"],
                      None, "build-vision-cuda12.bat" if toolkit == 12 else "build-vision.bat")
        shutil.copy2(vdir / "bin" / S.VEXE, engine / S.VEXE)

    # The stamp setup.py reads: the same keys upstream's own compile path writes, so
    # the container start reuses this engine (and knows which toolkit it is).
    bindir = pathlib.Path(nvcc).parent
    (engine / "BUILD.json").write_text(json.dumps(
        {"source": "local", "version": S.source_version(), "archs": arch, "vision": vision,
         **({"toolkit": 12} if toolkit == 12 else {}),
         "cuda_dirs": [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()],
         "src": S.source_hash(S.ENGINE_SOURCES),
         "vision_src": S.source_hash(S.VISION_SOURCES) if vision != "none" else None}, indent=1),
        encoding="utf-8")

    # upstream's own runtime checks, run here so a broken image is not published
    got = S.engine_archs(toolkit)
    if not got or list(got[0]) != arch:
        sys.exit(f"{engine}/BUILD.json lists {got} instead of {arch}")
    for a in arch:
        if not S.engine_runs_on({"arch": a}, toolkit):
            sys.exit(f"the engine has no code for sm_{a} ({got})")
    for want in ([S.EXE] + ([S.VEXE] if vision != "none" else [])):
        if not (engine / want).exists():
            sys.exit(f"{engine / want} is missing after the build")
    if S.stale_binary(engine / S.EXE):
        sys.exit(f"the built engine is version {S.binary_engine_version(engine / S.EXE)}, "
                 f"this source is {S.source_version()}")
    print(f"  engine built: {engine / S.EXE} (sm_{', sm_'.join(str(x) for x in arch)}, CUDA {toolkit})")


if __name__ == "__main__":
    main()
