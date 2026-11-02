#!/usr/bin/env bash
# Cross-compile a pinned llama.cpp llama-server on this x86 PC for the UNO Q (4x Cortex-A53, plain NEON, no dotprod/i8mm).
# Compiling here instead of on the board keeps the build's heat off it. The result is one fully static aarch64 binary, so
# the board's glibc version does not matter. Also builds native x86 llama-quantize + the GGUF converter at the same commit
# for the fine-tuning step.
#
# Usage: scripts/llm/build_llama_aarch64.sh [--variant NAME] [--deploy]
#   gcc14a53 (default, the chosen build) GCC 14 -O3, -mtune=cortex-a53, no LTO, OpenMP threads, static
#   gcc14    as gcc14a53 without -mtune; GCC 14 is the compiler the official release used
#   omp      GCC 15 -O3, -mtune=cortex-a53, LTO, OpenMP threads (libgomp, static)
#   noomp    same, but ggml's own threadpool instead of OpenMP
#   generic  -mtune left at the compiler's default, LTO, OpenMP
#   nolto    -mtune left at the compiler's default, no LTO, OpenMP
# Needs: apt install g++-14-aarch64-linux-gnu g++-aarch64-linux-gnu cmake ninja-build (no board access unless --deploy).
# The speed of each variant on the board is in results/llm/report_tables.md ("Binaries").
set -euo pipefail

LLAMA_TAG=v0.8.0
LLAMA_COMMIT=8642541b255bfb5dc39cd351a178be09013a55d6   # official prebuilt at this commit: release b11960
ARCH_FLAGS="armv8-a+crc+crypto"                         # exactly the board's features: fp asimd aes pmull sha1 sha2 crc32

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$REPO/build/llama.cpp-$LLAMA_TAG"
VARIANT=gcc14a53
DEPLOY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --deploy) DEPLOY=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
JOBS="${JOBS:-8}"  # the bench PC is shared: leave half the cores free


if [ ! -d "$SRC/.git" ]; then
  git clone --quiet --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
HEAD="$(git -C "$SRC" rev-parse HEAD)"
[ "$HEAD" = "$LLAMA_COMMIT" ] || { echo "source is $HEAD, expected $LLAMA_COMMIT" >&2; exit 1; }

TUNE="-mtune=cortex-a53"; LTO=ON; CC=aarch64-linux-gnu-gcc; CXX=aarch64-linux-gnu-g++
case "$VARIANT" in
  omp)     OMP=ON ;;
  noomp)   OMP=OFF ;;
  generic) OMP=ON; TUNE="" ;;
  nolto)   OMP=ON; TUNE=""; LTO=OFF ;;
  gcc14)   OMP=ON; TUNE=""; LTO=OFF; CC=aarch64-linux-gnu-gcc-14; CXX=aarch64-linux-gnu-g++-14 ;;
  gcc14a53) OMP=ON; LTO=OFF; CC=aarch64-linux-gnu-gcc-14; CXX=aarch64-linux-gnu-g++-14 ;;
  *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac

command -v "$CXX" >/dev/null || { echo "missing $CXX" >&2; exit 1; }
OUT="$SRC/build-a53-$VARIANT"
cmake -S "$SRC" -B "$OUT" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
  -DCMAKE_C_FLAGS="$TUNE" -DCMAKE_CXX_FLAGS="$TUNE" \
  -DCMAKE_EXE_LINKER_FLAGS="-static" \
  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -DGGML_BACKEND_DL=OFF \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="$ARCH_FLAGS" \
  -DGGML_OPENMP="$OMP" -DGGML_LTO="$LTO" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_LLGUIDANCE=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_COMMIT="${LLAMA_COMMIT:0:12}" \
  >"$OUT.cmake.log" 2>&1 || { tail -30 "$OUT.cmake.log" >&2; exit 1; }
ninja -C "$OUT" -j "$JOBS" llama-server >"$OUT.ninja.log" 2>&1 || { tail -30 "$OUT.ninja.log" >&2; exit 1; }

BIN="$OUT/bin/llama-server"
aarch64-linux-gnu-strip -o "$BIN.stripped" "$BIN"
file "$BIN.stripped"
ls -l "$BIN.stripped"

# Native host tools for the fine-tuning step (GGUF conversion + quantization) at the same commit.
HOST="$SRC/build-host"
if [ ! -x "$HOST/bin/llama-quantize" ]; then
  cmake -S "$SRC" -B "$HOST" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_APP=OFF >"$HOST.cmake.log" 2>&1 || { tail -30 "$HOST.cmake.log" >&2; exit 1; }
  ninja -C "$HOST" -j "$JOBS" llama-quantize >"$HOST.ninja.log" 2>&1 || { tail -30 "$HOST.ninja.log" >&2; exit 1; }
fi
ls -l "$HOST/bin/llama-quantize"

# Provenance record for results/llm (sizes and hashes quoted in the report come from here).
mkdir -p "$REPO/results/llm"
python3 - "$REPO/results/llm/build_$VARIANT.json" "$BIN.stripped" "$LLAMA_TAG" "$LLAMA_COMMIT" "$VARIANT" "$ARCH_FLAGS" "$OMP" "${TUNE:-default}" "$LTO" "$CXX" <<'PY'
import hashlib, json, os, subprocess, sys
out, bin_, tag, commit, variant, arch, omp, tune, lto, cxx = sys.argv[1:11]
h = hashlib.sha256(open(bin_, "rb").read()).hexdigest()
json.dump({"tag": tag, "commit": commit, "variant": variant, "march": arch, "mtune": tune, "openmp": omp,
           "lto": lto, "static": True, "bytes": os.path.getsize(bin_), "sha256": h,
           "compiler": subprocess.check_output([cxx, "--version"], text=True).splitlines()[0],
           "file": subprocess.check_output(["file", "-b", bin_], text=True).strip()}, open(out, "w"), indent=1)
PY

if [ "$DEPLOY" = 1 ]; then
  SSH=(ssh -F "$HOME/.config/twinread/ssh_config" -o BatchMode=yes uno-q)
  NAME=llama-server; [ "$VARIANT" = gcc14a53 ] || NAME="llama-server-$VARIANT"
  "${SSH[@]}" 'mkdir -p ~/twinread-conch-musk-edge/device/llm/bin'
  flock -w 600 "$HOME/.config/twinread/board.lock" scp -q -F "$HOME/.config/twinread/ssh_config" "$BIN.stripped" "uno-q:twinread-conch-musk-edge/device/llm/bin/$NAME"
  "${SSH[@]}" "chmod +x ~/twinread-conch-musk-edge/device/llm/bin/$NAME && ~/twinread-conch-musk-edge/device/llm/bin/$NAME --version 2>&1 | tail -3"
fi
