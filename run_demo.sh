#!/usr/bin/env bash
# TwinRead demo. Research prototype, not a diagnostic device.
#
#   bash run_demo.sh --smoke     # no GPU, no network: headline numbers + figures (to build/smoke_figures/) from the
#                                # committed results JSON, the manifest tests, and a dashboard check on 127.0.0.1:1777
#   bash run_demo.sh             # end to end from a clean clone (steps whose outputs exist are skipped):
#                                #   data -> OPEN embeddings (GPU lock) -> OPEN analysis targets -> OPEN student ->
#                                #   int8 bundle -> per-tile predictions -> headline + figures -> dashboard :1777
#   bash run_demo.sh --with-cm   # also the CONCH + MUSK local subset (needs Hugging Face access to both models)
#   bash run_demo.sh --no-serve  # everything except starting the dashboard
#   bash run_demo.sh --dry-run   # print the plan for the current state: heavy steps (downloads, GPU, analysis,
#                                # training, bundling) are only printed; cheap CPU steps (per-tile export, headline,
#                                # figures) run. Use it to check the ordering on a fresh clone.
#
# Clean-clone notes. results/**/*.npz, *.pt and bundles/open/model.onnx are git-ignored. So:
#  - the committed analysis JSON is never rewritten: when its npz targets are missing they are reproduced into
#    build/reproduce/<pair>/ by scripts/reproduce_analysis.py, which also checks the numbers match the committed JSON;
#  - with no committed checkpoint the OPEN student is retrained into build/reproduce/student/open_crc/ and bundled into
#    build/reproduce/bundle_open/, never over the committed files. On the build machine the retrained student and
#    bundle came out identical to the committed ones (same metrics, same SHA256SUMS); other GPUs may differ;
#  - figures/fig2 keeps its committed version when the tile gallery's inputs are missing.
# The rig page (camera-of-screen capture) is separate: python rig/show_tiles.py  -> http://127.0.0.1:1778/rig
# The device app is built for the Arduino UNO Q; today it runs as an x86 replay (see device/README.md).
set -euo pipefail
P="$(cd "$(dirname "$0")" && pwd)"
B="${BENCH:-$(cd "$P/.." && pwd)}"   # bench root (venv + shared GPU lock); set BENCH when running a clone elsewhere
PY="${PY:-$B/.venv/bin/python}"
LOCK="flock $B/.gpu.lock"
export PYTHONPATH="$P/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$P"

SMOKE=0; WITH_CM=0; SERVE=1; DRY=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --with-cm) WITH_CM=1 ;;
    --no-serve) SERVE=0 ;;
    --dry-run) DRY=1; SERVE=0 ;;
    *) echo "usage: $0 [--smoke] [--with-cm] [--no-serve] [--dry-run]" >&2; exit 2 ;;
  esac
done
say() { echo "[run_demo $(date +%T)] $*"; }
heavy() { if [ "$DRY" = 1 ]; then echo "    [dry-run] would run: $*"; else "$@"; fi; }

if [ "$SMOKE" = 1 ]; then
  export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  mkdir -p build
  say "headline numbers from the committed results JSON"
  $PY -m twinread.headline
  say "figures from JSON -> build/smoke_figures/ (figures/ is left untouched)"
  $PY -m twinread.figures --out build/smoke_figures
  say "manifest tests"
  $PY -m pytest tests/test_manifest.py -q -p no:cacheprovider
  say "dashboard check on 127.0.0.1:1777"
  $PY -m twinread.app_host --port 1777 > build/smoke_dashboard.log 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  for _ in $(seq 1 40); do curl -sf http://127.0.0.1:1777/api/headline > /dev/null && break; sleep 0.25; done
  curl -sf http://127.0.0.1:1777/ | grep -q "not a diagnostic device"
  curl -sf "http://127.0.0.1:1777/api/headline" | $PY -c "import json,sys; d=json.load(sys.stdin); \
print('dashboard OK; UNO Q latency:', d['device']['uno_q_latency'])"
  say "smoke OK"
  exit 0
fi

embed_loop() {  # embed one reader (exit 3 = time budget used -> rerun to resume), each run under the GPU lock
  local r=$1 ds=$2 rc
  while :; do
    set +e; $LOCK $PY -m twinread.embed --readers "$r" --dataset "$ds" --budget 480 > /dev/null; rc=$?; set -e
    [ "$rc" = 3 ] || break
  done
  [ "$rc" = 0 ] || { say "embedding $r/$ds failed (rc=$rc)"; exit "$rc"; }
}
train_loop() {  # train_student.py exits 3 when its time budget is used -> rerun to resume
  local rc
  while :; do
    set +e; $LOCK $PY scripts/train_student.py "$@" --budget 540; rc=$?; set -e
    [ "$rc" = 3 ] || break
  done
  [ "$rc" = 0 ] || { say "student training failed (rc=$rc)"; exit "$rc"; }
}
have_emb() {  # have_emb <dataset> <reader...>: all six embedding files of every reader
  local ds=$1 r n; shift
  for r in "$@"; do
    n=$(ls data/emb/$ds/${r}_{aligned,probe}_{train,val,test}.npy 2>/dev/null | wc -l)
    [ "$n" = 6 ] || return 1
  done
}

say "1/8 data (NCT-CRC-HE 20k train subset + CRC-VAL-HE-7K test, Camelyon17-WILDS 7.5k subset; HF, cached in data/)"
[ -f data/crc_test_x.npy ] || heavy $PY -c "from pathlib import Path; from twinread.data import fetch_crc; fetch_crc(Path('data'))"
[ -f data/cam_x.npy ] || heavy $PY -c "from pathlib import Path; from twinread.data import fetch_camelyon; fetch_camelyon(Path('data'))"

say "2/8 OPEN pair embeddings (PathGen-CLIP, QuiltNet-B-16, BiomedCLIP), GPU lock"
for ds in crc cam; do for r in pathgenclip quiltnet_b16 biomedclip; do
  have_emb $ds $r && echo "    cached: $r/$ds" || heavy embed_loop "$r" "$ds"
done; done

say "3/8 OPEN analysis targets (pre-registered; CPU)"
if [ ! -f results/open/OPEN/crc/review.json ]; then
  heavy bash scripts/run_open.sh           # first run on this machine: writes results/open/ (pre-registration first)
  TGT_OPEN=results/open/OPEN/crc
elif [ -f results/open/OPEN/crc/student_targets.npz ]; then
  echo "    targets present: results/open/OPEN/crc"
  TGT_OPEN=results/open/OPEN/crc
else
  TGT_OPEN=build/reproduce/open/OPEN/crc  # clean clone: JSON committed, npz git-ignored -> reproduce, never overwrite
  [ -f $TGT_OPEN/student_targets.npz ] && echo "    reproduced targets present: $TGT_OPEN" || \
    heavy $PY scripts/reproduce_analysis.py --pair OPEN --dataset crc --committed results/open --out build/reproduce/open
fi

say "4/8 OPEN student (mobilenetv4_conv_small, K2 augmentation)"
if [ -f results/student/open_crc/best.pt ]; then
  STU=results/student/open_crc; echo "    committed-run checkpoint present: $STU/best.pt"
else
  STU=build/reproduce/student/open_crc
  [ -f $STU/best.pt ] && echo "    retrained checkpoint present: $STU/best.pt" || \
    heavy train_loop --pair OPEN --dataset crc --aug k2 --targets-dir "$TGT_OPEN" --out "$STU"
fi

say "5/8 OPEN int8 bundle"
if [ -f bundles/open/model.onnx ]; then
  echo "    bundle present: bundles/open/ (verify: device/README.md)"
elif [ "$STU" = results/student/open_crc ]; then
  heavy $PY scripts/make_bundle.py --pair OPEN --dataset crc --targets "$TGT_OPEN"   # rebuilds the committed bundle
  heavy $PY scripts/int8_flag_fidelity.py --pair OPEN   # keep the fidelity JSON in step with the rebuilt bundle
else
  [ -f build/reproduce/bundle_open/model.onnx ] && echo "    bundle present: build/reproduce/bundle_open/" || \
    heavy $PY scripts/make_bundle.py --pair OPEN --dataset crc --student "$STU" --targets "$TGT_OPEN" \
      --out build/reproduce/bundle_open
fi

say "6/8 CONCH + MUSK"
if [ "$WITH_CM" = 1 ]; then
  for ds in cam crc; do have_emb $ds conch musk && echo "    cached: conch+musk/$ds" || heavy bash scripts/run_cm_local.sh embed "$ds"; done
  [ -f results/cm_local/CM/crc/review.json ] && [ -f results/cm_local/CM/cam/review.json ] || heavy bash scripts/run_cm_local.sh analyze
else
  echo "    skipped (pass --with-cm; needs HF access to MahmoodLab/CONCH and xiangjx/musk); committed results/cm_local/ JSON is used"
fi
for ds in crc cam; do  # per-tile predictions for the triage curves, the gallery and the tile viewer (CPU, ~10 s)
  if [ -f results/cm_local/CM/$ds/tile_preds.npz ]; then echo "    tile predictions present: $ds"
  elif have_emb $ds conch musk; then CUDA_VISIBLE_DEVICES="" $PY scripts/export_tile_preds.py --datasets $ds
  else echo "    tile predictions for $ds need the CONCH/MUSK embeddings (--with-cm); the tile viewer will say so"; fi
done

say "7/8 headline numbers + figures (fig2 keeps its committed version if the gallery inputs are missing)"
$PY -m twinread.figures

if [ "$SERVE" = 1 ]; then
  say "8/8 dashboard on http://127.0.0.1:1777 (Ctrl-C to stop)"
  exec $PY -m twinread.app_host --port 1777
fi
say "done$([ "$DRY" = 1 ] && echo ' (dry run: heavy steps were only printed)')"
