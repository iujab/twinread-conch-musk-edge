#!/usr/bin/env bash
# TwinRead demo. Research prototype, not a diagnostic device.
#
#   bash run_demo.sh --smoke     # no GPU, no network: headline numbers + figures (to build/smoke_figures/) from the
#                                # committed results JSON, the manifest tests, and a dashboard check on 127.0.0.1:1777
#   bash run_demo.sh             # end to end: data -> OPEN embeddings (GPU lock) -> analysis -> student -> bundle ->
#                                # figures -> dashboard on http://127.0.0.1:1777 (Ctrl-C to stop). Steps whose
#                                # outputs exist are skipped, so a rerun only serves the dashboard.
#   bash run_demo.sh --with-cm   # also the CONCH + MUSK local-subset run (needs Hugging Face access to both models)
#   bash run_demo.sh --no-serve  # everything except starting the dashboard
#
# The rig page (camera-of-screen capture) is separate: python rig/show_tiles.py  -> http://127.0.0.1:1778/rig
# The device app runs on the Arduino UNO Q: see device/README.md (x86 replay: python device/uno_q_app.py --mode replay).
set -euo pipefail
P="$(cd "$(dirname "$0")" && pwd)"
B="$(cd "$P/.." && pwd)"
PY="${PY:-$B/.venv/bin/python}"
LOCK="flock $B/.gpu.lock"
export PYTHONPATH="$P/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$P"

SMOKE=0; WITH_CM=0; SERVE=1
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --with-cm) WITH_CM=1 ;;
    --no-serve) SERVE=0 ;;
    *) echo "usage: $0 [--smoke] [--with-cm] [--no-serve]" >&2; exit 2 ;;
  esac
done
say() { echo "[run_demo $(date +%T)] $*"; }

if [ "$SMOKE" = 1 ]; then
  export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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

say "1/7 data (NCT-CRC-HE 20k train subset + CRC-VAL-HE-7K test, Camelyon17-WILDS 7.5k subset; HF, cached in data/)"
[ -f data/crc_test_x.npy ] || $PY -c "from pathlib import Path; from twinread.data import fetch_crc; fetch_crc(Path('data'))"
[ -f data/cam_x.npy ] || $PY -c "from pathlib import Path; from twinread.data import fetch_camelyon; fetch_camelyon(Path('data'))"

say "2/7 OPEN pair embeddings (PathGen-CLIP, QuiltNet-B-16, BiomedCLIP), GPU lock, cached"
for ds in crc cam; do for r in pathgenclip quiltnet_b16 biomedclip; do
  n=$(ls data/emb/$ds/${r}_{aligned,probe}_{train,val,test}.npy 2>/dev/null | wc -l)
  [ "$n" = 6 ] || embed_loop "$r" "$ds"
done; done

say "3/7 OPEN analysis (pre-registration, then test once; CPU)"
bash scripts/run_open.sh

if [ "$WITH_CM" = 1 ]; then
  say "4/7 CONCH + MUSK local subset (needs HF access to MahmoodLab/CONCH and xiangjx/musk)"
  for ds in cam crc; do bash scripts/run_cm_local.sh embed "$ds"; done
  bash scripts/run_cm_local.sh analyze
else
  say "4/7 CONCH + MUSK skipped (pass --with-cm); the committed results/cm_local/ JSON is used"
fi
if ls data/emb/crc/conch_probe_test.npy data/emb/crc/musk_probe_test.npy > /dev/null 2>&1; then
  for ds in crc cam; do
    [ -f results/cm_local/CM/$ds/tile_preds.npz ] || CUDA_VISIBLE_DEVICES="" $PY scripts/export_tile_preds.py --datasets $ds
  done
else
  say "    tile viewer: per-tile predictions need the CONCH/MUSK embeddings (--with-cm); the viewer will say so"
fi

say "5/7 OPEN student (K2 augmentation) + int8 bundle (skipped when present)"
if [ ! -f results/student/open_crc/metrics.json ]; then
  while :; do
    set +e; $LOCK $PY scripts/train_student.py --pair OPEN --dataset crc --aug k2 --budget 540; rc=$?; set -e
    [ "$rc" = 3 ] || break
  done
  [ "$rc" = 0 ] || exit "$rc"
fi
[ -f bundles/open/model.onnx ] || $PY scripts/make_bundle.py --pair OPEN --dataset crc --targets results/open/OPEN/crc

say "6/7 headline numbers + figures"
$PY -m twinread.figures

if [ "$SERVE" = 1 ]; then
  say "7/7 dashboard on http://127.0.0.1:1777 (Ctrl-C to stop)"
  exec $PY -m twinread.app_host --port 1777
fi
say "done"
