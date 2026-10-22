# TwinRead device app (Arduino UNO Q, offline)

Research prototype - not a diagnostic device. The int8 student runs offline on the board (measured: 18.7 ms per tile at
4 threads, `results/uno_q/onboard/`); a USB camera can film tissue images shown on a screen (not done yet; replay mode so far); the local UI (127.0.0.1:8080) and the
8x13 LED matrix show AGREE / REVIEW. The device's two "readers" are two heads on ONE shared student backbone (Head A in
CONCH-space and Head B in MUSK-space for the CM bundle; space A / space B for OPEN), so they are far less independent
than CONCH and MUSK; the UI calls them student heads. Measured on CRC-VAL-HE-7K, the deployed CM int8 flag agrees with
the teacher flag at kappa 0.34 on x86 and 0.32 on the board
(`results/student/cm_crc/int8_flag_fidelity.json`, `results/uno_q/onboard/testset_score.json`).

- `twinread_runtime.py`: bundle verification (SHA256SUMS), `EdgeModel` (onnxruntime), `tissue_fraction`, `crop_from_frame`, `device_review_rule`.
- `uno_q_app.py`: loop + stdlib `http.server` (`/`, `/state`, `/frame.jpg`), 127.0.0.1 only, no network calls.
- `bridge.py`: `send_status(state)` to the MCU, returns False if no Bridge.
- `mcu/twinread_led/twinread_led.ino`: LED icons, Bridge method `set_state(String)`.

Run: `python uno_q_app.py --bundle DIR --mode camera|replay|sim [--device /dev/video0] [--replay-dir DIR] [--port 8080]`.

## Offline install on the board (`install_offline.sh`)

1. On the board: `python3 --version` (Debian trixie ships Python 3.13).
2. On a machine with internet, in `device/`: `PYVER=3.13 ./install_offline.sh stage` (`PYVER` also accepts the whole
   `python3 --version` output, e.g. `PYVER="Python 3.13.5"`; default 3.13). It downloads aarch64 wheels into `wheels/`
   and records the Python version in `wheels/PYVER`.
3. Copy `device/` (with `wheels/`) to the board, then on the board: `./install_offline.sh install`. It refuses wheels
   staged for another Python version, then builds `.venv` with `pip install --no-index --find-links wheels`.

Why these flags: current onnxruntime (1.31.0), numpy (2.5.4) and pillow (13.0.0) publish aarch64 wheels only as
`manylinux_2_28` (glibc >= 2.28; trixie has 2.41), so the old `--platform manylinux2014_aarch64` found no onnxruntime at
all. `stage` passes `--platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64` (some dependencies, e.g.
protobuf, still ship manylinux2014) and pins `onnxruntime==1.31.0 numpy==2.5.4 pillow==13.0.0
opencv-python-headless==5.0.0.93`, each checked on PyPI (2026-10-17) to have a cp313 `manylinux_2_28_aarch64` wheel
(opencv as abi3). The stage step was run on this x86 host for Python 3.13 and resolved all four plus protobuf,
flatbuffers and packaging; the install step is unverified on a board.

## Latency harness (`bench_backbones.py`)

`python bench_backbones.py --dir backbones --threads 1 2 4 --out ../results/uno_q/onboard/latency.json` on the board
(done; that file holds the measured rows).
Each (model, threads) runs in a child process; a failing child becomes an error row and the run continues, and the
JSON is rewritten after every row (`"complete": false` until the end), so nothing measured is lost. The headline counts
the file as UNO Q latency only if it says `"simulated": false` with an aarch64 host; anything else stays "pending".
A missing camera, or one that stops delivering frames, falls back to replay (reason shown in the UI and logged).
A bundle that fails checksum verification is refused.

## Decisions
- Review rule: flag if argmax(pA) != argmax(pB), JS(pA,pB) (base 2) > thr.js, sigmoid(review_logit) > thr.review, or no_tissue.
- `tissue_fraction`: a pixel is tissue if OD sum > 0.15 (StainForge rule; OD = -log((I+1)/256) summed over RGB) AND max channel > 30
  (no black bezel/frame) AND chroma (max-min) >= 15 (no grey/dim-white screen with camera noise). `no_tissue` = tissue_fraction < 0.10
  OR pixel std < 6 grey levels. On all 7,180 NCT-CRC test tiles only 9 fall under 0.10, all ADI (near-empty fat), and they go to review.
- Crop: shorter side resized to 224 (bicubic), then centre crop. `--crop marker` tries the rig's marker warp, else centre.

## Bridge (verified on the board)
1. App Lab: `from arduino.app_utils import Bridge; Bridge.call("set_state", state)`.
2. Else a msgpack-RPC request `[0, 1, "set_state", [state]]` to the unix socket `/var/run/arduino-router.sock`
   (override with `ARDUINO_ROUTER_SOCK`). Socket path, framing and reply check are untested off-board.
3. The sketch registers `set_state` with `Bridge.provide`. It is flashed over USB-C with OpenOCD (arduino:zephyr:unoq);
   `set_state` is delivered and an unknown method is refused; one call costs about 8 ms (`results/uno_q/onboard/stage_timing_cm.json`).

## Measured on the board (`results/uno_q/onboard/`, `scripts/uno_q/`)
int8 CM student 18.7 ms per tile at 4 threads (52.8 ms at 1 thread), 34 MB RAM; full CRC-VAL-HE-7K 0.9305 (CM) and 0.9044
(OPEN) balanced accuracy, 7,180 tiles in 282 s; one app frame about 90 ms, half of it the numpy tissue check.
`figures/ui_device_unoq.png` is this app on the board in `--mode replay` with the CM bundle on one real CRC-VAL-HE-7K
tile. Not done: real camera captures.

## Building the bundle

The OPEN bundle's `config.json` and `SHA256SUMS` are committed; `model.onnx`, `probes.npz` and `prompt_bank.npz` are git-ignored
and rebuilt (byte-identical on the build machine: ONNX export, int8 quantisation and the npz writer are deterministic, so the
committed `SHA256SUMS` verifies after a rebuild). Prerequisites:

- `data/crc_train_{x,y,g}.npy`, `data/crc_test_{x,y}.npy` and the cached text banks `data/emb/crc/bank_pathgenclip.npy`,
  `bank_quiltnet_b16.npy`;
- the student `results/student/open_crc/best.pt` and `thresholds.json`;
- `student_targets.npz` and `probes_aligned.npz` (git-ignored; from `twinread.analyze`, e.g. `results/open/OPEN/crc/`).

```
OMP_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/make_bundle.py --pair OPEN --dataset crc \
    --targets results/open/OPEN/crc          # writes bundles/open/ and results/student/open_crc/int8_drift.json (~3 min, CPU)
```

Then `python -c "import sys; sys.path.insert(0,'device'); import twinread_runtime as r; print(r.verify_bundle_dir('bundles/open'))"`.
Thresholds are recalibrated on the int8 model's val outputs, so a changed checkpoint changes `config.json` too.
