# TwinRead device app (Arduino UNO Q, offline)

Research prototype - not a diagnostic device. The int8 student runs offline; a USB camera films tissue images shown on a
screen ("Camera filming a screen (demo)"); the local UI (127.0.0.1:8080) and the 8x13 LED matrix show AGREE / REVIEW.

- `twinread_runtime.py`: bundle verification (SHA256SUMS), `EdgeModel` (onnxruntime), `tissue_fraction`, `crop_from_frame`, `device_review_rule`.
- `uno_q_app.py`: loop + stdlib `http.server` (`/`, `/state`, `/frame.jpg`), 127.0.0.1 only, no network calls.
- `bridge.py`: `send_status(state)` to the MCU, returns False if no Bridge.
- `mcu/twinread_led/twinread_led.ino`: LED icons, Bridge method `set_state(String)`.

Run: `python uno_q_app.py --bundle DIR --mode camera|replay|sim [--device /dev/video0] [--replay-dir DIR] [--port 8080]`.
A missing camera, or one that stops delivering frames, falls back to replay (reason shown in the UI and logged).
A bundle that fails checksum verification is refused.

## Decisions
- Review rule: flag if argmax(pA) != argmax(pB), JS(pA,pB) (base 2) > thr.js, sigmoid(review_logit) > thr.review, or no_tissue.
- `tissue_fraction`: a pixel is tissue if OD sum > 0.15 (StainForge rule; OD = -log((I+1)/256) summed over RGB) AND max channel > 30
  (no black bezel/frame) AND chroma (max-min) >= 15 (no grey/dim-white screen with camera noise). `no_tissue` = tissue_fraction < 0.10
  OR pixel std < 6 grey levels. On all 7,180 NCT-CRC test tiles only 9 fall under 0.10, all ADI (near-empty fat), and they go to review.
- Crop: shorter side resized to 224 (bicubic), then centre crop. `--crop marker` tries the rig's marker warp, else centre.

## Bridge assumptions (verify on the board)
1. App Lab: `from arduino.app_utils import Bridge; Bridge.call("set_state", state)`.
2. Else a msgpack-RPC request `[0, 1, "set_state", [state]]` to the unix socket `/var/run/arduino-router.sock`
   (override with `ARDUINO_ROUTER_SOCK`). Socket path, framing and reply check are untested off-board.
3. The sketch registers `set_state` with `Bridge.provide`. It is compile-verified only (arduino:zephyr:unoq, see
   `results/uno_q/mcu_compile.json`), not flashed. The LED icon shapes have not been seen on hardware.

## Not measured
UNO Q latency (target 250 ms/tile at 4 threads) is not measured; the 20-65 ms in the simulation is an x86 host with a tiny test model.
`figures/ui_device_sim.png` is a simulation with a random test bundle, so its numbers are meaningless.


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
