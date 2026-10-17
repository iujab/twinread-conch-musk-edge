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
- `tissue_fraction`: fraction of pixels with OD sum > 0.15, OD = -log((I+1)/256) summed over RGB (StainForge rule).
  `no_tissue` = tissue_fraction < 0.10 OR pixel std < 6 grey levels. The std test is needed because an all-black tile has
  OD far above 0.15 and would otherwise count as tissue.
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
