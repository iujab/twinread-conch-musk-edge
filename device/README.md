# TwinRead device app (Arduino UNO Q, offline)

Research prototype - not a diagnostic device. The int8 student runs offline on the board (measured: 18.7 ms per tile at
4 threads, `results/uno_q/onboard/`); a USB camera can film tissue images shown on a screen; the dashboard (127.0.0.1:8080,
forwarded to the tailnet) and the 8x13 LED matrix show Confident / Needs review. The device's two heads share ONE student backbone
(Head A in CONCH-space and Head B in MUSK-space for the CM bundle; space A / space B for OPEN), so they are far less
independent than CONCH and MUSK. Measured on CRC-VAL-HE-7K, the deployed CM int8 flag agrees with the teacher flag at kappa
0.34 on x86 and 0.32 on the board (`results/student/cm_crc/int8_flag_fidelity.json`, `results/uno_q/onboard/testset_score.json`).

- `twinread_runtime.py`: bundle verification (SHA256SUMS), `EdgeModel` (onnxruntime, with `load()`/`unload()`), `tissue_fraction`, `crop_from_frame`, `device_review_rule`.
- `camera.py`: finds the USB camera, keeps the newest frame, survives replug, and really closes the device on `pause()`.
- `demo.py`: the preset slides from `scripts/uno_q/make_demo_slides.py`.
- `uno_q_app.py`: the loop and a stdlib `http.server`, 127.0.0.1 only, no network calls.
- `ui/`: the dashboard. System font, no external assets, light and dark by the viewer's setting.
- `explainer.py`: the Explain button's job and its llama-server (see "Explain").
- `bridge.py`: `send_status(state)` to the MCU, returns False if no Bridge.
- `mcu/twinread_led/twinread_led.ino`: LED icons, Bridge method `set_state(String)`.
- `deploy/`: the systemd unit, the installer and the kiosk session.

Run: `python uno_q_app.py --bundle DIR --mode camera|demo|replay|sim [--standby] [--device auto|/dev/videoN] [--camera-loss wait|replay] [--demo-dir DIR] [--port 8080]`.

## The dashboard

The page is always up on port 8080, at every boot. Only the model and the camera are ever off. At boot both are off, each
starts from its own button, and the same button turns it off again. This protects the board from heat. In standby, Start reading
opens the camera and loads the model with one command.

| Control | What really happens |
| --- | --- |
| Pause model | The ONNX session is dropped: its weights are freed, its thread pool stops, and freed heap goes back to the kernel. No decision is shown while it is off; the last one is only ever mentioned as last. Resume reloads it and verifies the bundle again. |
| Pause camera | The last frame stays on screen and the V4L2 device is closed. The capture thread sleeps on an event, so the sensor is off and USB is not polled. If the model is on, it reads the held frame once. Resume reopens the device. |
| Live camera / Demo slides | Demo slides also close the camera. The slides run by themselves, with previous, next, a pace choice and a row of thumbnails to jump with. Each slide shows its true label beside the model's call. |
| Standby | Both off. The loop wakes every 5 s, and the page polls every 2 s only while someone is viewing it. |

A hidden browser tab stops polling and drops the video feed. The board's own screen (`location.hostname` 127.0.0.1) has no
pointer, so it shows no controls and hides the cursor; tailnet viewers keep both. `kiosk.sh` starts no browser at all unless
a display is connected: with HDMI parked, Chromium cost CPU for nobody and kept the feed encoding at its full rate.

**Nothing a button starts can take the dashboard down.** A bundle that is missing or damaged, a model that cannot load, a
camera that cannot open or a failed read each stay off and the page says what failed and what to do. If the process itself
dies, `Restart=always` brings it back in standby.

`POST /control` takes any of these, as `application/json`; anything else gets 415, which stops a plain form on another site. A command
is checked whole and refused with 400 if any part is wrong or cannot be done, so nothing is half applied.

Every route also checks the `Host` header and answers 403 unless it is an IP address, `localhost`, this machine's short hostname or
`<short hostname>.<anything>.ts.net`, worked out when the app starts. That stops DNS rebinding. There is no login: anyone who can reach
the page can press its buttons, which today means anyone on the tailnet.

| Field | Values |
| --- | --- |
| `model_paused`, `feed_paused` | `true` or `false`. `feed_paused` is the camera, or the slides in demo. |
| `source` | `"camera"` or `"demo"`. Switching starts the new source unless `feed_paused` is also sent. |
| `demo_step` | `-1` or `1`; wraps round the deck. |
| `demo_index` | A slide number. |
| `demo_interval` | Seconds per slide, 1 to 60. |

`/stream.mjpg` sends the boundary and the next part's headers right after each image, because Chromium paints a part only
when the next part's headers arrive and a held frame may have no successor for seconds (a held frame took 5.15 s to appear
without this, 0.11 s with it, in Chromium). So its parts have no `Content-Length`. Verification used Chromium, and Firefox 158 ran the page,
the video and one cold answer (`results/uno_q/onboard/explainer/firefox.json`, earlier code).

Connections are bounded. A socket that stalls for 30 s is dropped, so a short POST body cannot hold a thread and video stops for a viewer who
has stopped taking it. There are at most 4 video streams (a fifth gets 503) and 32 connections.

**Nobody watching.** With no `/state` poll, command or open video for 10 minutes the app goes to plain standby, as at boot: model unloaded,
camera closed, deck stopped, any answer stopped, nothing kept on the page, and one `idle_standby` line in the log (`--idle-standby-min`, 0 means
never). A kiosk page and any visible tab poll, so they count as viewers; a hidden tab does not.

### Install

```
python scripts/uno_q/make_demo_slides.py      # 18 CRC test tiles, random per class with a fixed seed, never filtered by model output
sudo device/deploy/install.sh                 # the service (enabled), the kiosk session, tailscale serve, and the slides if missing
```

`install.sh` uses `bundles/cm` if it has weights, otherwise `bundles/open`. If that has no weights either, as in a public clone, which holds only the
open bundle's config, the install and the dashboard still work and the page says the model cannot load.

`/etc/default/twinread` is created only if missing, with `TWINREAD_ARGS=--standby`. The unit defaults to `--standby` without
that file too. Set `TWINREAD_ARGS=` there to start with both on, though that makes the board hot from power-up.

### Measured on the board

| State | App CPU, % of one core (the board has four) | App memory |
| --- | --- | --- |
| Standby, straight after boot | 0.03 | 57.4 MB |
| Camera and model both on | 110.63 | 137.3 MB |
| Model off, camera on | 41.19 | 109 MB |
| Camera off, model on, frame held | 0 | 138.1 MB |
| Both off again, after the model had been loaded | 0.13 | 103.3 MB |
| Demo slides at the 4 s pace, model on | 8.5 | 134 MB |
| Demo slides, model off | 0.1 | 112.6 MB |
| With one browser viewing: standby | 0.4 | 57.9 MB |
| With one browser viewing: both on | 169.08 | 137.8 MB |
| With one browser viewing: model off, camera on | 120.29 | 109.4 MB |
| Held slide, one browser viewing, no answer | 0.47 | not sampled |
| Held slide, one browser viewing, answer being written | 1.13 | not sampled |
| Camera and model on, no viewer, answer being written | 105.6, and llama-server 191.8 | llama-server 612 MB RSS, 207 MB private |
| Camera and model on, one browser viewing, answer being written | 163.7, and llama-server 146.5 | llama-server 614 MB RSS, 209 MB private |

The first ten rows were measured before the explainer existed (`results/uno_q/onboard/dashboard/`). In the last four, the explainer's own `viewer_cost` and
`worst` come from earlier code, and the no-viewer row from the final code (`final_tandem`).

Memory stays above the fresh figure after an unload because onnxruntime stays imported; a restart returns it. The camera shows
up on USB as `active` while open and `suspended` once closed, and no process holds `/dev/video*`. A held frame or slide was read
once (0 further reads in 15 s, 0 in 5 s). Pause model takes 0.11 s and resume 0.397 s; pause camera 0.072 s, and it is open again 2.694 s after resume.

All 18 demo slides on the board: 17 match the true label, 15 Confident and 3 Needs review. The one wrong call was flagged Needs review.
After `kill -9` the page was back in 3.578 s, in standby. After a reboot it answered over the tailnet 48.4 s after the command, in
standby (`reboot.json` has a note on that figure).

The real camera has not been measured on tissue. It first faced a dark scene, which read `no_tissue`, and later a desk with cables, which read Needs review,
background (`results/uno_q/onboard/explainer/camera_scene_frozen_review.json`). The owner also showed three hand-picked easy tiles on a screen in front of
it (NORM 3775, TUM 3195, ADI 1628) and reports that it went well; the board's log has Confident NORM, TUM and ADI reads consistent with that
(`owner_camera_check_log.json`). That is an informal check: no systematic accuracy was measured, and it is not evidence of camera accuracy.

Evidence and the script that produced it are in `results/uno_q/onboard/dashboard/`.

## Explain

The button "Explain this result" makes the on-board language model explain the decision on the page, either "For a doctor" or "In plain
words". It uses the model and the prompt from `device/llm/README.md`, on the board, with no network. TwinRead can keep reading beside it.

**What the reader sees, one text at a time, labelled by where it came from.** On a click the facts from the reading appear at once
(`explain_prompt.facts_text`). When the language model's first words arrive, its text replaces them, marked as not checked yet. When it
finishes, `explain_check` runs: a pass keeps the answer, and anything else (a failed check, a crash, a hang, too hot, the stop button)
puts the facts back with one plain line saying why. The check compares numbers, classes and flags with the reading and rejects words outside a list. It is
pattern matching, not proof that the wording is right, so the page says only "numbers and class checked against the reading".

**Which decision.** The one its viewer saw when they pressed, with a small copy of the tile. It stays frozen: pausing the model or switching
source lets the answer finish about that snapshot, and the page says when the view has moved on. On demo slides the deck waits for the
answer and then goes on. Changing a slide by hand stops the answer and closes the panel. With the model paused, the button explains the
last call for as long as the app holds it; a new slide or source drops that, and the button then says why it cannot be used. If what the viewer
pressed on is no longer held, the click gets 400 and the page says the result has gone: it never explains a newer one they did not see.

**When it runs.** `explainer.py` starts llama-server on a click, on a background thread, and nothing waits for it: not `/state`, the camera
or TwinRead's reads. There is one job at a time and every viewer sees the same one. The second click within the idle time reuses the warm
server with no prepare step. `--explain-idle` seconds (120) after the last answer the process, the thread and any finished panel are gone.
A loaded idle server used no CPU. The kernel kills llama-server if the app dies, inside or outside systemd (`setpriv --pdeathsig KILL`).

**Heat.** An answer does not start above 68 C and stops above 72 C, on the hottest thermal zone (`--explain-start-max-c`, `--explain-stop-max-c`), and the
page says so. The evidence behind those numbers is thin:

- One sustained run of 337 s with 7 answers and no viewer: peak 68.5 C, minute means 63.3, 65.0, 66.2, 66.4, 66.9 and 66.4 C. That is too short to claim a plateau.
- One run with a browser viewing, 190 s with 2 answers, reached 67.9 C, just under the start limit, so a second click there could be refused. Sustained load with a viewer was never run.
- In the sustained run clicks landed at up to 67.3 C, which is why the start limit is not lower. 72 C is above all of this and 18 C under the kernel's first passive trip at 90 C. Stopping costs only the wording, because the facts stay on the page.
- Both limits have only fired at lowered values, in a second app: refusal at a 30 C limit, and a stop at 61.6 C with a 61 C limit, during preparing. A stop while writing is covered only by a unit test.

| From click, final code | Facts on the page | First word | Finished |
| --- | --- | --- | --- |
| TwinRead paused, cold | 0.02 s | 31.5 s | 55.6 s |
| TwinRead paused, warm | 0.02 s | 9.5 s | 32.9 s |
| TwinRead reading the camera, no viewer, cold | 2.13 s | 39.3 s | 68.6 s |
| TwinRead reading the camera, no viewer, warm | 0.02 s | 10.2 s | 37.8 s |

The first word and finish in the no-viewer rows come from the app's own clock, because the browser left after pressing. The 2.13 s on the cold no-viewer row was one click, cause not found; every other click showed the facts in 0.02 s. With the camera, TwinRead and one browser viewing, on earlier code: cold 49.9 s to the first word and
91.8 s to finish, warm 14.4 s and 52.7 s (`worst.json`).

TwinRead beside it, final code (`final_tandem.json`): reads p50 / p95 / max 116.1 / 125.8 / 141.8 ms with no answer running and 124.4 / 153.5 / 220.3 ms while writing, at 1.98 reads a second, the language model at 3.18 and 3.16 tokens a second, lowest available memory 2796 MB, no swap, peak 63.8 C. These sample `/state` every 0.4 s on the board, so that poller is part of the load.

**How often an answer passes.** On earlier code and the checker of that time, at least 49 of 55 finished answers passed (0.891, Wilson 95% interval 0.782 to 0.949; `answer_stats.json`, `earlier_code`). Four earlier jobs have no stored record. On the final code and the stricter checker, 5 of 6 (`final_code`). Read these with care:

- Only 13 distinct decisions are behind them, and 28 of the answers come from three of those.
- 4 ran outside the service (a made-up disagreement and a blank frame). No slide in the deck has heads that disagree, and no real camera frame gave `no_tissue`.
- "In plain words" on Needs review passed 8 of 12. The sustained run passed 4 of 7.
- This must not sit beside the 0.975 on 200 held-out prompts on x86 as if comparable: that is another sample, on another machine.

The failures were two wrong Head B numbers, one answer that left out the head numbers, one too long, and two rejections of "the fibrous tissue of cancer". Every one fell back to the reading.

**Offline.** With Wi-Fi off and no default route, an answer took 54.7 s (`offline_radio.json`). In a software reboot on kernel 7.0 with the radio off, the page answered at 18.7 s of kernel uptime and an answer finished at 77.5 s (`offline_boot.json`). That uptime excludes firmware and bootloader time, which was not timed; over the tailnet a normal reboot took 49.2 s from the command to the page answering (`end.json`). Both are earlier code, and neither was a power cycle.

`POST /control` takes `{"explain": "doctor"|"user", "explain_decision": N}`, `{"explain_stop": true}` or `{"explain_close": true}`, each on its
own. `explain_decision` is the `explain.decision` from `/state`. Nothing a viewer types reaches the language model: only the audience enum and
the snapshot. llama-server binds to 127.0.0.1 only. `/state` carries `explain`, and the page reads along through its usual polling, at 700 ms
while an answer runs, which cost the app about 0.66 CPU points of one core with a viewer watching.

### Getting the explainer's files onto a board

`install.sh` never copies them and works without them: Explain then shows the reading itself and says why. They are not in git.

```
scripts/llm/build_llama_aarch64.sh --deploy      # device/llm/bin/llama-server
scp data/llm/finetune/twinread-explain-qwen3-v1-Q4_0-embQ8.gguf uno-q:twinread-conch-musk-edge/data/llm/
sha256sum data/llm/twinread-explain-qwen3-v1-Q4_0-embQ8.gguf     # on the board: 84a1d904388270f0b688e1a76632df3d045c4865218346905dfa8ec8ca118bd3
```

`scripts/llm/board.py run` can delete models to stay under its 1 GB limit, so do not run plans through it on a board holding this model.
`scripts/uno_q/push_app.sh` copies the app and page to a board and restarts it.

### Evidence

Everything is in `results/uno_q/onboard/explainer/`, and `summary.json` lists which file came from which code.

- **Final code** (`final_*.json`, each stamped with `device_code_sha256`, and the board's own hash beside it): timing, tandem, the stop button, the kill and hang paths, `kill -9` of the app, the Host check, a result that has gone, Stop past a stalled command, idle-off at a 30 s limit, Start reading.
- **Earlier code**, before the review's fixes and the stricter checker (every other file): the answer sweep, worst case, sustained, heat limits, port in use, missing files, two viewers, click spam, offline, the reboot, screenshots of most panel states and Firefox. These were not repeated.
- **Generators.** `scripts/uno_q/verify_explainer.py` produced them, and `offline_check.py` ran on the board for the offline logs. `phase_install` and `phase_no_tissue_sim` were lost from the script by an edit after they had run, and restored with the same code. Two rows from a real camera frame were first labelled `no_tissue_real_camera`, though they read Needs review, background; they are now `real_camera_frozen_frame`.
- **Network.** `summary.json` > `network_stalls` lists 10 requests from the PC that got no answer in 5 s, and says what they do and do not show.

`tests/test_explainer.py` covers the same paths offline against a small fake server. The fake is for tests only.

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
Under `--camera-loss replay`, a missing camera or one that stops delivering frames falls back to replay (reason shown in the
UI and logged). A bundle that fails checksum verification is refused: the model stays off and the dashboard says so.

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
`figures/ui_device_unoq.png` is this app on the board on demo slide 6 of 18, a real CRC-VAL-HE-7K tile, with the CM bundle. Tissue through the real
camera has only had the informal check described under "Measured on the board" above.

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
