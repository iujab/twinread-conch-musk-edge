# TwinRead — status and what's left (2026-10-17)

TwinRead uses CONCH and MUSK as two independent readers of every pathology tile. Fusing them makes the call more
accurate, and the fused reader's uncertainty flags tiles for a second look. Custom CUDA kernels speed up
preprocessing (K1) and simulate a camera filming a screen (K2). A 6 MB int8 student carries the idea to an offline
Arduino UNO Q.

This is a research prototype, not a diagnostic device. The judge-facing write-up is `README.md`; this file is the
working handoff.

## Where things stand

| Area | State | Evidence |
|---|---|---|
| Readers: CONCH, MUSK + the open pair PathGen-CLIP/QuiltNet-B-16 | done, local subset | `results/cm_local/`, `results/open/` |
| Fusion, review flag, pre-registration, bootstrap CIs | done, local subset | `results/headline.json` |
| K1 dual-reader preprocessing kernel (CUDA/Triton/torch) | done, bit-exact | `results/kernels/` |
| K2 camera-of-screen augmentation kernel | done | `results/kernels/`, `figures/k2_severity_grid.png` |
| Edge student + int8 ONNX bundles | done (CM bundle local only) | `results/student/`, `bundles/open/` |
| Device app (onnxruntime, offline, LED bridge) | done, x86 replay only | `device/`, `figures/ui_device_sim.png` |
| MCU LED sketch | compiled, not flashed | `results/uno_q/mcu_compile.json` |
| Rig (marker tiles shown on a screen, camera capture) | simulation only | `results/rig/decode_stress.json` |
| NCSA Delta kit (J1-J4, CPU analysis job, runbook) | ready, **not run** | `ncsa/README.md` |
| UNO Q bring-up (Task 15) | **pending: needs the board** | — |
| Delta full-data runs (Task 17) | **pending: needs SSH + per-command OK** | — |

Test suites: CPU `271 passed, 62 skipped` (about 70 s). GPU, under the lock with `TWINREAD_GPU_TESTS=1`: `62 passed`.

## Headline numbers so far

These come from a local subset: 20k CRC training tiles, the full CRC-VAL-HE-7K test set, and 7.5k Camelyon17 tiles.
Every value is copied from `results/headline.json`.

- **Fusion (CRC):** balanced accuracy CONCH 0.9413, MUSK 0.9480, fused 0.9555, a gain of +0.74 pts. Its tile-level
  CI, +0.18 to +1.29, is anti-conservative. Most of the gain is tumour recall (+3.6 pts), because the large
  muscle/stroma swings cancel. Fusion gives no gain on Camelyon17 (−0.40 pts) and none for the open pair.
- **Review flag (CRC, 10% budget):** the fused reader's uncertainty catches 80.3% of its errors. MUSK flagging itself
  catches 74.1% and CONCH 66.0%. Plain disagreement catches 59.0%, and adding disagreement to the uncertainty adds
  nothing.
- **K1:** bit-exact against the official transforms. It runs at 90,131 tiles/s with the input already on the GPU
  (46,157 including the host copy). In the real embedding run it speeds up the pipeline only ×1.11, because the model
  forward takes 89.4% of the time.
- **K2:** 46,315 images/s, and ×10.6 faster than torch inside student training.
- **Student:** 5.65 M parameters, 6.0 MB in int8. Balanced accuracy is 0.9415 in fp32 (ties CONCH) and 0.9219 in
  int8, below every teacher.
- **Device flag:** the int8 device flag matches the teacher flag at kappa 0.34, catching 33.3% of the tiles the
  teachers flag (fp32: 0.42 and 41.0%). On the device the two "readers" are two heads of one network, so say so.

## To do, in order

### 1. NCSA Delta (hack days, Oct 17-18). Every command needs the user's explicit OK first.
Follow `ncsa/README.md` step by step:
1. rsync the project. Its excludes already drop `.worktrees`, `bundles`, `results` and `build`.
2. Run `ncsa/setup_login.sh`. It creates `logs/` and installs pytest.
3. On the login node, run `hf auth login`, then `ncsa/download_weights.py`.
4. Run `ncsa/stage_data.sh`. If staging is needed, run `stage_crc.sbatch`.
5. Run J1 (`j1_check.sbatch`, 30 min).
   - It builds the sm_80 kernels and runs the K1/K2 GPU tests and a short benchmark.
   - If it reports no `nvcc`, load the CUDA module its probe lists and re-run.
6. Run J2 (`j2_embed.sbatch`).
   - It embeds CONCH and MUSK on 100k + 7k tiles.
   - Expect two submissions: CONCH stays fp32 with TF32 off (ruling R36), and MUSK's multi-scale probe is slow.
   - The job resumes from its cache.
7. Run `analyze_cpu.sbatch`. This CPU job is the only full-data analysis route (R34).
8. Run J3 (`j3_distill.sbatch`). It trains the CM student into `results/delta/student/cm_crc`.
9. Optional: J4 (`distill_seed2` or `kernel_build`). Optional: step 8b, the CM bundle on Delta (CPU).
10. Copy back only `results/delta/` and `results/ncsa/`.

Budget: about 5-6.5 A100-hours of the 30 for J1 + 2×J2 + J3 (+ J4).

### 2. UNO Q (Task 15). Plug the board into this machine by USB-C.
1. Bring it up with adb. Check `python3 --version` (Debian trixie = 3.13) and get network or Tailscale working.
2. Install `python3-venv` if it is missing, then run `device/install_offline.sh`. Stage it on the host, then install
   on the board.
3. Run `device/bench_backbones.py` on the board to get real latency and choose the backbone. `headline.json` only
   treats a run as on-board when `simulated: false` and the host is aarch64.
4. Run `device/uno_q_app.py` offline. Flash `device/mcu/twinread_led/` (FQBN `arduino:zephyr:unoq`).
5. Record real captures with the rig: `rig/show_tiles.py` on port 1778, a USB camera filming the screen, then
   `rig/record_frames.py` and `rig/match_captures.py`.

### 3. After the Delta and board runs
- Regenerate `headline.json` (`python -m twinread.headline`) and the figures. Swap the local-subset numbers for the
  Delta numbers in `README.md`, `project.json` and `docs/deck_outline.md`.
- Re-check the device tissue thresholds (chroma 15, max 30) on real rig frames.

### 4. Submission, Wed Oct 21 (watch Slack for the time)
- Submit the abstract and team-availability form. `README.md` opens with the abstract.
- Submit a stand-alone deck of at most 20 slides, built from `docs/deck_outline.md`: aims, methods, results,
  challenges, references.

## Hard rules
- CONCH and MUSK are CC-BY-NC-ND. The CM student and bundle (`bundles/cm`, `distributable: false`) are **never
  distributed or committed**. Only the open-pair bundle may be shared, and only for research use.
- Never commit data, weights, `*.npz`, `*.onnx` or `*.pt` files. Never send patient data to an LLM.
- No command runs on NCSA without the user's OK for that specific command.
- Every judge-facing number must trace to `results/**/*.json`.

## Pointers
- Design: `docs/design.md`. Plan: `docs/plan.md`. Build decisions: `docs/rulings.md` (R1-R39).
- Open minor issues: `docs/deferred-minors.md`. Per-task log: `PROGRESS.md`.
- One-command reproduction: `bash run_demo.sh --smoke`.
