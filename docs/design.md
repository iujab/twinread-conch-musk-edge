# TwinRead: CONCH and MUSK as two independent readers, fused for accuracy, distilled to an offline Arduino UNO Q

**Status:** design v2 for approval (updated with the user's answers after the builder check) · **Written:** 2026-10-16 (Fri night, before hack days Sat-Sun Oct 17-18) · **Owner:** Noah
**Project dir:** `ashbyhackathonbench/twinread-conch-musk-edge/` · **Demo ports:** 1777 (host dashboard), device UI is local to the UNO Q

---

## 1. Intent and why

Pathology already has a safety habit for hard cases: a **second read**. Two pathologists look independently. When they
agree the case moves on, and when they disagree it goes to consensus review. TwinRead copies that habit with the two
foundation models the organizers highlight.

- **Two readers.** CONCH (CoCa, ViT-B/16 image encoder, 448 px, 512-d) and MUSK (BEiT-3-style, 384 px input, 100-token
  text limit) were built and trained differently, so they should not always make the same mistakes. Each reads every
  tile independently.
- **A combined reader.** A small model trained on both readers' frozen features should beat either reader alone.
- **A review flag.** When the readers disagree, the case is flagged for a pathologist. The headline is a **triage
  curve**: "review the X% of tiles where the readers disagree and you catch Y% of the errors."
- **Shared blind spots.** Disagreement cannot catch errors that both readers share. Measuring where those happen is a
  finding in its own right.
- **Distilled to the edge.** The combined reader and its review flag are distilled into **one small model** that runs
  offline on an Arduino UNO Q (about the cost of a textbook) with a USB camera. It shows its call and an
  agree/review status on the board's LED matrix. The vision: a second-read aid for hospitals with no slide scanner,
  no GPU and no reliable internet.
- **Custom CUDA kernels.** They make the two-model pipeline and the student training fast enough for the event's GPU
  budget. They attack the bottleneck the organizers themselves measured: CPU image preprocessing.

How this meets the rubric (from `reference/ashby/resource_outline.md`):
- **Aim (precise, reproducible, interpretable):**
  - *Precise:* the fused reader beats either reader alone.
  - *Reproducible:* reader agreement is a measured reliability signal, with the shared blind spots quantified.
  - *Interpretable:* every flag shows which reader said what, with confidences.
- **Development progress:** A100 jobs on Delta, two verified custom CUDA kernels, built on CONCH + MUSK and the
  organizers' datasets.
- **Clinical impact:** triage for pathologists, and an offline device for low-resource settings.
- **Communication:** a live device demo plus slide-ready figures.

## 2. Goals (each has a measurable outcome)

| # | Goal | Measured by |
|---|---|---|
| G1 | The fused reader beats each single reader | Balanced accuracy and macro-F1 with 95% bootstrap CIs, plus paired McNemar vs each reader, on held-out test sets (§8) |
| G2 | The disagreement review flag is useful | Risk-coverage curve, AURC, % of errors caught at fixed review budgets (5/10/20%), accuracy of auto-cleared tiles. **Pre-registered comparison against single-reader confidence**; either outcome is reported |
| G3 | Shared blind spots are characterized | 2×2 (A right/wrong × B right/wrong) per class; a gallery of confident shared errors |
| G4 | The edge student keeps most of the fused reader's value | Student accuracy vs fused reader and vs each reader; agreement of the student's on-device flag with the teacher-level flag; int8 model size; latency on the UNO Q |
| G5 | Custom CUDA kernels are verified and useful | Correctness vs reference implementations; CUDA-event benchmarks vs CPU / PyTorch / Triton baselines; built for sm_120 (local) and sm_80 (Delta A100) |
| G6 | Live, fully offline device demo | The UNO Q runs camera → tile → student → call + flag on screen and LED matrix with networking disabled; a demo video and screenshots |
| G7 | Judge-ready deliverables | README with a ~200-word abstract, `project.json`, ≤ 8 slide-ready figures, tests, `run_demo.sh`, NCSA job scripts, a deck outline for Oct 21 |

## 3. Non-goals

- Fully fine-tuning CONCH or MUSK. It is not affordable in 10 GPU-hours/person and not needed: the readers stay frozen.
- Whole-slide pipelines. TwinRead works on tiles. Slide-level aggregation is a stretch at most.
- Custom kernels inside the transformer blocks. cuBLAS and FlashAttention already run near peak on an A100, and a
  hand-written GEMM or attention would lose. The README says this explicitly.
- Clinical validation, regulatory claims or any diagnostic use.
- The thermal IR camera and UWB. Not used; mention them only as future hardware if asked.

## 4. Anti-goals (a failure even if everything runs)

- Calling it diagnostic, or implying the student is ready for clinical use.
- Claiming that a CONCH/MUSK-derived student can be handed to hospitals. Both licences are CC-BY-NC-ND-4.0 (§5.4).
- Any number in the README, slides or `project.json` that is not traceable to a file in `results/`.
- A demo that passes off a camera pointed at a monitor or printout as a real microscope capture.
- Claiming kernel speedups not measured on the stated GPU, or claiming the student beats its teachers without
  evidence.
- Burning the team's GPU budget on avoidable reruns. A hard cap of ~6 A100-hours for the whole plan (§9.3).
- Anything on the device that needs the internet at run time.

## 5. Constraints (facts, with sources)

### 5.1 Event compute (organizer repo `ashby-hackathon-2026/`, HACKATHON_COMPUTE.md, plus the organizer's Slack note of 2026-10-16)

- **GPU access.** Delta `gpuA100x4` (A100 40 GB, sm_80), account `bfqi-delta-gpu`, no reservation flag. **One GPU
  per team at a time.**
- **Budget.** **10 GPU-hours per person**, billed for time used. Jobs should be 1-2 h. `*-interactive` partitions
  bill 2×. Requests above 16 CPUs or ~62 GB per GPU are billed by the larger share.
- **Reserved windows.** Sat 07:30 → Sun 07:59 and Sun 08:00 → 18:00. A job that crosses a boundary falls into the
  regular queue (days).
- **No internet on compute nodes.** Weights must be downloaded to `~/.cache/huggingface` on the login node first,
  then run with `HF_HUB_OFFLINE=1`. Nothing heavy on the login node.
- **Environment.** `module load pytorch-conda/2.8` then `unset PYTHONNOUSERSITE`.
  - Install with `pip install --user`: `open_clip_torch>=2.24`, `git+https://github.com/mahmoodlab/CONCH.git`, and
    `git+https://github.com/lilab-stanford/MUSK.git fairscale sentencepiece`.
  - The CUDA toolkit module name/version for building extensions is **unverified** and must be checked on day 1.
- **Storage.** Home `/u/<user>` is 100 GB (all outputs go here). `/projects/bfqi` is nearly full. Avoid `/work/hdd`.
- **Shared read-only data** in `/projects/bfqi/shared/datasets/`:
  - `NCT-CRC-HE-100K/data-samples/` (100k .tif, 9 class folders);
  - `pathgen_dataset/` (captions + a 2k-pair sample);
  - `pathmmu/` (ARCH, BACH, MHIST, NCT-CRC, Osteo folders; format uninspected);
  - Quilt-1M is pending storage.
- **Hosted servers.** CONCH / MUSK / QuiltNet / PLIP servers exist (final embeddings only, ~60 img/s shared). For
  tens of thousands of tiles the organizers ask teams to run their own GPU jobs, which we do.
- **Embedding definitions.** The organizer adapters in `ashby-hackathon-2026/models.py` define how CONCH and MUSK
  embeddings are produced:
  - CONCH: `encode_image(proj_contrast=True, normalize=True)`.
  - MUSK: fp16 on GPU, `with_head=True, out_norm=True, ms_aug=False`, with text through the same retrieval head and
    `max_len=100`.

  We reuse these exact semantics so our numbers match what the hosted servers return.
- **Use, licences and citation.** Any model, dataset or tool is allowed if its terms are followed; cite everything.
  Patient data must never be pasted into LLM chats.
- **Access.** The user has approved Hugging Face access to CONCH and MUSK.

### 5.2 Local machine (this bench)

- **GPU.** RTX 5060 8 GB (sm_120, ~4 GB free while the desktop runs). CUDA 13.1 at `/usr/local/cuda`. torch
  2.14.1+cu130. Triton 3.8. `cpp_extension` builds work. Nsight Compute counters are admin-only, so time with CUDA
  events.
- **HF token.** None on this machine yet, so CONCH and MUSK cannot run locally until the user runs `hf auth login`
  here.
- **Open models cached:** PathGen-CLIP ViT-L (CC-BY-4.0), QuiltNet-B-16/B-32 (MIT), BiomedCLIP (MIT), PLIP (no
  licence field), and small timm backbones (MobileNetV4-conv-small, EfficientNet-B0, MobileOne-S0, RegNetX-002,
  ResNet-18, LCNet-050).
- **Bench assets to reuse** (copy code in with attribution; never modify those projects):
  - StainForge: verified HED-jitter, Reinhard and Macenko CUDA kernels, built for sm_80/86/90/120.
  - BigBatch: a verified fused contrastive loss (CUDA C++ + Triton).
  - promptscope-edge: ONNX int8 export with ORT static QDQ, and the UNO Q benchmark harness pattern.
  - The organizer adapters.

### 5.2b Answers from the user (2026-10-16, after the builder check)

- **Board.** The UNO Q is the **4 GB** variant with Debian already set up. The user will get it online. The camera
  connects through a **USB-C hub**, and the board travels with the user all weekend.
- **GPU budget.** **30 A100-hours total** for this project. TwinRead is treated as the team's main submission.
- **Camera demo.** The camera films **a screen showing tissue images**, to demonstrate that the physical camera
  pipeline works. There is no microscope.
- **CONCH + MUSK locally tonight.** The user will run `hf auth login` on this machine. The `hf` CLI is on PATH via
  `~/.local/bin/hf` → bench `.venv`.
- **NCSA approval.** **Ask for an OK in chat before every command run on NCSA servers.** Anything on this machine is
  allowed. The user will provide an SSH'd terminal when Delta time comes.

### 5.3 Arduino UNO Q (vendor and retailer listings; community reports)

- **Processor.** Qualcomm Dragonwing QRB2210: 4× Cortex-A53 @ 2.0 GHz and an Adreno 702 GPU (no CUDA). There are 2 GB
  (16 GB eMMC) and 4 GB (32 GB eMMC) variants, and the user's variant is unknown. The OS is Debian.
- **Microcontroller.** STM32U585 (Cortex-M33). The **8×13 LED matrix** is on the board, and the **Bridge** RPC links
  Linux ↔ MCU. App Lab runs on the board or from a PC.
- **Camera.** UVC webcams on a USB-C hub appear as `/dev/video0` (community report).
- **Access path (decided).** The board has a single USB-C port.
  1. **Setup:** connect it by USB-C to this machine and use `adb` (installed, `/usr/bin/adb`; App Lab uses the same
     channel) to push files and run commands. Configure Wi-Fi with `nmcli` and install Tailscale; the user completes
     the Tailscale login.
  2. **Demo:** move it to the USB-C hub with the camera, power and display, and keep reaching it over the tailnet.
  3. **Fallback:** if Wi-Fi isn't possible, aarch64 wheels are pushed over `adb`, and pip goes through an
     `adb reverse` proxy to this machine.

  `arduino-cli` 1.5.1 is installed here, for the MCU sketch.
- **Performance.** There is no published ONNX Runtime benchmark for this board, so we measure our own.
  YOLOv5-pico reaches ~17 FPS at 640×480 through Edge Impulse (Foundries.io), which suggests a small CNN at 224 px is
  feasible.

### 5.4 Licences (checked on Hugging Face, 2026-10-16)

- **CONCH and MUSK are CC-BY-NC-ND-4.0** (non-commercial, no derivatives). Whether a distilled student counts as a
  derivative is legally unsettled. We must not distribute a CONCH/MUSK-derived student.
- So TwinRead is built **generic over the reader pair** and ships two configurations:
  - **TwinRead-CM** = CONCH + MUSK. This is the hackathon research result, run on Delta (and locally if a token is
    added). Its student is shown for research and is **not distributed**.
  - **TwinRead-Open** = PathGen-CLIP (CC-BY-4.0) + QuiltNet-B-16 or BiomedCLIP (MIT). This is the
    **distributable** edge model and the honest answer to "give it to any hospital". It is also what runs tonight
    before Delta access.
- **Data:**
  - NCT-CRC-HE-100K / CRC-VAL-HE-7K: CC-BY-4.0.
  - Camelyon17-WILDS: CC0.
  - PathMMU: check its terms before use.

### 5.5 Time

- Fri night (local build).
- Sat-Sun (Delta + device).
- Mon-Wed (deck + abstract; due Wed Oct 21, ≤ 20 main slides).

## 6. Architecture

```
            ┌──────────── teacher side (GPU: Delta A100 for CM, local RTX 5060 for Open) ────────────┐
 raw tiles ─┤ K1 fused dual-preprocess (uint8 → reader-A tensor + reader-B tensor, one GPU pass)     │
 (uint8)    │      → Reader A (CONCH | PathGen-CLIP)   → zA  ─┐                                    │
            │      → Reader B (MUSK  | QuiltNet/BiomedCLIP) → zB ─┼→ per-reader probes + zero-shot │
            │                                                 └→ Fusion head → fused probs        │
            │                         review signals (disagreement, JS divergence, confidence)    │
            └───────────────────────────────────────────────────────────────────────────────────────┘
                         │ cached embeddings + teacher outputs (.npy, manifest)
                         ▼
            ┌──────────── student training (GPU) ────────────────────────────────────────────────┐
 tiles ─ K2 fused camera-domain augmentation ─→ student CNN (224 px) ─→ hA ≈ zA, hB ≈ zB,        │
            │                                         fused-class head, review head            │
            │  losses: cosine + relational InfoNCE (BigBatch kernel) per head, KD on fused      │
            │          logits, CE on labels, BCE on teacher review labels                       │
            └───────────────────────────────────────────────────────────────────────────────────────┘
                         │ ONNX → ORT static int8 (QDQ, per-channel)
                         ▼
            ┌──────────── Arduino UNO Q (offline) ─────────────────────────────────────────────────┐
 USB cam ─→ │ Debian/Python: capture → crop → int8 student (ORT, 4 threads) → call + review flag  │
            │  → local UI on attached display + JSON log      ── Bridge RPC ──→ STM32: LED matrix   │
            │                                                                   (✓ agree / ? review) │
            └───────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.1 Reader abstraction (decided)

- **Interface.** `Reader` has `name`, `preprocess_spec` (target size, resize mode, crop, mean/std, read from each
  model's official transforms at load time, never hard-coded), `encode_images(tensor) → L2-normalized float32`,
  `encode_texts(list[str])` and `dim`.
- **Implementations:**
  - `conch` and `musk`: semantics copied from the organizer adapters (§5.1).
  - `pathgenclip`, `quiltnet_b16` and `biomedclip`.
- **Pair configs:** `CM = (conch, musk)` and `OPEN = (pathgenclip, X)`. X ∈ {quiltnet_b16, biomedclip} is chosen on the
  validation split as the partner with the most complementary errors (lowest error correlation at similar accuracy).
  This is a builder call, recorded in `results/`.

### 6.2 Readers' outputs

- **Zero-shot:** CONCH-style prompt ensembles per class (several templates × class-name synonyms), averaged text
  embeddings, softmax over cosine × a fixed scale. Prompts are frozen in `prompts/*.json` before any test evaluation.
- **Linear probe:** logistic regression on train-split features. Probes are the main readers for G1-G3, and zero-shot
  is reported alongside.
- **Feature choice (Claude's call, pre-registered).** For the teacher analysis (G1-G3), each reader's primary probe
  uses **the features its authors recommend for linear probing**, so "fusion beats each reader" is measured against
  each reader at its best:
  - CONCH: the image encoder's representation before the contrastive projection, as in its paper;
  - MUSK: the `ms_aug=True` features.

  The **text-aligned embeddings** (the hosted-server semantics in §5.1) are used for zero-shot, for the student's
  hA/hB targets and on the device, and are also probed as a sensitivity analysis. Both are reported.

### 6.3 Fusion ("TwinRead head")

All candidates are reported. The primary is chosen on validation only:
- (a) average of the two probes' probabilities;
- (b) logistic regression on `[zA, zB]`;
- (c) a small gated MLP on `[zA, zB, zA·W, |zA−zB|]` with dropout, few epochs.

### 6.4 Review flag (pre-registered in `results/preregistration.json` before test evaluation)

- **Signals:**
  - s1 = 1[argmax pA ≠ argmax pB];
  - s2 = Jensen-Shannon divergence(pA, pB);
  - s3 = 1 − max p_fused;
  - s4 = a logistic combination of s1-s3 fitted on validation.
- **Baselines:** 1 − max pA and 1 − max pB.
- **Metrics:** risk-coverage curve, AURC, % of errors caught at 5/10/20% review, and accuracy of auto-cleared tiles.
  Thresholds are chosen on validation and reported on test.

### 6.5 Shared blind spots

- Per-class 2×2 of reader correctness.
- The rate of confident shared errors (both wrong, same wrong label, both confident).
- A gallery figure.

### 6.6 Edge student ("TwinRead-Edge")

- **Backbone selection.**
  - Benchmark int8 latency for the cached timm candidates on the actual UNO Q if available. If not, measure on
    x86/ARM-emulation and label it an estimate.
  - Pick the most accurate backbone meeting the latency budget, **≤ 250 ms per 224-px tile at 4 threads**.
  - Input is 224 px, with normalization inside the graph.
- **Heads:**
  - hA (to reader-A space);
  - hB (to reader-B space);
  - fused-class head;
  - review head, predicting the teacher-level review label (s4 above its validation threshold).
- **Losses:**
  - per head, cosine feature distillation + relational InfoNCE against the teacher batch (the BigBatch kernel);
  - KD on fused logits (with temperature);
  - CE on labels;
  - BCE on the review head.
- **Robustness.** Teacher targets come from **clean** tiles. The student sees **K2-augmented** tiles (consistency
  distillation), so camera-like input maps to the clean-tile meaning.
- **On-device flag.** A combination of hA/hB disagreement (each through its own probe/prompt bank) and the review
  head. We measure how often it matches the teacher-level flag, and caveat that the two heads share a backbone, so
  they are less independent than the real readers. An ablation tests two separate tiny backbones if latency allows.

### 6.7 Export and bundle

- **Export.** ONNX (opset 17) → ORT static int8, QDQ, per-channel, calibrated on 512 train tiles. Report fp32 → int8
  drift (cosine, accuracy).
- **Bundle:** `twinread_edge_int8.onnx`, `prompt_bank.npz` (precomputed text embeddings for the supported vocabulary),
  `probes.npz`, `config.json` and a `SHA256SUMS` manifest. It installs from a local aarch64 wheel cache copied by USB,
  with **no network at run time**.

### 6.8 Device application (UNO Q)

- **Linux side (Python):** onnxruntime, numpy, opencv-python-headless, Pillow.
  - Loop: capture from `/dev/video0` → center crop (plus an optional 3-crop vote) → resize to 224 → inference →
    call, confidence, top concepts, review flag.
  - A local web UI on 127.0.0.1, shown on an attached display, plus a JSON log.
  - Two **modes**, each labelled on screen:
    - *Replay*: bundled CRC-VAL tiles.
    - *Live camera*: what the camera sees. In the demo this is a monitor, a printout or a microscope eyepiece, and
      it is labelled as such.
- **MCU side (Arduino sketch):** the LED matrix shows idle / thinking / ✓ agree / ? review. An optional button
  triggers a capture. Linux → MCU over Bridge RPC.
- **Fallback.** If the device isn't available, the same app runs on x86 in "device-simulation" mode, labelled as
  simulation, and on-board numbers are marked "not yet measured".

## 7. Custom CUDA kernels (decided set)

Same engineering standard as StainForge and BigBatch:
- CUDA C++ via `torch.utils.cpp_extension`, with a Triton or plain-PyTorch baseline.
- Correctness vs reference implementations, with tolerance tables.
- CUDA-event benchmarks with warmup and medians, plus achieved bandwidth vs device copy bandwidth.
- GPU-gated tests that skip cleanly on CPU.

| Kernel | What it does | Why it matters here | Verified against | Benchmarked against |
|---|---|---|---|---|
| **K1 fused dual-preprocess** | One launch takes a uint8 NHWC tile batch on the GPU and writes **both** readers' inputs: per-reader resize (bicubic/bilinear as each model's transform specifies), center crop, normalization, fp16 NCHW. Grid z selects the reader | The organizers' load test found CPU preprocessing, not the GPU, limits throughput. K1 makes dual reading feed the GPU from one pass | Each reader's official torchvision transform on the same tensor (pixel max abs diff); **downstream embedding cosine ≥ 0.999 target**, reported per reader. PIL-vs-torch bicubic differences are documented, and the reference is each model's own transform | CPU PIL+torchvision per image (organizer style), a DataLoader with N workers, torch GPU ops, Triton; end-to-end tiles/s for dual reading |
| **K2 fused camera-domain augmentation** | One launch per batch, per-image random parameters (Philox). Effects targeted at the **camera-of-a-screen** demo: display gamma and colour-profile shift, screen-pixel-grid moiré (resampling against a simulated subpixel grid), brightness/glare gradient, small perspective/rotation jitter, separable Gaussian defocus (shared-memory tiles), white-balance gain, sensor noise, and **JPEG/MJPEG-like 8×8 block DCT quantization** at a random quality. HED stain jitter (StainForge math) is also applied for stain robustness | Student training must simulate USB-camera capture of a displayed tile. Doing that per image on the CPU would starve the A100 | Numpy references for each op and for the composition; the DCT checked against `scipy.fft.dctn` | CPU numpy/PIL per image, torch GPU ops; student-training images/s with vs without K2 |
| *Reused* BigBatch | Fused relational InfoNCE for the distillation term | Large relational batches at low memory | Already verified (`bigbatch-flash-contrastive-kernel`) | Cited, not re-benchmarked |
| *Optional* StainForge Macenko | Stain normalization inside the K1 path | Off by default: StainForge found no out-of-distribution gain | Already verified | — |

**Delta build caveat.** Delta's torch 2.8 (pytorch-conda module) differs from local torch 2.14, so the extension must
be **JIT-built on Delta** for sm_80. Locally we compile-verify sm_80 with nvcc 13.1. If the Delta build fails, the
pipeline falls back to the Triton / torch-ops path automatically. The science never waits on a kernel.

## 8. Data and evaluation protocol

- **D1 (primary).** Train on NCT-CRC-HE-100K and test on CRC-VAL-HE-7K (9 classes; a separate patient set).
  - Validation = a stratified 10% of train. NCT-CRC-100K has no patient IDs, which is a documented limitation.
  - Local: download via HF parquet (`1aurent/NCT-CRC-HE`). Tonight uses a **stratified 20k-tile train subset** plus
    the full 7k test (my call: about 4 GB). Delta uses the full 100k from the shared read-only copy.
- **D2 (out-of-distribution).** Camelyon17-WILDS (hospital-labelled, tumor vs normal): train on hospitals 0/3/4,
  test on unseen hospital 2. Does the review flag fire more on an unfamiliar hospital, and does it still catch errors?
  Use a ≤ 1.5 GB own subset.
- **D3 (Delta only).** PathMMU subset folders (MHIST, BACH, Osteo, NCT-CRC). Inspect the format first, and use those
  whose labels are recoverable as extra tasks for G1-G3.
- **Camera-domain test.**
  - CRC-VAL tiles passed through K2 at fixed severities (simulated).
  - **A real camera-of-screen test set from a capture rig (Claude's call).**
    - This machine shows held-out CRC-VAL tiles full-screen, one at a time, with a frame ID marker.
    - The UNO Q's USB camera films the screen, and frames are matched to tile IDs by the marker and timestamps.
    - This gives a few hundred **labelled real captures**, with no manual work beyond pointing the camera.
  - These results are reported separately and labelled "real camera filming a screen".
  - The device crop takes the displayed tile region: the tile is shown at a fixed on-screen size, and the device
    finds and crops it (marker/border detection, with a center-crop fallback).
- **Statistics.** Bootstrap 95% CIs (tile-level, noting no patient IDs) and McNemar for paired accuracy. Every test
  number is computed once, after `preregistration.json` is written.

## 9. Where things run, and the runbook

### 9.1 Tonight, locally (no CONCH/MUSK unless a token is added)

1. Scaffold the project: package, tests, layout per bench CLAUDE.md, `project.json`, README skeleton.
2. Reader abstraction plus the OPEN pair, and D1 subset download.
3. **K1 and K2** written, verified and benchmarked on sm_120, with sm_80 compile-checked.
4. The full OPEN pipeline end to end: embeddings → probes/zero-shot → fusion → review flag → blind spots → student
   distillation → ONNX int8 → device bundle → device app in simulation mode.
5. Figures v1 and the README v1 with OPEN results, clearly labelled as the open pair.
6. **Once the user runs `hf auth login` here (planned for tonight),** the CM pair also runs locally on the D1 subset:
   CONCH and MUSK fp16, one at a time, within ~4 GB free VRAM. These are dev numbers and an early check that the
   install and offline loading work. Delta produces the final full-data numbers.
7. **Board bring-up as soon as it's plugged in by USB-C:** `adb` access, Wi-Fi + Tailscale, Python deps, then a
   latency benchmark of the backbone candidates at 1/2/4 threads. This fixes the student backbone **before** the
   Delta distillation job.
8. The screen-capture rig (§8): it records labelled real camera-of-screen frames once the camera is on the board.

### 9.2 On Delta (when the user connects SSH)

- **Login node only:** clone/copy the project, load modules, `pip install --user` the packages, `hf auth login`, and
  **pre-download CONCH + MUSK weights** to the HF cache.
- **Jobs**, each `sbatch` with a realistic `--time`, `--cpus-per-task=16`, `--mem=60G`, 1 GPU:
  - **J1 (≤ 0.5 h):** environment check, offline CONCH/MUSK load, K1/K2 JIT build for sm_80, correctness smoke and
    kernel benchmarks on the A100.
  - **J2 (≤ 1.5 h):** CM embeddings for D1 (107k tiles) + D3 + text prompt banks. Resumable, with checkpoints.
  - **J3 (≤ 1.5 h):** CM student distillation on the full D1 (preloaded into RAM as uint8, about 15 GB) with K2.
  - **J4 (≤ 1.5 h, only if needed):** a second seed, ablations, or a rerun.
- **CPU work** (probes, fusion, flag analysis) runs on `bfqi-delta-cpu` (`ncsa/analyze_cpu.slurm`, into
  `results/delta/`). Ruling R34 (final review) made this the only supported full-data route: analysing locally would
  overwrite the local 20k embedding caches and mix local labels with Delta embeddings.
- **How Claude operates there.** The user sets up SSH with a persistent ControlMaster (one Duo prompt), so `ssh delta
  '...'` works non-interactively from this machine. Claude never submits a GPU job without stating its `--time` and
  purpose first. No job resubmits itself in a loop.

### 9.3 GPU budget

- **Budget: 30 A100-hours total** (the user). Plan: J1-J4 ≈ 5 h, with ≤ 10 h in total so ≥ 20 h stays in reserve
  for reruns and for anything the event brings.
- **Stop rule:** if a job runs over 1.5× its estimate, cancel it and shrink the scope.
- **Every command on NCSA** (login node or `sbatch`) is shown to the user and run only after an explicit OK in chat.

### 9.2b Data on Delta

- The organizer runbook notes that, as of Oct 16, the shared datasets may not yet be readable by students. It also
  says CRC-VAL-HE-7K is not listed.
- **Plan:** check `/projects/bfqi/shared/datasets/` first (with an OK). If it is missing or unreadable, download the
  `1aurent/NCT-CRC-HE` parquet shards (~15 GB) on the login node into `/u/<user>` (with an OK), or rsync the copy
  already on this machine.

### 9.4 Device

- **Setup:** the UNO Q needs a network connection once, to copy the bundle (Wi-Fi + Tailscale or `scp`, or USB).
- **Then:** benchmark latency at 1/2/4 threads and run the offline demo with networking turned off.

## 10. Timeline

| When | What |
|---|---|
| Fri night | §9.1 steps 1-8. Deliverable: the OPEN pipeline complete with tests and kernels verified; CM dev numbers on the local subset; board benchmarked if plugged in; figures v1 |
| Sat morning | SSH to Delta, login-node setup, J1, then J2. Device setup if the board is reachable |
| Sat afternoon | CM probes/fusion/flag/blind spots (CPU). J3 student distillation. Device benchmark |
| Sun | J4 if needed, device demo polish, final figures, README v2, `project.json` |
| Mon-Wed | Deck (≤ 20 slides) and abstract. Submit Wed Oct 21 |

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The fusion doesn't beat the best single reader | Report it honestly. The flag (G2) and the edge work (G4/G6) still stand. Try a calibrated product of experts |
| Disagreement is no better than single-reader confidence | Pre-registered comparison, reported either way. s4 (the learned combination) is the practical flag |
| The CRC task is too easy (ceiling ~95%+), so few errors remain to triage | D2 (OOD hospital) and D3 tasks provide harder cases. Zero-shot readers have more errors to study |
| The K1 build fails on Delta's torch 2.8 / CUDA module | Automatic Triton/torch fallback; K1 benchmarked on the A100 only if it builds |
| CONCH/MUSK install or offline load breaks on compute nodes | J1 tests this first. The organizer adapters already document the MUSK tokenizer/cache fixes |
| The UNO Q is unavailable or too slow | Simulation mode labelled as such. Pick a smaller backbone. Report measured latency honestly |
| Camera-domain gap (screen/print capture) | K2 augmentation in training. The demo is labelled honestly. Real-camera captures reported separately |
| Licence | The CM student is never distributed. TwinRead-Open is the distributable artifact |
| The GPU budget runs out | Each job is resumable. The hard cap is §9.3. CPU work stays off the GPU |

## 12. Deliverables and layout

```
twinread-conch-musk-edge/
  README.md, project.json, run_demo.sh, PROGRESS.md
  docs/design.md (this file), docs/plan.md (implementation plan)
  src/twinread/        readers.py, data.py, embed.py, probes.py, fusion.py, review.py, blindspots.py,
                       student.py, export.py, figures.py, app_host.py (dashboard :1777)
  src/twinread/csrc/   k1_dual_preprocess.cu, k2_camera_aug.cu, bindings.cpp ; kernels.py (build + fallbacks)
  prompts/             per-dataset prompt ensembles (frozen before test)
  ncsa/                setup_login.sh, j1_check.slurm, j2_embed.slurm, j3_distill.slurm, j4_spare.slurm, README.md
  device/              uno_q_app.py, ui/ (static), mcu/twinread_led/twinread_led.ino, install_offline.sh, bundle/
  tests/               CPU tests (always) + GPU-gated kernel tests (skip without CUDA)
  figures/ results/ data/ (git-ignored style)
```

## 13. Decisions log

| Decision | Made by |
|---|---|
| Two-reader framing with CONCH + MUSK, a fused reader, a disagreement flag and shared blind spots | User (approved in chat) |
| Distill to an offline UNO Q with a USB camera, framed as a combined, more accurate yet efficient on-device model for low-resource hospitals | User |
| Thermal camera not needed. New subdir. Heavy emphasis on planning | User |
| Build generic over the reader pair. TwinRead-Open (open teachers) is the distributable edge model, and TwinRead-CM is the research result | Claude (licence-driven, §5.4) |
| Kernels in the data path (K1, K2) and reused losses, not inside the transformers | Claude (§3) |
| D1 local subset of 20k; full 100k on Delta | Claude |
| The student backbone is chosen by measured latency, with a 250 ms budget | Claude |
| Port 1777 for the host dashboard | Claude |
| 4 GB UNO Q, already on Debian; USB-C hub for the camera; board travels with the user | User |
| 30 A100-hours total; TwinRead is the main submission | User |
| The camera films a screen showing tissue images (no microscope) | User |
| CONCH + MUSK run locally tonight after `hf auth login` | User |
| An explicit OK in chat before every command on NCSA; anything on this machine is allowed | User |
| Board access over USB-C + adb first, then Wi-Fi + Tailscale | Claude |
| Probe features: each model's recommended features as primary; text-aligned features for zero-shot, student and device | Claude |
| K2 retargeted to camera-of-screen effects; a labelled screen-capture test rig | Claude |
| Data on Delta: shared path first, else home download (each with an OK) | Claude |

## 14. Left to the builder

- Exact prompt templates, fusion hyperparameters and student hyperparameters (validation only).
- The OPEN partner choice (§6.1).
- UI styling.
- Figure design, within the bench figure rules.
- Wheel-cache mechanics for the offline install.
- How K1 maps threads to the two outputs.
