# TwinRead status (2026-10-22)

CONCH and MUSK as two independent readers of every tile; fusion for accuracy, the fused reader's uncertainty as a
review flag; CUDA kernels K1 (dual preprocessing) and K2 (augmentation); a 6 MB int8 student that runs on an Arduino
UNO Q. Research prototype, not a diagnostic device. `README.md` is the judge-facing write-up; this file is the handoff.

## Where things stand

| Area | State | Evidence |
|---|---|---|
| Readers: CONCH, MUSK + the open pair PathGen-CLIP/QuiltNet-B-16 | done, local subset | `results/cm_local/`, `results/open/` |
| Fusion, review flag, pre-registration, bootstrap CIs | done, local subset | `results/headline.json` |
| K1 dual-reader preprocessing kernel (CUDA/Triton/torch) | done, bit-exact | `results/kernels/` |
| K2 augmentation kernel | done | `results/kernels/` |
| Edge student + int8 ONNX bundles | done (CM bundle local only) | `results/student/`, `bundles/open/` |
| Device app + LED sketch on the UNO Q | **done on the board** | `results/uno_q/onboard/`, `figures/fig9_unoq_system.png` |
| Real camera captures | not done (replay tiles only) | `rig/` is ready |
| NCSA Delta full-data run | not done | `ncsa/README.md` |

Tests: CPU suite `pytest tests -q` (about 80 s); GPU tests under the lock with `TWINREAD_GPU_TESTS=1`.

## Headline numbers (from `results/headline.json`)

- Fusion (CRC): CONCH 0.9413, MUSK 0.9480, fused 0.9555 (+0.74 pts, tile-level CI, mostly tumour recall). No gain
  on Camelyon17 or for the open pair.
- Review flag (CRC, 10% budget): fused uncertainty catches 80.3% of its errors; MUSK alone 74.1%, CONCH 66.0%.
- K1: bit-exact; 90,131 tiles/s on the GPU; ×1.11 end to end. K2: 46,315 images/s; ×10.6 in student training.
- Student: 5.65 M parameters, 6.0 MB int8. On the board: 18.7 ms per tile (4 threads), 0.9305 on the full CRC test
  set, 25.4 tiles/s full loop; board int8 flag vs teacher flag kappa 0.32.

## Next steps

1. Real camera captures: `rig/show_tiles.py` on :1778, USB camera filming the screen, `rig/record_frames.py`,
   `rig/match_captures.py`, then score on the board. Re-check the tissue thresholds (chroma 15, max 30) on real frames.
2. NCSA Delta full-data run (every command needs the user's OK first): follow `ncsa/README.md` J1 -> J2 ->
   `analyze_cpu.slurm` -> J3; copy back only `results/delta/`. Then swap the local-subset numbers in README,
   `project.json` and the deck outline.
3. Submission: abstract (README opens with it) and a deck of at most 20 slides from `docs/deck_outline.md`.

## Hard rules
- CONCH and MUSK are CC-BY-NC-ND. The CM student and bundle (`bundles/cm`, `distributable: false`) are never
  distributed or committed. Only the open-pair bundle may be shared, research use only.
- Never commit data, weights, `*.npz`, `*.onnx` or `*.pt`. Never send patient data to an LLM.
- No command on NCSA without the user's OK for that specific command.
- Every judge-facing number traces to `results/**/*.json`. No co-author trailers on commits.

## Pointers
Design `docs/design.md` · plan `docs/plan.md` · rulings `docs/rulings.md` (R1-R39) · open minors
`docs/deferred-minors.md` · log `docs/progress.md` · reproduce `bash run_demo.sh --smoke`.
