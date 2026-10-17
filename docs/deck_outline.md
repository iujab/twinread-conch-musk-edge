# TwinRead deck outline (submission Wed Oct 21: at most 20 main slides)

Each slide gives a title, the content, the figure and a one-line speaker note. Every number comes from
`results/headline.json` or the JSON it cites. Numbers are local-subset dev numbers until the Delta runs land; if
they land, update from `results/` and keep the honest framing. Every slide footer reads "Research prototype, not a
diagnostic device."

The deck covers, in order: aims (1-3), methods (4-6), results (7-16), challenges overcome and remaining (17-18),
impact (19) and references (20).

---

### Slide 1: TwinRead, two foundation-model readers, one honest second read

- CONCH and MUSK as two independent readers, plus a fused reader, a review flag and an int8 student built for an
  offline Arduino UNO Q (x86 replay today; on-board run pending).
- Team, the Ashby Prize 2026 hackathon, and the disclaimer.
- Figure: `figures/fig8_system.png` (thumbnail).
- Note: "We tested whether two pathology foundation models can act like a second pathologist."

### Slide 2: The clinical problem (aims)

- A second read catches errors, but it costs a second expert.
- The aim, in the organizers' words: make pathology more **precise** (a fused reader), **reproducible** (agreement
  and blind spots measured) and **interpretable** (every flag shows which reader said what).
- Who benefits: pathologists triaging review time, and clinics with no scanner, no GPU or no reliable internet.
- Note: "Our question is where the pathologist's next minute of attention should go."

### Slide 3: Aims and pre-registered hypotheses

- G1: the fused reader beats each reader.
- G2: the review flag beats single-reader confidence.
- G3: shared blind spots are measured.
- G4: an edge student keeps most of the value.
- G5: the kernels are verified and useful.
- G6: an offline device runs.
- Pre-registered in `preregistration.json` before any test label: the fusion rule, the best single reader, the flag,
  thresholds and prompt hashes.
- Note: "We wrote down what would count as success before looking at test data."

### Slide 4: Data and models (methods)

- Readers:
  - CONCH (`reference/papers/lu2024_conch.md`): CoCa, 448 px, 512-d.
  - MUSK (`reference/papers/xiang2025_musk.md`): 384 px, 100-token text limit.
  - Both are probed on the features their authors recommend; aligned embeddings serve zero-shot and the student.
- Data:
  - NCT-CRC-HE-100K, a 20k subset: 17,349 train and 2,658 val, group-disjoint.
  - CRC-VAL-HE-7K test: 7,180 tiles, a different patient set.
  - Camelyon17-WILDS: train on hospitals 0/3/4, test on unseen hospital 2 (1,500 tiles).
- Licences:
  - CONCH and MUSK are CC-BY-NC-ND, so the research configuration is CM.
  - The distributable configuration is OPEN: PathGen-CLIP + QuiltNet-B-16 (`reference/papers/ikezogwo2023_quilt1m.md`).
- Note: "Same protocol for both pairs; the open pair is also our negative control."

### Slide 5: Pipeline (methods)

- Steps: tiles → K1 → CONCH / MUSK → probes → fused reader → signals s1-s4 → triage.
- Distillation: teacher targets → K2 → student → int8 → UNO Q.
- Figure: `figures/fig8_system.png`.
- Note: "Teacher side on a GPU; the student side is built for a textbook-priced board, where it has not run yet."

### Slide 6: Statistics and honesty (methods)

- Everything is fitted on train/val, and test is evaluated exactly once.
- Out-of-fold (K=5) student targets.
- CIs:
  - Camelyon17: bootstrap clustered by slide.
  - CRC: tile-level, labelled anti-conservative (no slide ids). McNemar tests are paired.
- Post-hoc analyses are labelled as post hoc. The triage-curve reproduction matches `review.json` exactly
  (32 checks per dataset).
- Note: "Where our statistics are optimistic, we say so on the slide."

### Slide 7: Result 1. Fusion helps a little, on CRC only

- CRC: fused 0.9555 vs MUSK 0.9480 vs CONCH 0.9413; +0.74 pts, CI [+0.18, +1.29], tile-level.
- Camelyon17: −0.40 pts against CONCH.
- Open pair: +0.08 pts (CRC) and 0.00 (Camelyon17).
- Figure: `figures/fig1_two_readers.png`.
- Note: "A real but small gain, with optimistic CIs."

### Slide 8: Result 1b. Why the CRC gain is fragile

- Fused minus MUSK recall: MUS −13.2 pts, STR +13.1. Large opposing swings that cancel (net −0.01 pts of balanced
  accuracy).
- The net gain is mostly tumour recall: TUM +3.6 recall points, worth +0.40 of the +0.74 (balanced accuracy is the
  mean of 9 class recalls); ADI, DEB, LYM and NORM add the rest.
- Validation picked the fusion rule by 1-2 tiles (2,641 vs 2,640 vs 2,639 correct of 2,658). The runner-up, the gated
  MLP, scores −0.40 pts on test.
- The open control pair shows no fusion gain on either dataset.
- Note: "We'd rather show you the fragility than have a judge find it."

### Slide 9: Result 2. Triage: what 10% of review time buys

- CRC: fused uncertainty (s3) catches 80.3% of errors.
- The learned flag with disagreement (s4) also catches 80.3%; the s3 and s4 rankings correlate at Spearman 0.9999.
- Plain disagreement catches 59.0% and JS divergence 72.8%. Footnote: s1 is binary, so it flags the 5.6% of tiles
  where the readers disagree and the other 4.4% of the budget is filled in shuffled test order (effectively random).
- Each reader flagging itself: MUSK 74.1%, CONCH 66.0%.
- Figure: `figures/fig2_review_flag.png` (curves).
- Note: "The useful signal is the fused reader's own doubt, not the disagreement itself."

### Slide 10: Result 2b. Where the flag falls short

- Camelyon17: s4 clears tiles at 99.04% accuracy, against 99.11% for CONCH alone. No gain.
- The pre-registered val-calibrated threshold flags 25.1% of CRC test tiles instead of 10%, catching 97.5% of errors.
  CRC validation is near ceiling (0.9936) and val → test is a distribution shift.
- Takeaway: recalibrate thresholds on the target site before use.
- Note: "Pre-registration made this failure visible instead of tuned away."

### Slide 11: Result 3. Shared blind spots

- 53 confident shared errors on CRC (0.74%); 36 are MUS↔STR.
- On Camelyon17, 11 of 12 pass the pre-registered flag unflagged.
- Figure: `figures/fig2_review_flag.png` (blind-spot bars and gallery).
- Note: "No *disagreement* signal can catch what both readers get wrong together. The pre-registered flag still
  catches 47 of the 53 on CRC, through the fused reader's uncertainty. This is where pathologists stay essential."

### Slide 12: Kernel K1. Bit-exact dual preprocessing

- One CUDA launch produces both readers' inputs, bit-exact to the official transforms.
- Kernel bench (input already on the GPU): 90,131 tiles/s, against 221 for single-process PIL (×408) and 82,058 for
  our Triton version; 46,157 tiles/s including the host-to-device copy.
- Inside the CONCH + MUSK embedding run, preprocessing gets ×202 faster than PIL (42,631 vs 211 tiles/s), but end to
  end only ×1.11: the model forward is 89.4% of the time.
- The organizers' server bottleneck (per-image CPU preprocessing) is a different regime, not measured here.
- Figure: `figures/fig3_kernels.png` (top).
- Note: "We report the small end-to-end number next to the big kernel number."

### Slide 13: Kernel K2. Camera-of-screen augmentation (SIMULATION)

- Warp, stain jitter, display gamma, moiré, glare, defocus, noise and JPEG, in one launch.
- 46,315 images/s, within 1 level per op of a numpy reference; compute-sanitizer clean.
- Student-training augmentation is ×10.6 faster than torch.
- Figures: `figures/k2_severity_grid.png` and `figures/fig3_kernels.png` (bottom).
- Note: "It teaches the student what a USB camera filming a screen does to a tile image. It is a simulation."

### Slide 14: The edge student

- 5.65 M parameters, 1.4% of the teachers' vision towers; 6.0 MB int8.
- fp32 0.9415 (CONCH 0.9413); 0.9135 at the strongest simulated camera severity.
- The deployed int8 model scores 0.9219, below every teacher; int8 costs 2.0 points (CM) and 2.7 (OPEN).
- Honest caveat: on the device the two "readers" are two heads on one shared student backbone, so they are less
  independent than CONCH and MUSK. Against the teacher flag, the fp32 device flag agrees on 81.2% of tiles (kappa 0.42)
  and recovers 41.0% of teacher-flagged tiles; the deployed int8 flag agrees on 79.5% (kappa 0.34) and recovers 33.3%.
- Figure: `figures/fig5_edge_student.png`.
- Note: "The value is offline, small and cheap, not beating the foundation models."

### Slide 15: The device and the capture rig

- UNO Q app: camera or replay → crop → int8 student → AGREE/REVIEW, with a local UI and an LED-matrix sketch
  (compile-verified, not flashed). It has run only as an x86 replay so far.
- Deployed int8 bundle flag: 12.1% of clean CRC test tiles; it catches 57.6% of the int8 model's own errors and
  agrees with the teacher flag at kappa 0.34 (`results/student/cm_crc/int8_flag_fidelity.json`).
- The UI labels the bars as student heads (Head A CONCH-space, Head B MUSK-space), not as two independent readers.
- Rig (simulated captures): 8,000 trials, 7,572 accepted, 0 wrong labels.
- **Pending**: on-board latency (measured when the board is plugged in) and real camera captures.
- Figure: `figures/ui_device_sim.png`. It is an x86 replay; say so aloud.
- Note: "The demo camera films a screen, not a microscope."

### Slide 16: Interpretability in practice: the dashboard

- Every tile shows CONCH's call, MUSK's call, the fused call and the flag. Filters cover shared errors, missed errors
  and disagreements.
- Figure: `figures/ui_dashboard.png`.
- Note: "A pathologist can see why a tile was flagged and what each reader said."

### Slide 17: Challenges overcome

- The CONCH and MUSK loaders needed shims for transformers v5.
- PathGen-CLIP's real input size (336 px) was caught by asserting the processor against the spec.
- MUSK probe embedding is slow (about 2 h wall under GPU sharing).
- K2's JPEG differences were traced to provable rounding-boundary blocks.
- int8 calibration collapsed with MinMax/Entropy (val label agreement 0.4406), so percentile calibration was used.
- The licence problem was solved with two configurations.
- Note: "Each of these is logged with its evidence in PROGRESS.md."

### Slide 18: Challenges remaining and next steps

- NCSA Delta full-data runs (100k CRC, CONCH + MUSK, A100): pending; the kit is in `ncsa/`.
- UNO Q latency and offline demo: pending.
- Real camera captures: none yet.
- CRC slide ids for clustered CIs.
- Closing the int8 test gap.
- Site-specific threshold calibration.
- PathMMU subsets (`reference/papers/sun2024_pathmmu.md`) as extra tasks.
- Note: "Here is what we would do with one more day of Delta time."

### Slide 19: Why it matters to pathologists and patients

- Triage: on CRC, at a 10% review budget, the fused reader's uncertainty catches about 80% of its errors. On
  Camelyon17 it does not beat CONCH alone, and the pre-registered threshold over-flagged CRC. Recalibrate per site.
- Shared blind spots tell the pathologist where AI agreement is not reassurance.
- An offline, distributable OPEN bundle points toward low-resource settings, pending real-world validation.
- Context: multimodal pathology surveys (`reference/papers/li2025_multimodal_cpath_survey.md`,
  `reference/papers/muneer2026_multimodal_integration_review.md`).
- Note: "A second-read aid, not a diagnosis."

### Slide 20: References

- Lu et al. CONCH, Nature Medicine 2024: `reference/papers/lu2024_conch.md`.
- Xiang et al. MUSK, Nature 2025: `reference/papers/xiang2025_musk.md`.
- Ikezogwo et al. Quilt-1M / QuiltNet, 2023: `reference/papers/ikezogwo2023_quilt1m.md`.
- Sun et al. PathMMU, 2024: `reference/papers/sun2024_pathmmu.md`.
- Li et al. multimodal computational pathology survey, 2025: `reference/papers/li2025_multimodal_cpath_survey.md`.
- Muneer et al. multimodal integration review, 2026: `reference/papers/muneer2026_multimodal_integration_review.md`.
- Data and open models:
  - NCT-CRC-HE-100K / CRC-VAL-HE-7K (Kather et al.);
  - Camelyon17-WILDS;
  - PathGen-CLIP;
  - BiomedCLIP;
  - timm MobileNetV4.
