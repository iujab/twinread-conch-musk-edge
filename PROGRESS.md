# Progress log

- 2026-10-17: scaffold (gitignore, pytest.ini, conftest, compat shims for transformers v5).
- 2026-10-17: Task 2 readers (`twinread.readers`). Measured fix to the spec table: PathGen-CLIP is ViT-L/14-336, so its
  preprocess is 336/336 (not 224); the loader asserts the HF processor matches the static spec. All dims matched the
  table (pathgenclip 768/1024, quiltnet_b16 512/768, biomedclip 512/768). CONCH/MUSK match the organizer adapter
  (cos >= 0.9999 on images and texts). Fixture: tests/fixtures/tiles4.npy (CRC_VAL_HE_7K, classes 2/7/0/3).
- 2026-10-17: Task 3 data (`twinread.data`). Shards are class-sorted (CRC: one class per 3,226-row shard; Camelyon: sorted
  by label/centre), so sampling reads only the key columns remotely, picks random 100-row parquet groups per class
  (or per hospital x label) and range-reads just those groups; no shard stored. Labels in `1aurent/NCT-CRC-HE` already
  follow CRC_CLASSES order; Camelyon `center` 0-4 is the WILDS hospital id (2 = WILDS test centre), tiles are RGBA PNG
  96 px -> RGB. Results in `data/` (untracked): `crc_train` 20,007 x 224x224x3 (2,223 per class, 3.0 GB, 9 min fetch),
  `crc_test` = full CRC-VAL-HE-7K 7,180 (ADI 1338, BACK 847, DEB 339, LYM 634, MUC 1035, MUS 592, NORM 741, STR 421,
  TUM 1233; 1.08 GB), `cam` 7,500 x 96x96x3 (1,500 per hospital, 750 normal + 750 tumor each; 207 MB; ~10 min).
  load_split sizes: crc train/val/test 18,009 / 1,998 / 7,180; cam 5,400 / 600 / 1,500 (test = hospital 2).
  Caveat: group-level sampling means tiles within a 100-row group may share a slide (less slide diversity than row-level).
  Ruling R5 added: `fetch_crc(n_train=None)` = all 100k, `crc_from_folder` for the NCSA class-folder TIF layout.
- 2026-10-17: Task 3 fix round 1 (R10). cam splits are now by hospital: train 0/3/4 (4,500), val 1 (1,500), test 2 (1,500). CRC:
  `data/crc_train_g.npy` records the parquet group (shard*100000+row group; 209 groups) of every train-pool tile, rebuilt
  without re-downloading images by `reconstruct_crc_groups` (labels re-derived from the seeded plan must equal
  crc_train_y.npy; crc_train_x/y sha256 unchanged; 6 re-read tiles matched). `split_val(..., groups=)` makes val from whole
  groups, so crc train/val = 17,349 / 2,658, group-disjoint. Limitation: tiles within one 100-row group may come from the same
  slide, so val is group-disjoint but not guaranteed slide-disjoint; test is the separate CRC-VAL-HE-7K patient set.

- Task 5 (analysis library): gated-MLP head uses inputs [g*hA+(1-g)*hB, |hA-hB|, hA*hB] after per-reader linear maps to 256-d
  (a variant of the design's [zA, zB, zA.W, |zA-zB|]; accepted, R21). Train-split student targets are cross-fitted OOF (K=5);
  best single reader is chosen on val; probes_aligned.npz is stored in softmax form with `{A,B}_classes`; the pre-registration
  is crash-resumable only if the plan is byte-identical and no results exist; text-bank cache is keyed by reader/spec/prompt hash.
- 2026-10-17: Task 10 student (`twinread.student`, `scripts/train_student.py`). **Plan amendment R26:** the device student's
  `DEFAULT_WEIGHTS["rel"]` is 0.0 (the plan had 0.5). The choice rests on VAL evidence, comparing `results/student/open_crc`
  (rel 0; renamed `open_crc_torchaug` in Task 8b) with `results/student/open_crc_rel05` (rel 0.5) on val clean tiles:

  | metric | rel 0 | rel 0.5 |
  |---|---|---|
  | raw cos hA / hB | 0.970 / 0.976 | 0.774 / 0.768 |
  | centred cos (h−μ vs t−μ) | 0.793 / 0.776 | 0.744 / 0.728 |
  | probe KL(teacher‖student) A / B | 0.089 / 0.121 | 0.219 / 0.481 |
  | val-chosen JS threshold | 8.1e-3 | 6.4e-4 |
  | teacher-flag recall at sev 1.0 | 0.530 | 0.477 |
  | bal acc | 0.9884 | 0.9872 |

  - The rel 0.5 JS threshold is so small that int8 drift could move the flag rate.
  - Cause: the teacher embeddings sit in a narrow cone (unit-mean norm about 0.91 for A and 0.94 for B, an ad-hoc
    check of the train targets). Row-wise InfoNCE at τ 0.07 shrinks the component they all share.
  - The change is not test-favoured: test clean bal acc is slightly lower with rel 0 (0.9487 vs 0.9534).
  - InfoNCE stays available via `--loss-weights '{"rel": 0.5}'`, which uses BigBatch's Triton loss on CUDA.
  - K2 is not merged, so every `--aug k2` run used the plain-torch camera simulation (`aug_used` and `k2_fallback` in
    `metrics.json`).

## Task 8: OPEN pair (PathGen-CLIP + QuiltNet-B-16), local results (numbers copied from results/open/)
Partner chosen on val only (results/open/partner.json): quiltnet_b16 (mean val probe bal acc 0.9471, error phi vs PathGen-CLIP 0.365)
over biomedclip (0.9450, phi 0.405); both eligible (within 3 points). Test, reference-impl embeddings.
| dataset | A pathgenclip probe | B quiltnet_b16 probe | best single (val) | fused (primary) | delta vs best, 95% CI | McNemar fused vs best | CI mode |
|---|---|---|---|---|---|---|---|
| crc | 0.9435 [0.9376, 0.9490] | 0.9141 [0.9072, 0.9210] | A | gated_mlp 0.9443 [0.9381, 0.9501] | +0.0008 [-0.0052, +0.0066] | p=0.66 (b=125, c=133) | tile-level, anti-conservative |
| cam | 0.9733 [0.9197, 0.9857] | 0.9493 [0.8973, 0.9728] | A | concat_lr 0.9733 [0.9344, 0.9830] | 0.0000 [-0.0122, +0.0165] | p=1.0 (b=8, c=8) | cluster by slide |
Zero-shot bal acc (A / B): crc 0.815 / 0.423; cam 0.921 / 0.525. No `unstable` CI flag was raised anywhere.
Review flag, errors caught at 10% budget (crc / cam): s1 0.511 / 0.500; s2 0.502 / 0.425; s3 0.693 / 0.675; s4 0.607 / 0.700;
baselines A-confidence 0.633 / 0.725, B-confidence 0.460 / 0.375. Fusion gives no significant accuracy gain over PathGen-CLIP alone,
and s4 does not beat the best single reader's own confidence at 10% on either dataset (CIs overlap).

### Task 8 fix round 1 (corrections to the entry above)
- The "errors caught at 10% budget" figures above are top-10%-of-test rankings, NOT the deployed rule. The preregistered rule is "s4 > validation threshold at the 10% budget". Its actual test operating point (review.json, s4_val_threshold_operating_points, "0.1"): crc flags 26.3% of tiles (0.26337) and catches 85.9% of errors (0.8594), accuracy on cleared 0.9917; cam flags 5.7% (0.05667) and catches 62.5% (0.625), accuracy on cleared 0.9894.
- Cause: crc val is near ceiling (19 fused errors on val vs 313 on test), so s4 and its threshold were fit on 19 positives. crc val to test is also a distribution shift (val is group-disjoint NCT-CRC-HE-100K, test is CRC-VAL-HE-7K), so val-calibrated review thresholds over-flag on test.
- QuiltNet-B-16 zero-shot caveat: on cam it is a degenerate one-class predictor (1,461 of 1,500 test tiles predicted "tumor"). The Quilt paper's Table 9 reports 28.68% on NCT-CRC for this ViT-B/16 GPT/77 variant, so our 0.423 is consistent with the paper; its linear probe is strong (crc 0.914, cam 0.949).

## Task 9: CM pair (CONCH + MUSK), LOCAL-SUBSET DEV NUMBERS (copied from results/cm_local/CM/<ds>/*.json; final numbers come from Delta, Task 17)
Embeddings: `--impl reference` (K1 is not merged into lane-a, so the brief's k1 option did not apply). Test split, probe features; none of the CIs printed in this entry is flagged `unstable`.
| dataset | CONCH probe | MUSK probe | best single (val) | fused (primary) | delta vs best, 95% CI | McNemar fused vs best | CI mode |
|---|---|---|---|---|---|---|---|
| crc | 0.9413 [0.9356, 0.9473] | 0.9480 [0.9423, 0.9536] | B (MUSK) | average 0.9555 [0.9501, 0.9608] | +0.0074 [+0.0018, +0.0129] | p=0.0043 (b=154, c=107) | tile-level, anti-conservative (no slide ids) |
| cam | 0.9687 [0.9377, 0.9782] | 0.9587 [0.9246, 0.9713] | A (CONCH) | gated_mlp 0.9647 [0.9077, 0.9798] | -0.0040 [-0.0370, +0.0076] | p=0.45 (b=19, c=25) | cluster by slide |
Zero-shot bal acc (CONCH / MUSK): crc 0.767 / 0.809; cam 0.908 / 0.925. crc fusion is a small but CI-excluding-zero gain over MUSK alone (tile-level CI, so optimistic); cam shows no gain.
Review flag, errors caught at 10% budget (top-10% of test ranking), crc / cam: s1 0.590 / 0.302; s2 0.728 / 0.491; s3 0.803 / 0.811; s4 0.803 / 0.755; baselines CONCH-confidence 0.544 / 0.604, MUSK-confidence 0.699 / 0.491.
**Preregistered rule's actual test operating point** (s4 > val threshold at 10% budget; review.json `s4_val_threshold_operating_points["0.1"]`): crc flags 25.1% of tiles (0.2506), catches 97.5% of errors, accuracy on cleared 0.9989; cam flags 5.3% (0.0533), catches 58.5%, accuracy on cleared 0.9845. As in OPEN, crc val is near ceiling (val fused bal acc 0.9936) so the val-fit threshold over-flags on test; cam flags well under its 10% budget and so catches fewer errors than the ranked s4 figure (0.755).
MUSK embedding: reduction rule not applied. Measured MUSK rates (256 tiles, cam, otherwise idle GPU): aligned 37 tiles/s, probe (ms_aug) 8.2 tiles/s; crc probe projected ~55 min, over the 45-min trigger, but `run_pair_analysis` has no train-subset support for one kind (aligned and probe train rows must match labels and CONCH rows), so full-train probe was embedded instead (about 2 h wall under GPU sharing). No library change was made.

### Task 9 fix round 1 (R28; post-hoc descriptions of the numbers above, nothing re-selected; same local-subset dev status)
- **The review-flag gain is fused-model confidence, not disagreement.** s4 (the disagreement combiner) adds nothing over s3, the fused model's own uncertainty. crc: s3 and s4 both catch 0.803 of fused errors at 10% (AURC 0.00235 vs 0.00234); pure disagreement is weaker (s1 0.590, s2 0.728). On the test set s3 and s4 are almost the same ranking (Spearman 0.9999, top-10% overlap 99.7%; results/cm_local/CM/crc/post_hoc_class_candidates_s3s4.json). cam: s3 catches 0.811 vs s4 0.755, and s4 AURC (0.0066) is worse than s3 (0.0027) and than both single-reader baselines (0.0051 / 0.0056).
- **On cam the twin system does not beat CONCH alone.** Accuracy on cleared tiles at 10%: 0.9904 for the twin system (s4) vs 0.9911 for CONCH flagged by its own confidence (base_A_own_error; caught 0.745). Own-error baselines (each reader flagged by its own confidence against its own errors, review.json), caught at 10% crc / cam: CONCH 0.660 / 0.745, MUSK 0.741 / 0.661. The earlier base_A / base_B figures (0.544 / 0.699 crc, 0.604 / 0.491 cam) are the readers' confidence scored against the FUSED model's errors, so they are not like-for-like with the own-error ones. Against the own-error baselines on crc, s3/s4 (0.803) still lead (MUSK own 0.741), but the lead comes from the fused model's uncertainty.
- **The crc fusion gain (+0.0074) is fragile.** Per-class recall, fused minus MUSK: MUS -13.2 pts, STR +13.1, TUM +3.6, ADI +1.0; the gain is a MUS/STR trade. Validation chose `average` by 1-2 tiles (val correct tiles 2641 vs 2640 gated_mlp vs 2639 concat_lr, of 2658). Test delta vs MUSK by candidate: average +0.0074, concat_lr +0.0071, gated_mlp -0.0040 (the 0.0003-behind-on-val candidate). CI stays tile-level and anti-conservative (no slide ids).
- 2026-10-17: Task 6 K1 fused dual preprocess (`twinread.kernels`, `k1_triton`, `csrc/`). CUDA, Triton and torch paths
  share one host plan that mirrors Pillow `precompute_coeffs`/`normalize_coeffs_8bpc`; all three are bit-exact to
  `official(img).half()` for CONCH and MUSK (256 CRC tiles, mosaic frames, random 480x640/640x480/720x1280/225x301).
  CUDA hpass reads each input row once for both readers. Numbers: `results/kernels/{k1_bench,k1_parity,builds}.json`.
  Fixture: tests/fixtures/crc16.npy (CRC_VAL_HE_7K, all 9 classes, tools/make_crc_fixture.py).
- 2026-10-17: Task 7 K2 camera-of-screen augmentation (SIMULATION; `twinread.kernels.camera_aug` /
  `sample_camera_params`, `ref/camera_ops.py` numpy references, `k2_torch.py`, `csrc/k2_camera_aug.cu` in the K1
  module, K1-only rebuild if K2 fails to compile). 224 px, batch 256, severity 1, RTX 5060: CUDA 46,315 img/s
  (k2_fused 3.27 ms + k2_jpeg 2.17 ms) vs numpy 44 (1 proc) / 166 (4 procs), torch GPU 1,097 (same Philox stream) /
  2,421 (torch.randn noise). Every op <= 1 level vs numpy; composition mean <= 0.0008 levels; max up to 11 levels
  only in 8x8 blocks where a JPEG coefficient is within float32 error of a rounding boundary (<= 1 level in all other
  blocks, `camera_ops.explain_differences`). Added an optional `profile` field (colour-profile matrix) to
  CameraAugParams. sm_80/sm_90/sm_120 compile-checked only. Numbers:
  `results/kernels/{k2_bench,k2_parity,k2_figure,k2_sanitizer,builds}.json`; figure `figures/k2_severity_grid.png`.

## Task 8b: integration on lane-a (numbers copied from the JSON files named in each bullet)
- **Pair-level K1 (R12/R23).** `embed_dataset(impl="k1")` embeds both readers of a pair from one `dual_preprocess` launch
  per encode batch (both readers resident; lone reader = spec passed twice; pair OOM at load -> one at a time, warned).
  k1 sidecars record `impl`, `k1_backend` and `k1_group`; the cache key already contained impl (tested: a reference cache
  is never reused for k1). End to end, CONCH + MUSK aligned on 512 CRC test tiles, batch 16, through the library
  (`results/kernels/k1_embed_e2e.json`): reference 20.11 tiles/s, k1 22.38 tiles/s (x1.113); a 4-worker DataLoader that
  overlaps the PIL transforms with the GPU reaches 22.18 tiles/s (k1 x1.009 over it). **Where the time goes:** the model
  forward is 89.4% of the reference run (both models alone 22.48 tiles/s; CONCH fp32 48.76, MUSK fp16 41.58); PIL
  preprocessing is 9.5% of it (210.59 tiles/s for both readers), K1 preprocessing 0.05% of the k1 run (42,631 tiles/s incl.
  H2D, x202). So K1 removes the preprocessing cost, but the end-to-end ceiling with free preprocessing is x1.118: running
  both readers is cheap in preprocessing, not in forward passes. k1 vs reference cosine on the 512 tiles: min 0.999997
  (CONCH), 0.99999994 (MUSK). Shared GPU, estimate.
- **NCSA kit.** J2 embeds -> new `ncsa/analyze_cpu.sbatch` (bfqi-delta-cpu, cpu, 16 CPUs, 01:00:00; or locally) writes
  `results/CM/crc/{student_targets,probes_aligned}.npz` -> J3 trains with `--targets-dir` and its precheck asserts both
  files. `embed_loop.sh` caps `--budget` at 300 s per call for the CM pair (MUSK ms_aug probe chunks). Nothing in `ncsa/`
  passes loss weights (R26 default rel 0). Untested on Delta.
- **CM student, LOCAL-SUBSET DEV NUMBERS** (`results/student/cm_crc/metrics.json`; mobilenetv4_conv_small, 15 epochs, real K2
  for training and eval augmentation, `aug_used` = `eval_aug_used` = k2; targets `results/cm_local/CM/crc`; best epoch 13 on
  mean val bal acc over severities). Balanced accuracy, clean / K2 severity 0.5 / 1.0: val 0.9823 / 0.9830 / 0.9708; test
  (CRC-VAL-HE-7K) 0.9415 / 0.9311 / 0.9135. Cosine to the teacher embeddings, hA / hB: val clean 0.919 / 0.9644 (centred
  0.8493 / 0.834), test clean 0.8831 / 0.9505 (centred 0.7767 / 0.7552). Teachers on the same test tiles
  (`results/cm_local/CM/crc/readers.json`, `fusion.json`; frozen features + linear probes): CONCH 0.9413, MUSK 0.9480, fused
  (average) 0.9555. The student is fine-tuned end to end with labels (CE + KD), so it is not a like-for-like comparison and
  matching or beating frozen-feature probes would be expected, not evidence of being "better than the foundation models".
  Here it does not beat them: on clean test tiles it ties CONCH (0.9415 vs 0.9413) and is below MUSK (0.9480) and the
  fused reader (0.9555); at severity 1.0 (simulated camera filming a screen) it reaches 0.9135.
- **CM device flag** (torch AMP, val-chosen thresholds js 0.0008197, review 0.4126; `metrics.json`): flags 0.1001 of val and
  0.1432 / 0.157 / 0.2288 of test at clean / 0.5 / 1.0 (teacher flag 0.2506 of test); test clean agreement with the teacher
  flag 0.8118, kappa 0.4156, teacher recall 0.4102; it catches 0.7953 of the student's own test errors, accuracy on cleared
  tiles 0.9901.
- **CM int8 bundle** (`results/student/cm_crc/int8_drift.json`; ORT static QDQ, percentile99.99 picked on val drift; MinMax
  and Entropy collapsed, val label agreement 0.4406). Val: cos hA / hB mean 0.9933 / 0.9969 (min 0.8104 / 0.8900), label
  agreement 0.9887. Test: cos mean 0.9875 / 0.9953 (min 0.6133 / 0.8796), label agreement 0.9607. Balanced accuracy
  fp32 ONNX -> int8: val 0.9823 -> 0.9774, **test 0.9415 -> 0.9219**. Thresholds recalibrated on int8 val outputs:
  js 0.0008197 -> 0.0417, review 0.4126 -> 0.4298; int8 flag rate with the torch thresholds val 0.1317 / test 0.1765, with
  the recalibrated ones val 0.1001 / test 0.121. Size: model.onnx 6,012,392 bytes, bundle 6,139,199 bytes. `bundles/cm/` is
  `distributable: false` (CONCH/MUSK CC-BY-NC-ND) and git-ignored; checkpoints are not committed.
- **OPEN student retrained with the real K2** (`results/student/open_crc/`; the torch-fallback run is kept as
  `open_crc_torchaug`). Same recipe, best epoch 15. Bal acc clean / 0.5 / 1.0: val 0.985 / 0.9846 / 0.9677, test 0.9359 /
  0.9300 / 0.9014. The torch-aug run had test clean 0.9487 (val 0.9884), so the K2 run is lower on clean test (one seed
  each, not tested for significance; the torch-aug run's severity rows used the torch camera simulation, so they are not
  comparable). int8 (`int8_drift.json`,
  percentile99.99): test bal acc 0.9362 -> 0.9085 (torch-aug run: 0.9486 -> 0.9257), recalibrated js 0.0142, review 0.4089
  (committed `bundles/open/config.json` + `SHA256SUMS`), bundle 5,917,722 bytes.
- **K2 vs torch augmentation** (`results/student/{open_crc,cm_crc}/throughput.json`, batch 128, severity 0.5, CUDA events,
  shared GPU): augmentation path K2 25,046 / 25,038 vs torch 2,350 / 2,355 images/s (OPEN / CM); full training step K2
  3,902 / 3,880 vs torch 1,555 / 1,559 vs no augmentation 4,488 / 4,507 images/s.
- **Concern for the device:** int8 costs test bal acc for every student so far (fp32 ONNX -> int8: CM 0.9415 -> 0.9219,
  OPEN 0.9362 -> 0.9085, OPEN torch-aug 0.9486 -> 0.9257) much more than val (CM 0.9823 -> 0.9774, OPEN 0.985 -> 0.9769);
  calibration uses 512 clean train tiles and test is a different patient set. Not addressed here.

### Task 8b fix round 1 (R31; honesty additions, numbers copied from the JSON named in each bullet)
- **The deployed int8 CM student is below every teacher.** Its test bal acc is 0.9219 (`results/student/cm_crc/int8_drift.json`,
  int8). Every CM teacher scores higher on the same test tiles (`results/cm_local/CM/crc/readers.json`, `fusion.json`): the
  aligned-feature probes (CONCH 0.9355, MUSK 0.9310), the primary probes (CONCH 0.9413, MUSK 0.9480) and the fused reader
  (0.9555). Its value is keeping most of their accuracy at a small fraction of the size, offline, not beating them
  (`results/student/model_sizes.json`): 5,654,634 parameters (int8 ONNX 6,012,392 bytes) against 395,380,992 in the two
  teachers' vision towers (CONCH 90,392,832; MUSK 304,988,160), a ratio of 0.0143; the int8 file is 0.0028 of the
  teachers' checkpoint bytes (2,152,729,423). CONCH's count is its released checkpoint (200,522,497 parameters); the CoCa
  text decoder the library builds (194,710,272) has no released weights and is not counted.
- **Like-for-like comparison: aligned features.** The student distils the teachers' text-aligned embeddings, and the
  device classifies hA/hB with the aligned-feature probes, so those probes are its like-for-like reference. Test, clean:

  | | reader A | reader B | class head / fused |
  |---|---|---|---|
  | CM teachers, primary probes (frozen features; `readers.json` probe) | CONCH 0.9413 | MUSK 0.9480 | fused 0.9555 |
  | CM teachers, aligned-feature probes (`readers.json` aligned_probe) | CONCH 0.9355 | MUSK 0.9310 | – |
  | CM student fp32 (torch AMP), hA/hB through those same aligned probes (`metrics.json` probe_A/B_bal_acc) | 0.9419 | 0.9401 | class head 0.9415 |
  | CM student int8 (`int8_drift.json`) | – | – | class head 0.9219 |
  | OPEN teachers, primary probes (`results/open/OPEN/crc/readers.json`, `fusion.json`) | PathGen-CLIP 0.9435 | QuiltNet-B-16 0.9141 | fused 0.9443 |
  | OPEN teachers, aligned-feature probes | 0.943 | 0.9172 | – |
  | OPEN student fp32, hA/hB through the aligned probes (`open_crc/metrics.json`) | 0.9218 | 0.9062 | class head 0.9359 |
  | OPEN student int8 (`open_crc/int8_drift.json`) | – | – | class head 0.9085 |

  The fp32 CM student's hA/hB score slightly above the teachers' own aligned features through the same probes. Its trunk
  is fine-tuned with labels (CE + KD), so this is expected and is not evidence of better features. After int8 the
  deployed class head is below all of them.
- **K1 context** (`results/kernels/k1_embed_e2e.json`, `context`). The organizers' observed bottleneck was a per-image
  hosted server: CPU preprocessing in the uvicorn request handler, CONCH on one A40. Our x1.11 end-to-end figure comes
  from a different regime: a batched, GPU-forward-bound pipeline on an RTX 5060 with CONCH in fp32, where preprocessing
  is 9.5% of the reference run. The preprocessing speedup itself is x202. On faster GPUs or in per-image serving the
  preprocessing share is different, and neither regime was measured here.
