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
  (rel 0) with `results/student/open_crc_rel05` (rel 0.5) on val clean tiles:

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
