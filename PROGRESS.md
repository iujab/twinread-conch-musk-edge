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
