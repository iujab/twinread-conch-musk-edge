# Progress log

- 2026-10-17: scaffold (gitignore, pytest.ini, conftest, compat shims for transformers v5).
- 2026-10-17: Task 2 readers (`twinread.readers`). Measured fix to the spec table: PathGen-CLIP is ViT-L/14-336, so its
  preprocess is 336/336 (not 224); the loader asserts the HF processor matches the static spec. All dims matched the
  table (pathgenclip 768/1024, quiltnet_b16 512/768, biomedclip 512/768). CONCH/MUSK match the organizer adapter
  (cos >= 0.9999 on images and texts). Fixture: tests/fixtures/tiles4.npy (CRC_VAL_HE_7K, classes 2/7/0/3).
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
