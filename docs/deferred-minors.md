# Deferred minor issues (build ledger snapshot, 2026-10-17)

Minor findings from per-task reviews that were parked rather than fixed. Several were fixed later by the final review
fix wave and are kept here only for the record: the missing `logs/` dir at submit, `bridge.py`'s missing `struct`
import, the install_offline.sh platform tag, the npz git-ignore, the MUSK embed budget, and the unmeasured int8 flag
quality (now in `results/student/*/int8_flag_fidelity.json`). The rest are open, none blocking.

- Task 1: minor (deferred): test_patch_conch_idempotent leaves conch patched for the session (no restore fixture)
- Task 1: minor (deferred): tests import tools/make_tokenizer_fixture.py by path for STRINGS (coupling)
- Task 1: minor (deferred): pytest run shows 1 warning (output not pristine)
- Task 2: minor (deferred): close() via closure-cell mutation brittle; encode_* after close gives opaque TypeError
- Task 2: minor (deferred): close() memory test covers only quiltnet_b16
- Task 2: minor (deferred): test inserts organizer repo into sys.path permanently, imports generic module name `models`
- Task 2: minor (deferred): assert at readers.py:205-206 should be raise; torch_dtype= deprecated (use dtype=)
- Task 2: minor (deferred): MUSK transform adds .convert("RGB") (harmless)
- Task 3: minor (deferred): no test pins parquet label encoding to CRC_CLASSES (consistent with published CRC-VAL counts)
- Task 3: minor (deferred): crc_from_folder silently accepts missing class folders; no recursion; case-sensitive folder names; partial .npy on mid-run failure
- Task 3: minor (deferred): fetch_crc rewrites manifest, dropping crc_from_folder source_* entries
- Task 3: minor (deferred): _fetch_images leaves .scratch.npy on failure; load_split hard-codes seed 0
- Task 3: minor (deferred): injected-_shards path defaults test shards to train shards (tests only)
- Task 4: minor (deferred): chunk size not in cache key/key.txt; no unit-norm-after-fp16 test; loose resume assertion; k1-unavailable raise not in docstring; truncated .npy crash outside try
- Task 16: minor (deferred): logs/ must exist at submit; no timeout record (--signal trap); SBATCH_TIMELIMIT empty; J1 smoke rc not propagated; no set -e; weak exit-3 tests; unused `called` list; HF_TOKEN in shell history
- Task 16: minor (deferred): embed_loop stall metric can misread split completion as no progress; J4 --seed/--pair OPEN unverified vs Task 10/4 CLIs (Task 10 must accept --seed); report lacks pasted pytest output
- Task 12: minor (deferred): no-torch test is grep-only (use subprocess sys.modules check); x86 numbers noisy; fixed input/30 runs; no int8 drift check; JSON lacks trailing newline
- Task 5: minor (deferred): prereg incompleteness (ci n fixed at 1000; s4 coefs/fallback, epochs, arch, zs scale 100, conf=0.8 not recorded); tie-breaking in risk_coverage; NaN quantiles in bootstrap; torch.set_num_threads(4) missing; _labels duplicates load_split w/o length check; test passes full labels dict; unused out["_s4_test"]; gated-MLP val optimism note
- Task 13: minor (deferred): unbounded PNG cache; page assumes ≥1280 px; no HTTP/bind/camera tests; unused local f; MJPG/resolution/exposure not verified; unused --y; decode test marker ~85 px
- Task 5: minor (deferred): bootstrap_ci lacks len(groups) assert; _bank writes npy before sidecar; unused review.json-last comment; dead C assignment in test; prereg resume compares parsed JSON not bytes
- Task 13: minor (deferred): find_markers max_n=3 may crowd out true pair (safe rejection); stale "4096/12 bits" text in report
- Task 6: minor (deferred): non-square tests use ≤1-level tolerance (assert exact); no B>65535 test; builds.json lacks "executed": false + git rev (slide over-claim risk); cuda→triton fallback untested, _triton_ok import-only, triton/torch exact tests depend on ext fixture; bounds via assert; sanitizer + 49-shape sweep not committed; report interpolate error range 8–45 not 11–45; .cu header scalar-path comment
- Task 14: minor (deferred): bridge.py uses struct.pack without import (latent NameError; send_status swallows); _last_state unused; _fallback drops cap without release(); install_offline.sh platform tag likely wrong (manylinux_2_28_aarch64); no mid-run camera-drop or msgpack encoder test; awkward tissue test assert; verify_bundle_dir doesn't require the 5 R24 files; 3 Bridge calls/step with 1 s timeouts; after merge replace review.py path fallback with plain import
- Task 14: minor (deferred): real-tile tissue test skips silently without data; chroma 15 / max 30 magic numbers (tuned on normalised NCT-CRC; re-check on real rig frames); dim-white docstring claim not asserted
- Merge: minor (deferred): results/**/*.npz not git-ignored (student_targets/probes_aligned show untracked); 5 pytest warnings (torch.onnx.export deprecations) in merged suite
- Task 8: minor (deferred): possible QuickGELU mismatch for QuiltNet (official load path kept; post-hoc sensitivity only; embed cache key lacks model config); run_open.sh dead guard + not reproducible from scratch; PROGRESS wording (cam McNemar tile-level under cluster row; 10 slides/614-tile slide; base_A vs fused errors; pooled partner rule picks biomedclip on cam alone; s3 0.693 beats s4 on crc)
- Task 10: minor (deferred): budget doesn't reserve final eval phase; training-step OOM exits 1 not 3; aug=none ablation also drops flips and runs 5 vs 15 epochs (word README claim accordingly); no AUG_VERSION in config hash; thresholds calibrated on torch AMP outputs — Task 11 must recalibrate on exported ONNX val outputs (carry to Task 11 dispatch); add centred-cosine metric
- Task 11: minor (deferred): 99.5 collapse not recorded; minmax/entropy bit-identical unexplained; no explicit int8−fp32 delta / test flag-rate caveat in JSON note (README must surface 2.3-pt test drop and 16.4% test flag rate); eval-mode test near-vacuous; export_onnx mutates caller model; redundant .gitignore lines; label-id == class-index assumption
- Task 11: minor (deferred): quantize_s timing changes per rebuild; bundle rebuild needs git-ignored targets/probes npz (re-run analyze first); unicode classes path untested
- Task 9: minor (deferred): run_cm_local.sh hardcodes --readers musk (CONCH cmd not in repo); MUSK probe locked runs ~12.5 min > 10-min limit (use --budget ≤300, also on Delta); baselines scored vs fused errors not labelled, own-error variants not reported; cam McNemar tile-level under cluster row; s1 binary tie-break in index order at budgets > disagreement rate; run_cm_local.sh analyze lacks set -e
- Task 9: minor (deferred): dead line in post_hoc script; post_hoc JSON per-class recall indexed without class names
- Task 7: minor (deferred): PROGRESS "every op ≤1 level" should say "on the 8 fixture tiles"; torch pre-JPEG diff 0.098 near-black ill-conditioning undocumented; moiré phase π/3 vs stated 2π/3 rationale; k2_bench.json lacks methods block; DCT_F32_ERR is empirical; no B>65535 test; host int narrowing unchecked (unreachable); fixed-seed noise repetition not in camera_aug docstring
- Task 7: minor (deferred): regression test uses generic pytest.raises; exempt blocks up to 7 levels unasserted (q100 cap not asserted on exempt blocks); torch ε 0.2 could tighten with a near-black mask; exempt cap headroom thin (0.096% vs 0.1%)
- Task 8b: minor (deferred): OPEN bundle from K2 run (val doesn't justify switch; best epoch = last → maybe under-trained); int8 flag quality (agreement/kappa) unmeasured, CM JS thr 0.00082→0.0417 under int8; _load_group OOM path lacks gc.collect, k1 loads readers before kernel check, k1_group on partial resume; test hardening (flock shim, GPU parity test fails not skips w/o weights, misnamed fallback test); analyze_cpu threads 4 vs 16 CPUs, 1 h walltime unmeasured; model size only in bytes
- Task 18: minor (deferred): traceability test loose (map to headline.json keys); headline.py hard-codes mcu_flashed/real_camera_captures; smoke port-in-use false pass; smoke rewrites tracked headline.json; step 6/7 label; footer font 8.6-10 pt; fig2 dense; colour role reuse; abstract lacks explicit patient-benefit sentence; README:61 ND wording
