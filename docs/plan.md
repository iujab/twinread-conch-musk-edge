# TwinRead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build TwinRead, plus the kit to run it on NCSA Delta:
- CONCH and MUSK (and an open-licence reader pair) as two independent readers;
- a fused reader;
- a disagreement review flag with triage curves and a shared-blind-spot analysis;
- custom CUDA kernels for dual preprocessing and camera-of-screen augmentation;
- a distilled int8 student;
- an offline Arduino UNO Q app with a USB camera and the LED matrix.

**Architecture:**
- **Teacher side.** A reader abstraction wraps each foundation model. Cached embeddings feed probes, fusion and
  review-flag analysis, and that analysis is pre-registered before any test evaluation.
- **Kernels.** CUDA C++ kernels (with Triton/torch fallbacks) sit in the data path. K1 feeds both readers from one
  pass. K2 simulates USB-camera capture of a screen for student training.
- **Student.** It is distilled from the teacher outputs, exported to ONNX int8, and runs on the UNO Q with numpy +
  onnxruntime only.

**Tech Stack:**
- Python 3.12 bench `.venv`: torch 2.14.1+cu130, transformers 5.19, timm 1.0.30, open_clip 3.3, scikit-learn,
  onnx + onnxruntime.
- CUDA 13.1 (`/usr/local/cuda`), Triton 3.8.
- `conch`/`musk` git packages installed `--no-deps`.
- On the device: Debian aarch64, onnxruntime, opencv-python-headless, numpy, Pillow.
- MCU: Arduino sketch via `arduino-cli` 1.5.1.
- Delta: `pytorch-conda/2.8` + Slurm.

**Spec:** `twinread-conch-musk-edge/docs/design.md` (approved 2026-10-17). Read it first. This plan argues from it.

## Global Constraints

- **Project root:** `/home/nanashi-mumei/dev/ashbyhackathonbench/twinread-conch-musk-edge/` (below, `$P`). Bench root
  `/home/nanashi-mumei/dev/ashbyhackathonbench/` (`$B`). Python is always `$B/.venv/bin/python`.
- **Stay inside `$P`.** Read other bench projects only. The exception is Task 1's `uv pip install --no-deps` of
  `conch`, `musk`, `fairscale` and `sentencepiece` into `$B/.venv`. Never upgrade or downgrade torch, timm,
  transformers or open_clip.
- **GPU work:**
  - Wrap every GPU command in `flock $B/.gpu.lock`, with each locked job ≤ 10 min, resumable and cached.
  - `torch.set_num_threads(4)`; `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`; compile with `MAX_JOBS=4`.
  - ~4 GB VRAM free: fp16, batch ≤ 16 for ViT-L/MUSK. Catch OOM and halve the batch.
- **CUDA builds:** `PATH=/usr/local/cuda/bin:$PATH`. `TORCH_CUDA_ARCH_LIST` defaults to `12.0` locally. The sm_80
  build is compile-checked locally and JIT-built on Delta. Build dirs go under `$P/build/`. Nsight counters are
  unavailable, so time kernels with CUDA events (warmup 3, median of 10).
- **Data:** downloads go to `$P/data/` (untrusted; read with `-I` where practical; never commit). Results go to
  `$P/results/` and figures to `$P/figures/` (2-8 PNGs, 1200-1600 px wide, dpi ≥ 120).
- **Ports:** the host dashboard is **127.0.0.1:1777**; the device UI is 127.0.0.1:8080 on the UNO Q.
- **Honesty:**
  - Every number in README/project.json/figures is traceable to a file in `results/`.
  - Test-set metrics are computed only after `results/<pair>/<dataset>/preregistration.json` exists, and that file
    is never edited afterwards.
  - Label simulation, "camera filming a screen" and "estimate" wherever they apply.
  - Research prototype, not a diagnostic device, on every UI and in the README.
- **Licences:** the CM (CONCH+MUSK) student/bundle is never put in a distributable location. The OPEN bundle is the
  distributable one.
- **NCSA:** nothing runs on Delta without the user's explicit OK in chat, per command. Scripts in `ncsa/` are written
  and tested locally only.
- **Tests:**
  - `$B/.venv/bin/python -m pytest $P/tests -q` passes in < 2 min with `CUDA_VISIBLE_DEVICES=""`, offline.
  - GPU tests carry `@pytest.mark.gpu` and skip when CUDA or the extension is unavailable.
  - The GPU suite runs under the lock and only when asked for: `flock $B/.gpu.lock env TWINREAD_GPU_TESTS=1 PATH=/usr/local/cuda/bin:$PATH $B/.venv/bin/python -m pytest $P/tests -q -m gpu` (without `TWINREAD_GPU_TESTS=1` gpu tests skip; reader tests skip when the weights are not cached).
- **Targets from the spec:**
  - K1 embedding cosine vs the reference transform ≥ **0.999** per reader;
  - student latency ≤ **250 ms** per 224-px tile at 4 threads on the UNO Q;
  - Delta GPU use ≤ **10 A100-hours** (budget 30);
  - each Delta job ≤ 1.5 h.
- **Commits:** `$P` is its own local git repo (Task 1). Commit after every task. Never commit `data/`, `build/` or
  bundles of the CM pair.

## Review Focus

1. **Non-square or large inputs** (camera frames 640×480 or 1280×720; tiles larger than 224) through K1 and the device
   crop. Expected: the same result as the reference transform (resize the shorter side, then center crop), and no
   out-of-bounds reads. Pinned in Task 6 (`test_k1_nonsquare_matches_reference`) and Task 14
   (`test_crop_from_frame_shapes`).
2. **Blank glass or near-uniform frames** (camera pointed at nothing, background tiles). Expected: the device reports
   `no_tissue=True` and review, never a confident tissue class; no NaN/Inf from the kernels. Pinned in Task 7
   (`test_k2_uniform_input_finite`) and Task 14 (`test_blank_frame_is_no_tissue`).
3. **Batch tails:** batch of 1, batch not a multiple of the block size, empty batch. Expected: correct output or an
   empty tensor, never a crash. Pinned in Task 6 (`test_k1_batch_tails`) and Task 7 (`test_k2_batch_tails`).
4. **Stale caches:** cached embeddings after the reader, kind, preprocessing implementation or input data changed.
   Expected: the cache is invalidated and recomputed, never silently reused. Pinned in Task 4
   (`test_cache_invalidated_on_spec_change`).
5. **Device peripherals missing:** no camera, camera unplugged mid-run, no MCU Bridge. Expected: the app falls back
   to replay mode or continues without the LED, logs the reason and keeps serving the UI. Pinned in Task 14
   (`test_camera_missing_falls_back_to_replay`, `test_bridge_absent_noop`).

---

## Execution lanes (what can run in parallel)

- **Lane A (science, sequential):** Tasks 2 → 3 → 4 → 5 → 8 → 9.
- **Lane B (kernels):** Tasks 6, 7. They depend only on Task 1 and Task 2's `PreprocessSpec`.
- **Lane C (edge):** Task 10 (student; needs Tasks 5 and 7, falls back to torch augmentation if Task 7 isn't done) →
  Task 11 (export) → Task 14 (device app).
- **Lane D (independent):** Task 12 (backbone latency harness), Task 13 (screen-capture rig), Task 16 (NCSA kit).
- **Hardware-gated:** Task 15 (UNO Q bring-up) waits for the user to plug the board in.
- **User-gated:** Task 17 (Delta runs) needs the user's SSH terminal and a per-command OK.
- **Last:** Task 18 (dashboard, figures, README, manifest, showcase).

---

### Task 1: Scaffold, environment and the transformers-v5 compatibility shims

**Files:**
- Create: `$P/.gitignore` (`data/ build/ bundles/cm/ __pycache__/ *.onnx` except `tests/fixtures/*.onnx`),
  `$P/pytest.ini` (markers: `gpu`; testpaths `tests`), `$P/src/twinread/__init__.py`, `$P/src/twinread/compat.py`,
  `$P/tests/conftest.py` (adds `$P/src` and `$P/device` to `sys.path`; a `gpu` skip helper),
  `$P/tests/test_compat.py`, `$P/tests/fixtures/tokenizer_v4_ids.json`, `$P/tools/make_tokenizer_fixture.py`,
  `$P/PROGRESS.md`.

**Interfaces:**
- Produces:
  - `twinread.compat.conch_tokenize(tokenizer, texts: list[str]) -> torch.LongTensor [N,128]`: identical to the
    original `conch.open_clip_custom.custom_tokenizer.tokenize` (max_length 127 + one pad).
  - `twinread.compat.patch_conch() -> None`: idempotent; replaces `tokenize` in `conch.open_clip_custom` and in
    `conch.open_clip_custom.custom_tokenizer`.
  - `twinread.compat.XLMRSpTokenizer(vocab_file: str)` with `.encode(text) -> list[int]` and attributes
    `bos_token_id=0, pad_token_id=1, eos_token_id=2, unk_token_id=3`. The rule: piece id `i` → `i+1`, except `0` →
    `3`, then wrapped in bos/eos. These are the transformers-v4 slow `XLMRobertaTokenizer` semantics MUSK expects.

- [ ] **Step 1:** `git init` in `$P`, then write `.gitignore`, `pytest.ini`, `conftest.py` and `PROGRESS.md` (a dated
  log; first entry: scaffold).
- [ ] **Step 2:** Install the FM packages into the bench venv **without deps**:
  `uv pip install --python $B/.venv/bin/python --no-deps "git+https://github.com/mahmoodlab/CONCH.git" "git+https://github.com/lilab-stanford/MUSK.git" fairscale sentencepiece`.
  Verify that `$B/.venv/bin/python -c "import torch, timm, transformers; print(torch.__version__, timm.__version__, transformers.__version__)"`
  still prints `2.14.1+cu130 1.0.30 5.19.0`.
- [ ] **Step 3: Generate the reference fixture with a throwaway transformers-v4 env.**
  - `tools/make_tokenizer_fixture.py` runs in a scratch venv:
    `uv venv /tmp/.../tok4 && uv pip install --python /tmp/.../tok4/bin/python "transformers==4.46.3" sentencepiece tokenizers`.
  - It tokenizes 40 fixed strings with CONCH's original `tokenize` (`PreTrainedTokenizerFast` + `batch_encode_plus`)
    and MUSK's original `XLMRobertaTokenizer(spm_path)` + `musk.utils.xlm_tokenizer(max_len=100)`.
  - The strings are: pathology prompts, a 300-char string (truncation), `""`, `"H&E"`, unicode `"µm"`,
    `"pN0 (0/12)"`, uppercase.
  - It writes `tests/fixtures/tokenizer_v4_ids.json` as `{"conch": [[...]], "musk_ids": [[...]], "musk_pad": [[...]]}`.
  - The script lives in `tools/`; only its output JSON is committed.
- [ ] **Step 4: Write the failing tests** in `tests/test_compat.py`:

```python
def test_conch_tokenize_matches_v4_fixture(): assert conch_tokenize(get_tokenizer(), STRINGS).tolist() == FIX["conch"]
def test_xlmr_shim_matches_v4_fixture():
    ids, pads = zip(*[musk_xlm_tokenizer(s, XLMRSpTokenizer(SPM), max_len=100) for s in STRINGS])
    assert [list(i) for i in ids] == FIX["musk_ids"] and [list(p) for p in pads] == FIX["musk_pad"]
def test_patch_conch_idempotent(): patch_conch(); patch_conch(); assert conch.open_clip_custom.tokenize is twinread.compat.conch_tokenize
```

  `SPM` is `hf_hub_download("xiangjx/musk","tokenizer.spm")` from the HF cache (offline). Skip with a reason if it is
  not cached.
- [ ] **Step 5:** Run `$B/.venv/bin/python -m pytest $P/tests/test_compat.py -q`. Expected: FAIL (import error).
- [ ] **Step 6:** Implement `compat.py` per the Interfaces. This shim was validated in a spike on 2026-10-16:
  `tokenizer(texts, max_length=127, truncation=True, padding="max_length", return_tensors="pt")` then
  `F.pad(..., (0,1), value=pad_id)`; and the XLM-R offset rule above.
- [ ] **Step 7:** Run the test. Expected: 3 passed.
- [ ] **Step 8:** `git add -A && git commit -m "scaffold + transformers-v5 tokenizer shims for CONCH/MUSK"`.

### Task 2: Readers

**Files:** Create `$P/src/twinread/readers.py` and `$P/tests/test_readers.py`.

**Interfaces:**
- Consumes: `twinread.compat.patch_conch`, `XLMRSpTokenizer`.
- Produces:

```python
@dataclass(frozen=True)
class PreprocessSpec:
    size: int; crop: int; resample: str        # "bicubic" (PIL semantics, a=-0.5) or "bilinear"
    mean: tuple[float, float, float]; std: tuple[float, float, float]
@dataclass
class Reader:
    name: str; spec: PreprocessSpec; dim_aligned: int; dim_probe: int
    preprocess_pil: Callable[[PIL.Image.Image], torch.Tensor]       # the model's official transform
    encode_aligned: Callable[[torch.Tensor], np.ndarray]            # float32, L2-normalized, [B, dim_aligned]
    encode_probe: Callable[[torch.Tensor], np.ndarray]              # float32, [B, dim_probe]
    encode_texts: Callable[[list[str]], np.ndarray]                 # float32, L2-normalized, [N, dim_aligned]
READER_NAMES = ("conch", "musk", "pathgenclip", "quiltnet_b16", "biomedclip")
def load_reader(name: str, device: str = "cuda") -> Reader
def spec_of(name: str) -> PreprocessSpec          # no weights needed; static table, verified against loaded transforms by a GPU test
```

- **Spec table and dimensions.** The CONCH and MUSK values were measured in the 2026-10-16 spike. The open readers'
  dims are to be confirmed by the test.

| name | size/crop | resample | mean / std | dim_aligned | dim_probe | probe feature |
|---|---|---|---|---|---|---|
| conch | 448/448 | bicubic | OpenAI CLIP (0.48145466, 0.4578275, 0.40821073)/(0.26862954, 0.26130258, 0.27577711) | 512 | 512 | `encode_image(proj_contrast=False, normalize=False)` |
| musk | 384/384 | bicubic | (0.5,0.5,0.5)/(0.5,0.5,0.5) | 1024 | 2048 | `ms_aug=True, with_head=False, out_norm=False` |
| pathgenclip | 224/224 | from `CLIPProcessor` | from processor | 768 | 1024 | `vision_model(...).pooler_output` |
| quiltnet_b16 | 224/224 | from open_clip transform | from transform | 512 | 768 | visual trunk before `proj` |
| biomedclip | 224/224 | from open_clip transform | from transform | 512 | 768 | visual trunk before `proj` |

- **Aligned semantics.** CONCH and MUSK follow the organizer adapters exactly (`ashby-hackathon-2026/models.py`):
  - CONCH: `encode_image(proj_contrast=True, normalize=True)`.
  - MUSK: fp16, `with_head=True, out_norm=True, ms_aug=False, return_global=True`. Text uses
    `musk.utils.xlm_tokenizer(max_len=100)` with `XLMRSpTokenizer`.

- [ ] **Step 1:** Write tests:
  - `test_spec_table_matches_loaded_transform` (gpu): for each reader, `load_reader(n).preprocess_pil` contains a
    Resize of `spec.size`, a CenterCrop of `spec.crop` and a Normalize with `spec.mean/std`.
  - `test_reader_dims` (gpu): encode 4 fixture tiles; assert the shapes match the table, and aligned norms equal 1 to
    within 1e-3.
  - `test_matches_organizer_adapter` (gpu; conch and musk only): the cosine between `encode_aligned` and the
    organizer adapter's `encode_images` on the same preprocessed batch is ≥ 0.9999 (import
    `$B/ashby-hackathon-2026/models.py` via `sys.path`; apply `patch_conch()` and the XLM-R shim first).
  - `test_spec_of_no_weights` (cpu): `spec_of("conch").size == 448`, and `spec_of("musk").mean == (0.5, 0.5, 0.5)`.
- [ ] **Step 2:** Add 4 real CRC tiles as `tests/fixtures/tiles4.npy` (uint8 [4,224,224,3]), from the data task's
  first download or from `1aurent/NCT-CRC-HE` streaming. Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `readers.py`. Load models lazily. Free GPU memory with `del` +
  `torch.cuda.empty_cache()` in a `Reader.close()`, and add `close()` to the dataclass.
- [ ] **Step 4:** Run the CPU tests, then the GPU tests under the lock. Expected: all pass. If an open reader's
  `dim_probe` differs from the table, fix the table from the measured value and note it in PROGRESS.md.
- [ ] **Step 5:** Commit.

### Task 3: Data

**Files:** Create `$P/src/twinread/data.py` and `$P/tests/test_data.py`.

**Interfaces:**
- Produces:

```python
CRC_CLASSES = ("ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM")
def fetch_crc(out_dir: Path, n_train: int = 20000, seed: int = 0) -> dict
    # writes crc_train_x.npy uint8[N,224,224,3], crc_train_y.npy int64, crc_test_x.npy, crc_test_y.npy (full CRC-VAL-HE-7K),
    # crc_manifest.json {source, shards, n_train, n_test, class_counts, seed, sha256 of arrays}
def fetch_camelyon(out_dir: Path, per_hospital: int = 1500, seed: int = 0) -> dict
    # cam_x.npy uint8[N,96,96,3], cam_y.npy (0 normal / 1 tumor), cam_h.npy (hospital 0-4), cam_manifest.json
def split_val(y: np.ndarray, frac: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]   # stratified (train_idx, val_idx)
def load_split(out_dir: Path, dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]          # dataset in {"crc","cam"}; split in {"train","val","test"} (cam: test = hospital 2)
```

- **Sources:**
  - `1aurent/NCT-CRC-HE` parquet: `data/NCT_CRC_HE_100K-*.parquet` (31 shards, ~0.3-0.5 GB each) and
    `data/CRC_VAL_HE_7K-*.parquet` (3 shards).
  - Download only the shards needed for a stratified 20k subset. Read each shard's label column first, and stop once
    every class has its quota of ⌈20000/9⌉.
  - Camelyon17-WILDS: `wltjr1007/Camelyon17-WILDS`. Inspect the schema first; ≤ 1.5 GB.
  - If `huggingface_hub` hangs, use curl on the `resolve/main` URL.
- [ ] **Step 1:** Write tests on a tiny synthetic parquet fixture (`tests/fixtures/mini_crc.parquet`, 27 rows, 3 per
  class, 224×224 PNG bytes):
  - `test_split_val_stratified`: each class appears in val, and train/val are disjoint.
  - `test_fetch_crc_from_local_shards(tmp_path)`: with `fetch_crc(..., _shards=[fixture])`, the arrays have the
    right dtype and shape, and the manifest class counts sum to N.
  - `test_load_split_test_is_crc_val`: the test split comes only from CRC_VAL shards.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. A private `_shards` parameter allows fixture injection; images are decoded with PIL into
  RGB uint8.
- [ ] **Step 4:** Run the tests. Expected: pass. Then run the real fetches:
  `$B/.venv/bin/python -c "from twinread.data import *; fetch_crc(Path('$P/data')); fetch_camelyon(Path('$P/data'))"`.
  Record the sizes in PROGRESS.md.
- [ ] **Step 5:** Commit (code + tests only).

### Task 4: Embedding cache

**Files:** Create `$P/src/twinread/embed.py` and `$P/tests/test_embed.py`.

**Interfaces:**
- Consumes: `load_reader`, `Reader`, `load_split`, and optionally `twinread.kernels.dual_preprocess` (Task 6).
- Produces:

```python
def cache_key(reader_name: str, kind: str, spec: PreprocessSpec, impl: str, x: np.ndarray) -> str
    # sha256 over (reader_name, kind, spec, impl, x.shape, sha256(x[:64].tobytes()+x[-64:].tobytes()))
def embed_array(reader: Reader, x: np.ndarray, kind: Literal["aligned","probe"], out_path: Path,
                batch: int = 16, impl: Literal["reference","k1"] = "reference") -> np.ndarray   # float16 saved, float32 returned
def embed_dataset(pair: str, dataset: str, kinds=("aligned","probe"), impl="reference") -> dict   # CLI entry; writes $P/data/emb/<dataset>/<reader>_<kind>_<split>.npy + .json
```

- `impl="reference"` runs each reader's official `preprocess_pil` on PIL images (what the organizers' server does).
  `impl="k1"` uploads uint8 to the GPU and calls `dual_preprocess` (Task 6). That matters only when both readers of
  a pair are embedded in the same pass, so `embed_dataset` takes a pair.
- Resumable: chunks of 2,048 tiles saved as `.part` files and concatenated at the end; each invocation stays ≤ 10 min
  of GPU (it processes chunks until a time budget, default 540 s, is used, then exits with code 3 meaning "rerun me").
- CLI: `python -m twinread.embed --pair OPEN|CM --dataset crc|cam [--impl reference|k1]`.
- [ ] **Step 1:** Write tests with a fake reader (cpu; `encode_*` = mean-pool):
  - `test_embed_array_shapes_and_dtype`.
  - `test_cache_hit_skips_compute`: a counter on the fake encode stays the same on the second call.
  - `test_cache_invalidated_on_spec_change`: change `spec.size`, and the encode counter increases. **This is Review
    Focus 4.**
  - `test_resume_from_parts(tmp_path)`: a simulated exit 3 is followed by completion; the result equals a
    single-pass run.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: pass. Commit.

### Task 5: Readers' outputs, fusion, review flag, blind spots and statistics (the analysis library)

**Files:**
- Create in `$P/src/twinread/`: `prompts.py`, `probes.py`, `fusion.py`, `review.py`, `blindspots.py`, `stats.py`,
  `analyze.py`.
- Create `$P/prompts/crc.json`, `$P/prompts/cam.json` and `$P/prompts/templates.json`.
- Tests: `$P/tests/test_analysis.py`.

**Interfaces:**
- Produces:

```python
# prompts.py
def load_prompts(dataset: str) -> tuple[list[str], dict[str, list[str]], list[str]]   # classes, synonyms per class, templates
def text_bank(reader: Reader, dataset: str) -> np.ndarray          # [C, dim_aligned], mean of template×synonym embeddings, L2-normalized
def zeroshot_probs(z: np.ndarray, bank: np.ndarray, scale: float = 100.0) -> np.ndarray
# probes.py
def fit_probe(z_tr, y_tr, z_val, y_val, Cs=(0.01, 0.1, 1.0, 10.0), seed=0) -> "Probe"   # standardize + LogisticRegression(max_iter=3000); C by val balanced accuracy
class Probe: def predict_proba(self, z) -> np.ndarray; C: float; classes_: np.ndarray
# fusion.py
def fuse_average(pA, pB) -> np.ndarray
def fit_concat_lr(zA_tr, zB_tr, y_tr, zA_val, zB_val, y_val) -> "Fused"
def fit_gated_mlp(zA_tr, zB_tr, y_tr, zA_val, zB_val, y_val, seed=0, epochs=30) -> "Fused"   # torch, CPU ok; inputs [zA, zB, |zA_p - zB_p|] after per-reader linear maps to 256-d; dropout 0.2
class Fused: def predict_proba(self, zA, zB) -> np.ndarray; name: str
# review.py
def signals(pA, pB, pF) -> dict[str, np.ndarray]     # keys: "s1_disagree" (0/1), "s2_js" (JS divergence, base 2), "s3_fused_unc" (1 - max pF), "base_A" (1-max pA), "base_B" (1-max pB)
def fit_s4(sig_val: dict, err_val: np.ndarray) -> Callable[[dict], np.ndarray]   # logistic on s1..s3
def risk_coverage(score, correct) -> dict             # sort by score ascending (most confident first); arrays coverage, risk; float aurc
def errors_caught(score, correct, budgets=(0.05, 0.10, 0.20)) -> dict  # per budget: fraction of errors in top-score budget, acc of cleared
def threshold_for_budget(score_val, budget) -> float
# blindspots.py
def blindspot_table(predA, predB, y, classes) -> pd.DataFrame   # per class: both_right, A_only, B_only, both_wrong, both_wrong_same_label
def confident_shared_errors(pA, pB, y, conf=0.8) -> np.ndarray   # indices
# stats.py
def balanced_accuracy(y, pred) -> float; def macro_f1(y, pred) -> float
def bootstrap_ci(fn, *arrays, n=1000, seed=0, alpha=0.05) -> tuple[float, float]
def mcnemar_exact(correct_a, correct_b) -> dict    # {"b": int, "c": int, "p": float} two-sided exact binomial
# analyze.py
def write_preregistration(pair, dataset, results_dir) -> Path   # freezes: signals, baselines, budgets, fusion candidates, primary fusion = chosen on val, primary flag = s4, CIs; includes sha256 of prompts/*.json; refuses to overwrite
def run_pair_analysis(pair: str, dataset: str, results_dir: Path, partner: str | None = None) -> dict
    # needs embeddings from Task 4. Order: fit on train/val -> write_preregistration -> evaluate test exactly once ->
    # results/<pair>/<dataset>/{readers.json, fusion.json, review.json, blindspots.csv, preregistration.json, student_targets.npz}
    # student_targets.npz: tA_*, tB_* (aligned embeddings), fused_logits_* (log of primary fused probs), review_* (0/1: s4 > val threshold at 10% budget), y_* for train/val/test
    # probes_aligned.npz: per reader A/B a probe fitted on the ALIGNED train embeddings (the sensitivity analysis), stored as arrays
    #   {A_coef, A_intercept, A_mean, A_scale, B_coef, ...}. The device applies these to the student's hA/hB (≈ tA/tB).
```

- [ ] **Step 1:** Write the tests (cpu, synthetic arrays with known answers):
  - `test_js_divergence_identical_is_zero` and `test_js_divergence_disjoint_is_one`.
  - `test_risk_coverage_perfect_ranking`: errors scored highest give aurc equal to the analytic value.
  - `test_errors_caught_budget`: 10 errors out of 100, all ranked in the top 10% → `caught[0.10] == 1.0`.
  - `test_mcnemar_known_values`: b=10, c=2 → p ≈ 0.0386 (two-sided exact).
  - `test_bootstrap_ci_contains_point`.
  - `test_blindspot_table_counts`.
  - `test_fuse_average_rows_sum_to_one`.
  - `test_fit_concat_lr_beats_chance_on_separable`.
  - `test_preregistration_refuses_overwrite(tmp_path)`.
  - `test_run_pair_analysis_test_evaluated_once(tmp_path)`: monkeypatch the test-metric function to count calls;
    expect exactly 1, called only after the prereg file exists.
- [ ] **Step 2:** Write `prompts/*.json` before any real-data run:
  - CRC: per class 3-5 synonyms (e.g. TUM: "colorectal adenocarcinoma epithelium", "tumor epithelium", "malignant
    glands"; BACK: "background", "blank glass with no tissue"…).
  - Camelyon: "lymph node tissue" / "metastatic breast carcinoma in lymph node" families.
  - Templates: CONCH-style, e.g. "an H&E image of {}", "{}", "a histopathology image showing {}", "an image of {}".
- [ ] **Step 3:** Run the tests. Expected: FAIL.
- [ ] **Step 4:** Implement.
- [ ] **Step 5:** Run. Expected: pass. Commit.

### Task 6: K1 fused dual-preprocess kernel (CUDA C++ + Triton baseline)

**Files:**
- Create in `$P/src/twinread/`: `csrc/k1_dual_preprocess.cu`, `csrc/bindings.cpp`, `kernels.py`, `k1_triton.py`.
- Create `$P/bench/bench_k1.py`.
- Tests: `$P/tests/test_k1.py`.

**Interfaces:**
- Consumes: `PreprocessSpec` (Task 2).
- Produces:

```python
def load_ext(arch_list: str | None = None, verbose: bool = False) -> types.ModuleType | None   # JIT build into $P/build/ext_<arch>; None + logged reason on failure
def dual_preprocess(x: torch.Tensor, specA: PreprocessSpec, specB: PreprocessSpec,
                    impl: Literal["cuda","triton","torch"] = "cuda") -> tuple[torch.Tensor, torch.Tensor]
    # x: uint8 [B,H,W,3] on CUDA (any H,W). Returns fp16 NCHW [B,3,specA.crop,specA.crop], [B,3,specB.crop,specB.crop].
    # Semantics = official transform on a PIL image: resize the SHORTER side to spec.size with PIL's resampling
    # (bicubic a=-0.5, antialiased support scaling when downsampling), center crop spec.crop, /255, normalize.
    # impl falls back cuda -> triton -> torch automatically if the extension is unavailable (warns once).
```

- **Algorithm (the part the signature doesn't determine): match PIL `ImagingResample` exactly.**
  - Separable, two passes: horizontal into a **uint8 intermediate with PIL's fixed-point rounding**
    (`PRECISION_BITS = 32-8-2 = 22`; coefficients normalized, then scaled by `1<<22` and rounded; accumulate in
    int32; clip to 0..255), then vertical.
  - Filter support = `2.0 * max(scale, 1)` for bicubic (`a=-0.5`); `center = (i + 0.5) * scale`.
  - Coefficients are precomputed per output row/column on the host (tiny tensors), and the kernel does the
    multiply-accumulate.
  - **One launch** covers both outputs: `blockIdx.z ∈ {0,1}` selects spec A or B (two passes = two launches total,
    each covering both readers). Crop offsets and normalization are fused into the vertical pass, which writes fp16.
- [ ] **Step 1:** Write the tests:
  - `test_k1_matches_pil_square` (gpu): 16 CRC fixture tiles. For CONCH and MUSK specs, the max absolute difference
    against the official transform output (Normalize applied), compared in units of one uint8 level divided by std,
    is ≤ 1 level.
  - `test_k1_nonsquare_matches_reference` (gpu): random uint8 frames of 480×640 and 720×1280 against the PIL
    reference, same tolerance. **Review Focus 1.**
  - `test_k1_batch_tails` (gpu): B ∈ {0, 1, 3, 33}; B=0 returns empty tensors with the right trailing shape.
    **Review Focus 3.**
  - `test_k1_embedding_cosine` (gpu, slow, marker `gpu`): CONCH and MUSK aligned embeddings from K1 vs the reference
    transform on 32 tiles: min cosine ≥ 0.999.
  - `test_triton_and_torch_paths_agree` (gpu): ≤ 1 level vs CUDA.
  - `test_fallback_when_ext_missing` (cpu-safe): monkeypatch `load_ext` to return None; `impl="cuda"` warns and
    falls back to torch.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement the CUDA kernel, bindings and `kernels.load_ext`
  (`torch.utils.cpp_extension.load(name=f"twinread_ext_{arch}", sources=[...], build_directory=..., extra_cuda_cflags=["-O3"])`).
  Keep exact math: **do not pass `--use_fast_math`**.
- [ ] **Step 4:** Implement the Triton and torch fallbacks (same coefficients).
- [ ] **Step 5:** Run all K1 tests under the lock. Expected: pass.
- [ ] **Step 6: Benchmark.** `bench/bench_k1.py` measures end-to-end dual-reader preprocessing tiles/s for:
  - (a) PIL + official transforms, per image, in one process (the organizer-server style);
  - (b) a `DataLoader` with 4 and 8 workers;
  - (c) torch GPU ops;
  - (d) Triton;
  - (e) CUDA K1.

  Use batch 256 of 224-px CRC tiles and 640×480 frames. Also report achieved GB/s vs a `cudaMemcpy` device-to-device
  baseline. Writes `results/kernels/k1_bench.json` with an env block (GPU, driver, CUDA, torch).
- [ ] **Step 7:** Compile-check sm_80:
  `TORCH_CUDA_ARCH_LIST="8.0" $B/.venv/bin/python -c "from twinread.kernels import load_ext; assert load_ext('8.0')"`
  (compile only). Record the result in `results/kernels/builds.json`.
- [ ] **Step 8:** Commit.

### Task 7: K2 fused camera-of-screen augmentation kernel

**Files:**
- Create `$P/src/twinread/csrc/k2_camera_aug.cu` and `$P/src/twinread/ref/camera_ops.py` (numpy references).
- Extend `$P/src/twinread/kernels.py`.
- Create `$P/bench/bench_k2.py`.
- Tests: `$P/tests/test_k2.py`.

**Interfaces:**
- Produces:

```python
@dataclass
class CameraAugParams:          # each field a float32 tensor [B] or [B,k] on the same device as x
    gamma: Tensor; wb_gain: Tensor[B,3]; moire_amp: Tensor; moire_period: Tensor; moire_angle: Tensor
    glare_amp: Tensor; glare_angle: Tensor; warp: Tensor[B,8]   # homography delta from identity (normalized coords)
    blur_sigma: Tensor; noise_std: Tensor; jpeg_quality: Tensor  # 10..95
    hed_alpha: Tensor[B,3]; hed_beta: Tensor[B,3]
def sample_camera_params(B: int, severity: float, seed: int, device="cuda") -> CameraAugParams   # severity 0 = identity params
def camera_aug(x: torch.Tensor, params: CameraAugParams, seed: int, impl: Literal["cuda","torch"] = "cuda") -> torch.Tensor
    # x uint8 [B,H,W,3] -> uint8 [B,H,W,3]. Op order fixed: warp(bilinear) -> HED jitter -> display gamma + colour profile
    # -> moire (sinusoidal screen-grid beat pattern) -> glare gradient -> defocus (separable Gaussian, radius ceil(3σ), shared-memory tiles)
    # -> white balance -> sensor noise (Philox, keyed by seed and image index) -> JPEG-like: RGB->YCbCr, 8x8 DCT-II, quantize with the
    # standard JPEG luma/chroma tables scaled by quality (IJG formula), dequantize, IDCT, ->RGB. Chroma not subsampled (documented).
# ref/camera_ops.py: one numpy function per op with identical signatures on a single HWC image + scalar params, and ref_camera_aug(img, p) composing them.
```

- **HED math:** port from `$B/stainforge-cuda-stain-kernels/src/stainforge/csrc/stainforge_kernels.cu` (read only),
  with attribution in comments.
- [ ] **Step 1:** Write the tests:
  - `test_identity_params_are_identity` (gpu): severity 0 → output equals input exactly. Allow JPEG at quality 100
    to differ by ≤ 1 level; the identity params skip JPEG.
  - `test_each_op_matches_numpy` (gpu): for each op alone (others identity), ≤ 1 level vs `ref/camera_ops.py` on 8
    tiles.
  - `test_dct_matches_scipy` (cpu): the numpy reference DCT equals `scipy.fft.dctn(norm="ortho")` to within 1e-6.
  - `test_composition_matches_numpy` (gpu): ≤ 2 levels mean absolute error, ≤ 8 levels max.
  - `test_deterministic_given_seed` (gpu).
  - `test_k2_batch_tails` (gpu): B ∈ {0,1,5}. **Review Focus 3.**
  - `test_k2_uniform_input_finite` (gpu): all-255 and all-0 tiles stay in range with no NaN. **Review Focus 2.**
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement the numpy references first, then the CUDA kernel. JPEG is a separate launch over 8×8
  blocks: one thread block per 8×8×channel block, with the DCT via precomputed cosine table in shared memory. Then
  the torch fallback, composing the same math with torch ops for the `impl="torch"` path.
- [ ] **Step 4:** Run all K2 tests under the lock. Expected: pass.
- [ ] **Step 5: Benchmark.** `bench/bench_k2.py` measures images/s at 224 px with batch 256 for:
  - (a) numpy per image, single process;
  - (b) numpy with a 4-process pool;
  - (c) torch GPU ops;
  - (d) the CUDA kernel.

  Writes `results/kernels/k2_bench.json` with an env block. Also save `figures/k2_severity_grid.png`: 6 tiles × 4
  severities, labelled "simulated camera-of-screen".
- [ ] **Step 6:** Do the sm_80 compile-check (as in Task 6), then commit.

### Task 8: OPEN pair end-to-end analysis (local)

**Files:** Create `$P/scripts/run_open.sh` and `$P/results/open/...` (generated). Modify `PROGRESS.md`.

**Interfaces:**
- Consumes: Tasks 3-5 (and Task 6's `impl="k1"` if it is done; otherwise `reference`).

- [ ] **Step 1: Embed.** For `pathgenclip`, `quiltnet_b16` and `biomedclip`, embed CRC (train subset / val / test)
  and Camelyon, `aligned` + `probe`. Use `--impl reference`, loop on exit code 3, all under the lock.
- [ ] **Step 2: Choose the partner** on **val only**: between `quiltnet_b16` and `biomedclip`, take the one whose val
  probe errors have the lowest φ-correlation with PathGen-CLIP's while its val balanced accuracy is within 3 points
  of the better candidate. Write `results/open/partner.json` with both candidates' numbers.
- [ ] **Step 3:** `run_pair_analysis("OPEN", "crc", ...)` then `("OPEN", "cam", ...)`. Prereg files are written
  before the test evaluation, by construction.
- [ ] **Step 4:** Sanity-check:
  - `readers.json` has both readers' zero-shot and probe balanced accuracy with CIs;
  - `review.json` has curves for s1-s4 and the baselines;
  - `student_targets.npz` exists.

  Append the headline numbers to PROGRESS.md.
- [ ] **Step 5:** Commit (scripts and small JSON results only; no npz/npy).

### Task 9: CM pair analysis (local subset; Delta full later)

**Files:** Create `$P/scripts/run_cm_local.sh` and `$P/results/cm_local/...` (generated).

- [ ] **Step 1:** Embed CONCH and MUSK on the same CRC subset and on Camelyon, `aligned` + `probe`. MUSK
  `probe` (`ms_aug=True`) is ~5× the compute, so measure the rate on 256 tiles first. If the projected time is over
  45 min locally, embed `probe` features for val/test plus a stratified 5k train subset and record the reduction.
  Use `--impl k1` if Task 6 passed, else `reference`.
- [ ] **Step 2:** `run_pair_analysis("CM", "crc", results_dir=$P/results/cm_local)` and `("CM", "cam", ...)`.
- [ ] **Step 3:** Record in PROGRESS.md that these are local subset numbers. The final numbers come from Delta
  (Task 17).
- [ ] **Step 4:** Commit.

### Task 10: Student model and distillation trainer

**Files:**
- Create `$P/src/twinread/student.py` and `$P/scripts/train_student.py`.
- Tests: `$P/tests/test_student.py`.

**Interfaces:**
- Consumes:
  - the `student_targets.npz` layout (Task 5);
  - `camera_aug`, `sample_camera_params` (Task 7; with `impl="torch"` fallback);
  - optionally BigBatch's fused CLIP loss: read-only import of
    `$B/bigbatch-flash-contrastive-kernel/src/bigbatch` via `sys.path`, guarded. Fall back to a torch InfoNCE with
    identical math.
- Produces:

```python
BACKBONES = ("mobilenetv4_conv_small", "efficientnet_b0", "mobileone_s0", "regnetx_002", "resnet18", "lcnet_050")
class TwinStudent(nn.Module):
    def __init__(self, backbone: str, dimA: int, dimB: int, n_classes: int, pretrained: bool = True): ...
    def forward(self, x: Tensor) -> dict[str, Tensor]
        # x float [B,3,224,224] in [0,1]; ImageNet mean/std normalization INSIDE the module (so ONNX takes [0,1] input)
        # returns {"hA": L2-norm [B,dimA], "hB": L2-norm [B,dimB], "logits": [B,C], "review_logit": [B]}
def distill_loss(out: dict, tA: Tensor, tB: Tensor, fused_logits: Tensor, y: Tensor, review_y: Tensor,
                 T: float = 2.0, w: dict | None = None) -> tuple[Tensor, dict[str, float]]
    # w defaults {"cos": 1.0, "rel": 0.5, "kd": 1.0, "ce": 0.5, "rev": 0.5};
    # cos: mean(1 - <h, t>) for A and B; rel: symmetric InfoNCE between student h and teacher t batch (temperature 0.07);
    # kd: T^2 * KL(softmax(fused_logits/T) || softmax(logits/T)); ce: cross-entropy(y); rev: BCE-with-logits(review_y)
def train_student(pair: str, dataset: str, backbone: str, epochs: int, batch: int,
                  aug: Literal["k2","torch","none"], out_dir: Path, severity_max: float = 1.0, seed: int = 0) -> dict
    # images from load_split (uint8, GPU-resident when they fit) ; per batch: sample severity ~ U(0, severity_max), K2 aug on GPU,
    # resize 224 if needed; AdamW lr 2e-3 (backbone 1e-3), cosine schedule, AMP fp16; saves best-on-val checkpoint + metrics.json:
    # val/test balanced acc (clean + K2 severities 0.5/1.0 fixed seed), hA/hB cosine to teachers, review AUROC vs teacher review label,
    # agreement of on-device flag with teacher flag, images/s with the chosen aug.
    # Also writes thresholds.json {"js": float, "review": float}: chosen on VAL student outputs so the device_review_rule (Task 14)
    # flags ≈ 10% of val tiles; Task 11 copies it into the bundle config.
```

- [ ] **Step 1:** Write the tests (cpu):
  - `test_forward_shapes_and_norms`: backbone `lcnet_050` with `pretrained=False`.
  - `test_distill_loss_zero_when_perfect`: student outputs equal the teacher targets and logits equal fused_logits
    gives cos=0 and kd=0.
  - `test_distill_loss_gradients_flow`.
  - `test_train_student_one_step_cpu(tmp_path)`: a 16-tile fixture targets file, `aug="none"`, 1 epoch;
    `metrics.json` is written.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: pass. Commit.
- [ ] **Step 5: Local OPEN run** (after Tasks 8 and 12):
  - Train the backbone chosen from the latency results (Task 12/15; until then `mobilenetv4_conv_small`) on the
    OPEN CRC targets for 15 epochs, batch 128, `aug=k2`.
  - Also an `aug=none` ablation for 5 epochs.
  - Report images/s with `k2` vs `torch` aug in `results/student/open_crc/throughput.json`.
  - Optional ablation from design §6.6, only if two backbones fit the latency budget: two separate tiny students
    (one per reader) vs the shared-backbone student. Compare how well their disagreement tracks the teacher review
    label.
  - Commit the metrics JSON.

### Task 11: ONNX int8 export and edge bundle

**Files:**
- Create `$P/src/twinread/export.py` and `$P/scripts/make_bundle.py`.
- Tests: `$P/tests/test_export.py`.

**Interfaces:**
- Consumes: `TwinStudent` checkpoints, `Probe` objects, `text_bank` outputs.
- Produces:

```python
def export_onnx(model: TwinStudent, path: Path, opset: int = 17) -> Path        # input "x" float [N,3,224,224] in [0,1]; outputs hA, hB, logits, review_logit
def quantize_int8(fp32_path: Path, calib_x: np.ndarray, out_path: Path) -> Path  # ORT quantize_static, QDQ, per_channel=True, MinMax or Entropy (pick lower val drift)
def drift_report(fp32_path, int8_path, x: np.ndarray) -> dict                   # cosine(hA/hB fp32 vs int8), label agreement, review agreement
def build_bundle(pair: str, onnx_int8: Path, banks: dict[str, np.ndarray], probes: dict, classes: list[str],
                 thresholds: dict, out_dir: Path) -> Path
    # banks: {"A": text_bank(readerA, dataset), "B": ...}; probes: the arrays from results/<pair>/<dataset>/probes_aligned.npz;
    # thresholds: the student's thresholds.json (Task 10)
    # writes model.onnx, prompt_bank.npz (bankA, bankB), probes.npz (coef/intercept/mean/scale per head), config.json
    # {pair, classes, thresholds, licence_note, distributable: pair=="OPEN"}, SHA256SUMS; CM bundles go under $P/bundles/cm/ (git-ignored)
def verify_bundle(bundle_dir: Path) -> bool
```

- [ ] **Step 1:** Write the tests (cpu):
  - `test_export_and_int8_run(tmp_path)`: an `lcnet_050` random-init student; export, quantize with 32 random
    calibration images, and check the int8 ORT session outputs have the right shapes.
  - `test_bundle_checksums(tmp_path)`: `verify_bundle` is True, and a tampered byte gives False.
  - `test_cm_bundle_marked_not_distributable(tmp_path)`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: pass.
- [ ] **Step 5:** Build the OPEN bundle from Task 10's model into `$P/bundles/open/` and write
  `results/student/open_crc/int8_drift.json`.
- [ ] **Step 6:** Commit (code, the drift JSON, and the OPEN bundle **config + SHA256SUMS only**; ONNX files stay
  out of git).

### Task 12: Backbone latency harness (runs on the UNO Q; x86 simulation flagged)

**Files:**
- Create `$P/device/bench_backbones.py` (numpy + onnxruntime only) and `$P/scripts/export_backbones.py` (torch side).
- Tests: `$P/tests/test_bench_backbones.py`.

**Interfaces:**
- Produces:
  - `scripts/export_backbones.py --out $P/device/backbones/` exports each of `BACKBONES` as a full `TwinStudent`
    (random heads, pretrained trunk) to ONNX fp32 and int8 (calibrated on 64 CRC tiles).
  - `device/bench_backbones.py --dir device/backbones --threads 1 2 4 --runs 30 [--simulate]` writes
    `results/uno_q/backbone_latency.json`:
    `{"host": platform info, "simulated": bool, "rows": [{"backbone", "precision", "threads", "p50_ms", "p90_ms", "rss_mb", "size_mb"}]}`.
- [ ] **Step 1:** Write `test_bench_runs_simulated(tmp_path)` (cpu): exporting one tiny backbone and running the bench
  with `--simulate --runs 3` writes JSON whose `rows` carry all keys and `simulated == True`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: pass.
- [ ] **Step 5:** Export all backbones. Run `--simulate` on x86 as a placeholder, written to
  `results/uno_q/backbone_latency_x86_sim.json`. Commit.

### Task 13: Screen-capture rig (labelled real camera-of-screen test set)

**Files:**
- Create `$P/rig/marker.py`, `$P/rig/show_tiles.py`, `$P/rig/record_frames.py` (runs on the UNO Q; cv2 + numpy) and
  `$P/rig/match_captures.py`.
- Tests: `$P/tests/test_rig.py`.

**Interfaces:**
- Produces:

```python
# marker.py
def encode_marker(frame_id: int, size: int = 120) -> np.ndarray   # uint8 HxWx3: black border + 4x4 grid of 12 data bits + 4 parity bits, white quiet zone
def decode_marker(img: np.ndarray) -> int | None                   # finds the marker (largest dark square contour), perspective-normalizes, reads bits, checks parity
# show_tiles.py: serves a page on 127.0.0.1:1777/rig (or a standalone fullscreen OpenCV window via --window) cycling held-out CRC test tiles
#   at a fixed on-screen size (default 672 px = 3x) with encode_marker(frame_id) beside each tile; dwell 1.5 s; writes rig_session.json {frame_id: test_index}.
# record_frames.py --device /dev/video0 --seconds N --out frames/: saves timestamped JPEG frames (camera MJPEG) on the UNO Q.
# match_captures.py frames/ rig_session.json -> data/capture/{x.npy (crops resized to 224), y.npy, idx.npy, meta.json}
#   crop = the tile region located relative to the decoded marker (tile and marker positions fixed in the page layout); frames with no decoded marker are dropped and counted.
```

- [ ] **Step 1:** Write the tests (cpu):
  - `test_marker_roundtrip_ids` over 0..4095.
  - `test_marker_decodes_after_simulated_capture`: apply `ref/camera_ops.ref_camera_aug` at severity 0.7 with a
    small warp; ≥ 95% decode rate over 200 ids, and **no wrong ids** (parity).
  - `test_match_captures_crops_tile_region`: a synthetic page render → crop equals the source tile to within 4 levels
    MAE after resize.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: pass. Commit.

### Task 14: Device application (UNO Q, offline) and MCU sketch

**Files:**
- Create in `$P/device/`: `twinread_runtime.py` (numpy + onnxruntime only; shared with tests), `uno_q_app.py`,
  `bridge.py`, `ui/index.html`, `ui/app.js`, `ui/style.css`, `mcu/twinread_led/twinread_led.ino`,
  `install_offline.sh` and `README.md`.
- Tests: `$P/tests/test_device.py`.

**Interfaces:**
- Consumes: the bundle layout (Task 11); `marker` from the rig is not needed.
- Produces:

```python
@dataclass
class TileResult:
    label: str; prob: float; probsA: list[float]; probsB: list[float]; probsF: list[float]
    review: bool; review_score: float; reasons: list[str]; no_tissue: bool
class EdgeModel:
    def __init__(self, bundle_dir: Path, threads: int = 4): ...      # verify_bundle first; refuse to load on checksum mismatch
    def infer(self, x_uint8: np.ndarray) -> list[TileResult]           # x [N,224,224,3]
def tissue_fraction(x_uint8: np.ndarray) -> float                      # fraction of pixels with OD sum > 0.15 (StainForge's background rule)
def crop_from_frame(frame: np.ndarray, mode: Literal["center","marker"] = "center") -> np.ndarray   # -> [224,224,3]; resize shorter side then center crop (same semantics as K1/PIL)
def device_review_rule(pA, pB, pF, review_logit, thr: dict) -> tuple[bool, float, list[str]]
    # review if: readers' argmax differ, OR JS(pA,pB) > thr["js"], OR sigmoid(review_logit) > thr["review"], OR no_tissue; reasons lists which fired
# bridge.py
def send_status(state: Literal["idle","busy","agree","review","no_tissue","error"]) -> bool   # Bridge RPC to MCU; returns False (no-op) if unavailable
# uno_q_app.py --bundle DIR --mode camera|replay|sim --device /dev/video0 --port 8080 --threads 4
#   loop: frame -> crop -> infer -> send_status -> JSON log line -> UI state (GET /state, GET /frame.jpg, static UI). Binds 127.0.0.1 only. No network calls.
```

- **MCU sketch:** shows 4 icons on the 8×13 matrix (idle dot, spinner, check, question mark) plus "no tissue" (a dash),
  and receives the state over Bridge.
  - Compile with `arduino-cli compile --fqbn <UNO Q FQBN>`. Discover the FQBN with
    `arduino-cli core install arduino:zephyr` + `arduino-cli board listall | grep -i "uno q"`. Record the actual FQBN
    and compile log in `results/uno_q/mcu_compile.json`.
  - Call it "compile-verified" only if the compile actually ran clean.
- [ ] **Step 1:** Write the tests (cpu; use Task 11's tiny test bundle fixture, generated in a session fixture):
  - `test_edge_model_infers_bundle`.
  - `test_refuses_tampered_bundle`.
  - `test_device_rule_matches_review_signals`: on 200 random prob triples, `device_review_rule`'s disagreement and
    JS parts equal `twinread.review.signals` results.
  - `test_blank_frame_is_no_tissue`: all-white and all-black frames → `no_tissue=True`, `review=True`. **Review
    Focus 2.**
  - `test_crop_from_frame_shapes`: 480×640, 720×1280 and 224×224 → [224,224,3], and the center pixel is preserved.
    **Review Focus 1.**
  - `test_camera_missing_falls_back_to_replay`: `--mode camera` with a nonexistent device → the app object reports
    mode `replay` with a reason. **Review Focus 5.**
  - `test_bridge_absent_noop`: `send_status` returns False without raising. **Review Focus 5.**
  - `test_app_binds_localhost_only`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. The UI shows: the live frame with the crop box; both readers' calls (bars); the fused
  call; a big AGREE / REVIEW badge with reasons; tissue fraction; latency; and a permanent "Research prototype - not
  a diagnostic device" banner. In camera mode it adds "Camera filming a screen (demo)".
- [ ] **Step 4:** Run the tests. Expected: pass.
- [ ] **Step 5:** Run the app in `--mode sim` on x86 against the OPEN bundle. Take a Playwright screenshot to
  `figures/ui_device_sim.png`, labelled simulation. Compile the MCU sketch.
- [ ] **Step 6:** Commit.

### Task 15: UNO Q bring-up and on-board measurements (needs the board plugged in by USB-C)

**Files:** Create `$P/device/bringup.md` (log of exact commands) and `$P/results/uno_q/*.json`.

- [ ] **Step 1:** `adb devices` shows the board, and `adb shell 'uname -a; cat /etc/os-release; free -m; nproc; python3 --version'`
  is recorded.
- [ ] **Step 2: Network.**
  - Ask the user for Wi-Fi credentials, or let them type them: `adb shell nmcli dev wifi connect <SSID> password <...>`.
  - Install Tailscale (`curl -fsSL https://tailscale.com/install.sh | sh`, then `tailscale up`; the user completes
    the login URL).
  - Fallback if there's no Wi-Fi: `adb reverse` plus a local HTTP proxy for pip, and push wheels.
- [ ] **Step 3:** Install the Python deps on the board (`python3 -m venv ~/twinread-venv`; `pip install onnxruntime numpy pillow opencv-python-headless`),
  and save the wheel cache (`pip download -d wheels ...`) back to `$P/device/wheels/` for the offline installer.
- [ ] **Step 4:** Push `device/` (bench + backbones), then run `python3 bench_backbones.py --threads 1 2 4 --runs 30`
  → `results/uno_q/backbone_latency.json` (`simulated: false`).
- [ ] **Step 5: Choose the student backbone.** It is the most accurate on Task 10's OPEN val among those with int8
  p50 ≤ 250 ms at 4 threads. Write `results/uno_q/backbone_choice.json` and tell the user before Delta's J3.
- [ ] **Step 6:** After Task 14, push the app + OPEN bundle and `install_offline.sh`. Upload the MCU sketch.
  Disable networking (`nmcli networking off`), run `--mode replay` and `--mode camera`, and record latency +
  screenshots + a short demo video.
- [ ] **Step 7:** Run the screen-capture rig (Task 13): `show_tiles.py` on this machine's monitor and
  `record_frames.py` on the board, then `match_captures.py` → `data/capture/`. Evaluate the OPEN student on it →
  `results/student/open_crc/real_capture.json`.
- [ ] **Step 8:** Commit the results and logs.

### Task 16: NCSA kit (written and tested locally; nothing runs on Delta here)

**Files:**
- Create in `$P/ncsa/`: `README.md` (runbook + approval protocol), `setup_login.sh`, `download_weights.py`,
  `stage_data.sh`, `j1_check.slurm`, `j2_embed.slurm`, `j3_distill.slurm`, `j4_spare.slurm`.
- Tests: `$P/tests/test_ncsa_scripts.py`.

**Interfaces:**
- Produces scripts that, once run on Delta with the user's OK:
  - **`setup_login.sh` (login node):** `module load pytorch-conda/2.8; unset PYTHONNOUSERSITE`;
    `pip install --user "open_clip_torch>=2.24" onnx onnxruntime`; CONCH/MUSK with `--no-deps` + fairscale +
    sentencepiece. Then a version printout. The `twinread.compat` shims are no-ops when transformers < 5. Make that
    a conditional in `compat.py` and test it with a monkeypatched version string.
  - **`download_weights.py` (login node):** `hf_hub_download` for CONCH `pytorch_model.bin`, MUSK
    `model.safetensors` + `tokenizer.spm`, and the open readers.
  - **`stage_data.sh`:** checks `/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples` readability. Otherwise
    prints the HF download or rsync command for the user to approve.
  - **sbatch headers:** `#SBATCH --account=bfqi-delta-gpu --partition=gpuA100x4 --gpus-per-node=1 --cpus-per-task=16 --mem=60G`
    with explicit `--time`: J1 00:30:00, J2 01:30:00, J3 01:30:00, J4 01:30:00. Env:
    `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_CUDA_ARCH_LIST=8.0`. No `--reservation`. **No self-resubmission
    anywhere.**
  - **J2** runs `python -m twinread.embed --pair CM --dataset crc --impl k1` (falling back to `reference` if J1
    recorded a failed build) on the full 100k + 7k.
  - **J3** runs `scripts/train_student.py --pair CM --dataset crc --backbone <choice> --epochs 15 --aug k2`.
  - Every job writes `results/ncsa/<job>_<SLURM_JOB_ID>.json` with timing and GPU info.
- [ ] **Step 1:** Write the tests (cpu, static):
  - `test_sbatch_headers`: each sbatch has account `bfqi-delta-gpu`, partition `gpuA100x4`, ≤ 1 GPU, cpus ≤ 16,
    mem ≤ 62G, and time ≤ 01:30:00.
  - `test_no_resubmit`: no `sbatch` call inside sbatch files.
  - `test_offline_env_set`.
  - `test_readme_has_approval_protocol`: the string "Ask the user for an explicit OK before every command".
  - `test_compat_noop_on_transformers_v4` (monkeypatch).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: pass. Commit.

### Task 17: Delta runs (user-gated; at the event)

**Files:** `$P/results/ncsa/*`, `$P/results/cm/*` (generated), `PROGRESS.md`.

- [ ] **Step 1:** Show the user each command from `ncsa/README.md` in order. Run it only after an explicit OK. Order:
  1. copy the code (rsync or `git clone` of `$P` via a bundle);
  2. `setup_login.sh`;
  3. `download_weights.py`;
  4. `stage_data.sh`;
  5. `sbatch j1_check.slurm`;
  6. read the J1 output;
  7. `sbatch j2_embed.slurm`.
- [ ] **Step 2:** Copy the embeddings back (small). Run `run_pair_analysis("CM", "crc", results_dir=$P/results/cm)`
  locally (CPU), plus D3 if the PathMMU subsets have recoverable labels (inspect with a user-approved `ls`/`head`).
- [ ] **Step 3:** J3 student distillation (CM, full data, the backbone from Task 15). Copy the checkpoint back.
  Build the CM bundle into `$P/bundles/cm/` (git-ignored, not distributable). Run on the board for the research demo.
- [ ] **Step 4:** Record GPU-hours used per job in `results/ncsa/budget.json`, and stop at 10 A100-hours total
  unless the user approves more.
- [ ] **Step 5:** Commit the results JSON.

### Task 18: Host dashboard, figures, README, manifest, showcase

**Files:**
- Create `$P/src/twinread/app_host.py` (FastAPI, 127.0.0.1:1777: results explorer, tile viewer with both readers'
  calls and the flag, the `/rig` page from Task 13) and `$P/src/twinread/figures.py`.
- Create `$P/README.md`, `$P/project.json`, `$P/run_demo.sh` and `$P/docs/deck_outline.md`.
- Tests: `$P/tests/test_manifest.py`.

- [ ] **Step 1:** Write the tests:
  - `test_project_json_schema`: all CLAUDE.md keys, `demo_port == 1777`, a `lay` with what/why/impressive of 25-55
    words each, and every `images[].path` exists.
  - `test_numbers_traceable`: every number with a decimal point in the `project.json` highlights/metrics appears in
    some `results/**/*.json`.
  - `test_readme_has_disclaimer_and_abstract`.
- [ ] **Step 2: Figures (≤ 8),** each from results JSON:
  1. two-readers headline: single readers vs fused, with CIs, CM and OPEN;
  2. triage curves: s1-s4 vs the baselines;
  3. blind-spot matrix + confident shared-error gallery;
  4. K1 throughput;
  5. K2 severity grid + K2 throughput;
  6. student accuracy vs teachers + int8 drift + UNO Q latency;
  7. device UI screenshot / photo;
  8. the system diagram.
- [ ] **Step 3:** Write the README:
  - a ~200-word abstract;
  - a "Results at a glance" table;
  - the two configurations (OPEN distributable, CM research), with the licence explanation;
  - kernels;
  - device;
  - how to run;
  - event scale-up;
  - limitations.

  Then `project.json` and `run_demo.sh` (with `--smoke`). Then `docs/deck_outline.md`: ≤ 20 slides covering aims,
  methods, results, challenges, references.
- [ ] **Step 4:** Run the full CPU test suite and the GPU suite under the lock, then `bash run_demo.sh --smoke`.
- [ ] **Step 5:** Rebuild the showcase: `$B/.venv/bin/python $B/showcase/check_tests.py twinread-conch-musk-edge && $B/.venv/bin/python $B/showcase/build.py`.
  Commit.
