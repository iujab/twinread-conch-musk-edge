# NCSA Delta kit (written and tested locally; nothing here has run on Delta)

Research prototype, not a diagnostic device. These scripts are for the hackathon weekend (Sat-Sun Oct 17-18).

## Approval protocol (binding)

**Ask the user for an explicit OK before every command** that touches Delta: ssh, rsync, module loads, pip installs,
downloads, `sbatch`, `scancel`. One OK covers one command, not a session.

- Claude never submits a GPU job without stating its `--time` and purpose first, and waiting for the OK.
- No script resubmits itself. No `sbatch` appears inside any `.sbatch` file. J2 loops on exit code 3 *inside* one job
  (that is not resubmission). Each job is submitted by hand, once.
- Budget: 10 GPU-h per person (team plans 30 total); target <= 10 A100-h for this project. Jobs that cross Sun 08:00 or
  18:00 fall into the regular queue, so check the clock before submitting.
- Compute nodes have no internet. All downloads happen on the login node first.

## Facts used (from `ashby-hackathon-2026/HACKATHON_COMPUTE.md`)

Login `ssh <user>@login.delta.ncsa.illinois.edu`; accounts `bfqi-delta-gpu` / `bfqi-delta-cpu`; partition `gpuA100x4`,
no `--reservation`; `module load pytorch-conda/2.8` then `unset PYTHONNOUSERSITE`; `pip install --user`; outputs under
`/u/<user>` (100 GB); shared data `/projects/bfqi/shared/datasets/`. Delta has torch 2.8 and an older transformers, so
`twinread.compat` shims are no-ops there (`patch_conch()` returns immediately when transformers < 5; tested).
The CUDA toolkit module name for building the kernels is unverified: J1 probes `module avail cuda` / `nvcc --version`
and records it in `results/ncsa/j1_toolkit_<jobid>.txt`.

## Runbook (each line needs its own OK)

Every job is submitted from `~/twinread`: each `#SBATCH --output=logs/<job>_%j.out` path is relative to the submit
directory, and Slurm does not create `logs/` (a missing `logs/` fails the job before its first line runs).
`setup_login.sh` creates `~/twinread/logs`; step 2b repeats it explicitly before the first `sbatch`.
Everything Delta produces lands in `results/delta/` (analysis, students, kernel checks) and `results/ncsa/` (job
records), never in the committed local-subset results.

1. Ship the code from the laptop (run there):
   `rsync -av --exclude data --exclude build --exclude .git --exclude .worktrees --exclude bundles --exclude results --exclude .superpowers --exclude __pycache__ --exclude .pytest_cache ./ <user>@login.delta.ncsa.illinois.edu:~/twinread/`
   (measured locally: 13 MB in 115 files, against 1.07 GB with the old excludes: no worktrees, no CM bundle, no
   local results or caches). Never ship the CM student/bundle to a shared location.
2. Login node: `bash ~/twinread/ncsa/setup_login.sh` (installs open_clip, onnx, CONCH/MUSK `--no-deps`, fairscale,
   sentencepiece, pytest; prints versions; creates `~/twinread/logs`).
2b. Login node: `mkdir -p ~/twinread/logs ~/twinread/results/ncsa` (no-op if step 2 ran; needed before any `sbatch`).
3. Login node, once: `hf auth login` (older huggingface_hub: `huggingface-cli login`; the token is stored under
   `$HF_HOME`, never typed into a command line), then `python ~/twinread/ncsa/download_weights.py` (CONCH
   `pytorch_model.bin`, MUSK `model.safetensors` + `tokenizer.spm`, open readers; it uses the stored login).
3b. Login-node offline load smoke (CPU, a few seconds; catches a missing cache before any A100 time; ask first):
   `cd ~/twinread && PYTHONPATH=src HF_HUB_OFFLINE=1 python -c "from twinread.readers import load_reader; r=load_reader('conch','cpu'); r.close()"`
   and the same with `'musk'`. HF cache is `$HF_HOME` (default `~/.cache/huggingface`, hub cache in `$HF_HOME/hub`), the same in every job.
   `download_weights.py --backbones mobilenetv4_conv_small` (the default) also caches the timm student backbone, since J3 loads it offline. It is the trainer's default and the backbone of every local student and of the latency harness rows (`results/uno_q/backbone_latency_x86_sim.json`).
4. Login node: `bash ~/twinread/ncsa/stage_data.sh` (print-only; it runs nothing). It checks `/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/<CLASS>/*.tif`
   and, if readable, tells you to submit `cd ~/twinread && mkdir -p logs && sbatch ncsa/stage_crc.sbatch` (CPU partition, `--time=01:00:00`, `crc_from_folder` for train; needs its own OK). CRC-VAL-HE-7K is not on Delta; the script prints the HF (`fetch_crc`) or
   rsync-from-laptop command for the user to approve.
5. `cd ~/twinread && mkdir -p logs && sbatch ncsa/j1_check.sbatch` (`--time=00:30:00`, expected ~10-15 min).
   Purpose: env + CUDA toolkit probe; the sm_80 K1/K2 JIT build, asserted (`load_ext` returns None on failure, so
   J1 checks the module and its `camera_aug`); the GPU correctness smoke on the A100
   (`TWINREAD_GPU_TESTS=1 pytest -m gpu tests/test_k1.py tests/test_k2.py`, including K1's embedding-cosine test,
   which loads CONCH and MUSK offline, so a missing weight cache fails here, not in J2); and a short K1 benchmark
   (`bench/bench_k1.py --part gpu --batch 64`). Outputs: `results/delta/kernels/` (build log, pytest log + JUnit XML,
   `k1_bench.json`) and `results/ncsa/j1_build_status.txt` = `ok` only if the build and the kernel tests passed.
   Read the result.
6. `sbatch ncsa/j2_embed.sbatch` (`--time=01:30:00`, CM embeddings of 100k + 7k at batch 64; uses `--impl k1`, which embeds CONCH and MUSK together from one `dual_preprocess` launch per batch, and falls back to `reference` unless J1 recorded `ok`; loops on exit 3 inside the job with `--budget` = remaining walltime minus 300 s, which `twinread.embed` checks before every 2,048-tile chunk, so the readers load once per job; it stops when budget < 300 s or two exit-3 runs add no cache parts).
6b. Analysis, CPU only (J3 needs its outputs): `sbatch ncsa/analyze_cpu.sbatch` (account `bfqi-delta-cpu`, partition `cpu`, 16 CPUs, `--time=01:00:00`, purpose: `run_pair_analysis("CM", "crc", results_dir=Path("results/delta"))` on the J2 embeddings). It writes `results/delta/CM/crc/` including `student_targets.npz` and `probes_aligned.npz` (and the preregistration, written once). It refuses to start if any `data/emb/crc/<reader>_<kind>_<split>.npy` is missing or a `*.parts` dir is left.
    This Delta job is the **only** supported full-data analysis route: it reads Delta's embeddings, labels and text
    banks together. (Analysing on the laptop would overwrite the local 20k-tile embedding caches and mix local
    labels with Delta embeddings.)
7. `sbatch ncsa/j3_distill.sbatch` (`--time=01:30:00`, `train_student.py --pair CM --dataset crc --targets-dir results/delta/CM/crc --backbone $BACKBONE --epochs 15 --aug k2 --out results/delta/student/cm_crc`; set `BACKBONE` via `sbatch --export=ALL,BACKBONE=...`, `TARGETS` likewise if the analysis outputs live elsewhere). The trainer reads `student_targets.npz` and `probes_aligned.npz` from `--targets-dir`; the embeddings in `data/emb` are not read by J3. Loss weights are the trainer's default (R26: rel = 0); nothing in `ncsa/` overrides them.
8. `ncsa/j4_spare.sbatch` (`--time=01:30:00`) only if budget remains: `sbatch --export=ALL,ACTION=<distill_seed2|kernel_build>`; any other ACTION is rejected. (There is no OPEN-pair embed on Delta: the CPU analysis job is CM-only, so the open control pair stays a local-subset result.) `distill_seed2` writes `results/delta/student/cm_crc_seed2`. J3 refuses to start if `student_targets.npz` / `probes_aligned.npz` (or the training tiles) are missing or the backbone is not cached, and uses `--aug torch` unless J1 recorded `ok`.
8b. Optional, CM int8 bundle on Delta (CPU, ~5 min, peak RAM ~13 GB: val + test as float32, calibration rows read from the memory map; private home only, never `/projects`). In the login shell after
   `module load pytorch-conda/2.8 && unset PYTHONNOUSERSITE`:
   `cd ~/twinread && srun --account=bfqi-delta-cpu --partition=cpu --cpus-per-task=4 --mem=32G --time=00:20:00 env PYTHONPATH=src OMP_NUM_THREADS=4 python scripts/make_bundle.py --pair CM --dataset crc --student results/delta/student/cm_crc --targets-dir results/delta/CM/crc --data-dir data --out bundles/cm_delta`
   It uses Delta's own tiles, analysis targets and text banks (`data/emb/crc/bank_{conch,musk}.npy`) and writes
   `results/delta/student/cm_crc/int8_drift.json`; `bundles/cm_delta` stays on Delta (git-ignored if copied).
9. Copy the results back (run on the laptop), only these two directories:
   `rsync -av <user>@login.delta.ncsa.illinois.edu:~/twinread/results/delta/ ./results/delta/`
   `rsync -av <user>@login.delta.ncsa.illinois.edu:~/twinread/results/ncsa/ ./results/ncsa/`
   (`*.npz` and `*.pt` stay git-ignored; never commit them.)

Every job writes `results/ncsa/<job>_<SLURM_JOB_ID>.json` (elapsed time, GPU, torch/CUDA/transformers versions, exit code).

## Local verification

`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_ncsa_scripts.py -q` (static: headers, no resubmit, offline env,
approval protocol text, `bash -n`, J2 -> analysis -> J3 order, the `logs/` directory, the rsync excludes and copy-back
paths, `results/delta` outputs; plus `embed_loop.sh` run against a fake `python` to check the budget, batch and reader
arguments). The scripts themselves are untested on Delta; expect first-run fixes (module names, the `BACKBONE`
default).
