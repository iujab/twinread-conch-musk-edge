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

1. Ship the code from the laptop (run there):
   `rsync -av --exclude data --exclude build --exclude .git --exclude 'results/CM' ./ <user>@login.delta.ncsa.illinois.edu:~/twinread/`
   (or `git bundle create tw.bundle lane-d` and `scp` it). Never ship the CM student/bundle to a shared location.
2. Login node: `bash ~/twinread/ncsa/setup_login.sh` (installs open_clip, onnx, CONCH/MUSK `--no-deps`, fairscale, sentencepiece; prints versions).
3. Login node: `HF_TOKEN=... python ~/twinread/ncsa/download_weights.py` (CONCH `pytorch_model.bin`, MUSK `model.safetensors` + `tokenizer.spm`, open readers).
3b. Login-node offline load smoke (CPU, a few seconds; catches a missing cache before any A100 time; ask first):
   `cd ~/twinread && PYTHONPATH=src HF_HUB_OFFLINE=1 python -c "from twinread.readers import load_reader; r=load_reader('conch','cpu'); r.close()"`
   and the same with `'musk'`. HF cache is `$HF_HOME` (default `~/.cache/huggingface`, hub cache in `$HF_HOME/hub`), the same in every job.
   `download_weights.py --backbones mobilenetv3_large_100` also caches the timm student backbone, since J3 loads it offline.
4. Login node: `bash ~/twinread/ncsa/stage_data.sh` (print-only; it runs nothing). It checks `/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/<CLASS>/*.tif`
   and, if readable, tells you to submit `sbatch ncsa/stage_crc.sbatch` (CPU partition, `--time=01:00:00`, `crc_from_folder` for train; needs its own OK). CRC-VAL-HE-7K is not on Delta; the script prints the HF (`fetch_crc`) or
   rsync-from-laptop command for the user to approve.
5. `cd ~/twinread && sbatch ncsa/j1_check.sbatch` (`--time=00:30:00`, purpose: env + CUDA probe + sm_80 kernel build). Read the result.
6. `sbatch ncsa/j2_embed.sbatch` (`--time=01:30:00`, CM embeddings of 100k + 7k; uses `--impl k1`, falls back to `reference` if J1 recorded a failed build; loops on exit 3 inside the job with `--budget` = remaining walltime minus 300 s, and stops when budget < 300 s or two exit-3 runs add no cache parts).
7. `sbatch ncsa/j3_distill.sbatch` (`--time=01:30:00`, `train_student.py --pair CM --dataset crc --backbone $BACKBONE --epochs 15 --aug k2`; set `BACKBONE` via `sbatch --export=ALL,BACKBONE=...`).
8. `ncsa/j4_spare.sbatch` (`--time=01:30:00`) only if budget remains: `sbatch --export=ALL,ACTION=<embed_open|distill_seed2|kernel_build>`; any other ACTION is rejected. J3 refuses to start if J2 outputs are missing or the backbone is not cached, and uses `--aug torch` if J1 recorded a failed build.
9. Copy `results/ncsa/*.json` back (run on the laptop): `rsync -av <user>@login.delta.ncsa.illinois.edu:~/twinread/results/ ./results/`.

Every job writes `results/ncsa/<job>_<SLURM_JOB_ID>.json` (elapsed time, GPU, torch/CUDA/transformers versions, exit code).

## Local verification

`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_ncsa_scripts.py -q` (static: headers, no resubmit, offline env,
approval protocol text, `bash -n`). The scripts themselves are untested on Delta; expect first-run fixes (module names,
`twinread.embed --pair OPEN` spelling for J4, the `BACKBONE` default).
