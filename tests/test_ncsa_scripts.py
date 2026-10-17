"""Static checks of the NCSA Delta kit. Nothing here runs on Delta or submits a job."""
import re
import subprocess
from pathlib import Path

import pytest

NCSA = Path(__file__).resolve().parents[1] / "ncsa"
SBATCH = sorted(NCSA.glob("j*.sbatch"))
TIMES = {"j1_check": "00:30:00", "j2_embed": "01:30:00", "j3_distill": "01:30:00", "j4_spare": "01:30:00"}


def _directives(p):
    return dict(re.findall(r"^#SBATCH\s+--([\w-]+)=(\S+)", p.read_text(), re.M))


def _secs(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s


def test_four_sbatch_files():
    assert [p.stem for p in SBATCH] == sorted(TIMES)


@pytest.mark.parametrize("p", SBATCH, ids=lambda p: p.stem)
def test_sbatch_headers(p):
    d = _directives(p)
    assert d["account"] == "bfqi-delta-gpu"
    assert d["partition"] == "gpuA100x4"
    assert int(d["gpus-per-node"]) <= 1
    assert int(d["cpus-per-task"]) <= 16
    assert int(d["mem"].rstrip("G")) <= 62
    assert d["time"] == TIMES[p.stem] and _secs(d["time"]) <= 5400
    assert "reservation" not in p.read_text().replace("No --reservation", "")


@pytest.mark.parametrize("p", SBATCH, ids=lambda p: p.stem)
def test_no_resubmit(p):
    body = "\n".join(l for l in p.read_text().splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"\bsbatch\b|\bsrun\b.*\bsbatch\b|\bscontrol\s+requeue|--requeue", body)
    assert "--requeue" not in p.read_text()


@pytest.mark.parametrize("p", SBATCH, ids=lambda p: p.stem)
def test_offline_env_set(p):
    t = p.read_text()
    for s in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "TORCH_CUDA_ARCH_LIST=8.0"):
        assert re.search(rf"^export {re.escape(s)}", t, re.M), s


@pytest.mark.parametrize("p", SBATCH, ids=lambda p: p.stem)
def test_writes_job_record(p):
    assert "results/ncsa/" in p.read_text() and "SLURM_JOB_ID" in p.read_text()


def test_job_commands():
    j2 = (NCSA / "j2_embed.sbatch").read_text()
    loop = (NCSA / "embed_loop.sh").read_text()
    assert 'run_embed crc "$IMPL" --pair CM' in j2 and "reference" in j2 and "embed_loop.sh" in j2
    assert "-m twinread.embed" in loop and "--budget" in loop and "-eq 3" in loop
    j3 = (NCSA / "j3_distill.sbatch").read_text()
    assert "scripts/train_student.py --pair CM --dataset crc" in j3 and "--epochs 15 --aug" in j3 and "AUG=k2" in j3
    assert "load_ext(\"8.0\")" in (NCSA / "j1_check.sbatch").read_text()


def test_readme_has_approval_protocol():
    t = (NCSA / "README.md").read_text()
    assert "Ask the user for an explicit OK before every command" in t
    assert "never submits a GPU job without stating its `--time` and purpose" in t
    assert "no script resubmits itself" in t.lower()


def test_login_scripts_content():
    s = (NCSA / "setup_login.sh").read_text()
    assert "module load pytorch-conda/2.8" in s and "unset PYTHONNOUSERSITE" in s
    assert "import musk.modeling" in s and "musk_large_patch16_384" in s and "einops" in s and "safetensors" in s
    assert "pip install --user torchscale" not in s and "timm.__version__" in s
    assert "pip install --user" in s and "--no-deps" in s and "fairscale" in s and "sentencepiece" in s
    w = (NCSA / "download_weights.py").read_text()
    for s_ in ("pytorch_model.bin", "model.safetensors", "tokenizer.spm", "MahmoodLab/CONCH", "xiangjx/musk"):
        assert s_ in w
    st = (NCSA / "stage_data.sh").read_text()
    assert "/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples" in st and "crc_from_folder" in st


@pytest.mark.parametrize("p", [*SBATCH, NCSA / "setup_login.sh", NCSA / "stage_data.sh", NCSA / "stage_crc.sbatch",
                               NCSA / "embed_loop.sh", NCSA / "analyze_cpu.sbatch"], ids=lambda p: p.name)
def test_bash_syntax(p):
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_compat_noop_on_transformers_v4(monkeypatch):
    import twinread.compat as compat
    monkeypatch.setattr(compat, "_transformers_major", lambda: 4)
    called = []
    monkeypatch.setitem(__import__("sys").modules, "conch.open_clip_custom", None)  # import would raise if reached
    assert compat.patch_conch() is None and not called


@pytest.mark.parametrize("p", [*SBATCH, NCSA / "stage_crc.sbatch", NCSA / "analyze_cpu.sbatch"], ids=lambda p: p.name)
def test_hf_home_pinned(p):
    assert re.search(r"^export HF_HOME=\$\{HF_HOME:-\$HOME/\.cache/huggingface\}", p.read_text(), re.M)
    assert 'os.environ.setdefault("HF_HOME"' in (NCSA / "download_weights.py").read_text()


def test_stage_crc_cpu_header():
    p = NCSA / "stage_crc.sbatch"
    d = _directives(p)
    assert d["account"] == "bfqi-delta-cpu" and d["partition"] == "cpu"
    assert int(d["cpus-per-task"]) <= 16 and _secs(d["time"]) <= 3600
    assert "gpus" not in p.read_text().split("set -uo")[0] and "crc_from_folder" in p.read_text()
    assert "sbatch " not in "\n".join(l for l in p.read_text().splitlines() if not l.startswith("#"))


def test_stage_data_is_print_only():
    for l in (NCSA / "stage_data.sh").read_text().splitlines():
        s = l.strip()
        assert not s.startswith(("python", "rsync", "sbatch", "pip", "module")), l


def test_download_weights_has_meta_and_backbones():
    w = (NCSA / "download_weights.py").read_text()
    assert "meta.yaml" in w and "--backbones" in w and "timm.create_model" in w


def test_j3_prechecks():
    j3 = (NCSA / "j3_distill.sbatch").read_text()
    assert "die()" in j3 and "j1_build_status.txt" in j3 and "AUG=torch" in j3
    assert "pretrained=True" in j3
    # Task 8b: the student trains on the analysis outputs, not on data/emb (it never derives targets itself)
    assert "derives targets from data/emb" not in j3 and "data/student_targets" not in j3
    assert "TARGETS=${TARGETS:-results/delta/CM/crc}" in j3
    for f in ("student_targets.npz", "probes_aligned.npz"):
        assert re.search(rf'\[ -f "\$TARGETS/{f}" \] \|\| die', j3), f
    assert '--targets-dir "$TARGETS"' in j3 and "analyze_cpu.sbatch" in j3


def test_train_student_flag_exists():
    """The flag J3/J4 pass is the trainer's real flag."""
    import sys
    sys.path.insert(0, str(NCSA.parent / "scripts"))
    import train_student
    a = train_student.parse_args(["--pair", "CM", "--dataset", "crc", "--targets-dir", "results/CM/crc"])
    assert str(a.targets_dir) == "results/CM/crc"


def test_j4_distill_uses_analysis_targets():
    j4 = (NCSA / "j4_spare.sbatch").read_text()
    assert '--targets-dir "$TARGETS"' in j4 and "student_targets.npz" in j4 and "probes_aligned.npz" in j4


def test_analyze_cpu_job():
    p = NCSA / "analyze_cpu.sbatch"
    t = p.read_text()
    d = _directives(p)
    assert d["account"] == "bfqi-delta-cpu" and d["partition"] == "cpu"
    assert int(d["cpus-per-task"]) <= 16 and _secs(d["time"]) <= 3600
    assert "gpus" not in t.split("set -uo")[0] and "reservation" not in t.replace("No --reservation", "")
    body = "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"\bsbatch\b|--requeue|scontrol\s+requeue", body)
    for s in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        assert re.search(rf"^export {re.escape(s)}", t, re.M), s
    assert "run_pair_analysis(\"CM\", \"crc\"" in t and 'results_dir=Path("results/delta")' in t
    assert "data/emb/crc/" in t and "die()" in t and "*.parts" in t
    assert "student_targets.npz" in t and "probes_aligned.npz" in t
    assert "results/ncsa/" in t and "SLURM_JOB_ID" in t and "job_record.py analyze_cpu" in t


def test_runbook_order_j2_analysis_j3():
    t = (NCSA / "README.md").read_text()
    i2, ia, i3 = t.index("sbatch ncsa/j2_embed.sbatch"), t.index("sbatch ncsa/analyze_cpu.sbatch"), \
        t.index("sbatch ncsa/j3_distill.sbatch")
    assert i2 < ia < i3
    assert "--targets-dir" in t and "student_targets.npz" in t and "probes_aligned.npz" in t
    assert "derives targets from data/emb" not in t


def test_no_old_student_loss_weights():
    """R26: the student default is rel=0; nothing in ncsa/ may pass loss weights (the old default had rel 0.5)."""
    for p in NCSA.iterdir():
        if p.is_file():
            t = p.read_text()
            assert "--loss-weights" not in t and '"rel"' not in t and "rel=0.5" not in t, p.name


def test_embed_loop_guards():
    loop = (NCSA / "embed_loop.sh").read_text()
    assert "--budget" in loop and "margin=300" in loop and "stalls" in loop and "-lt 300" in loop


def _run_loop(tmp_path, selector, rcs, **env):
    """Source embed_loop.sh with a fake ``python`` that logs its args, adds a cache part and exits with rcs[i]."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (tmp_path / "data" / "emb").mkdir(parents=True)
    fake = bin_ / "python"
    fake.write_text("#!/bin/bash\n"
                    'n=$(wc -l < calls.txt 2>/dev/null || echo 0); echo "$@" >> calls.txt\n'
                    'touch data/emb/part_$n.npy\n'
                    f'rcs=({" ".join(map(str, rcs))}); exit ${{rcs[$n]}}\n')
    fake.chmod(0o755)
    cmd = f'source {NCSA / "embed_loop.sh"}; T0=$(date +%s); WALL=5400; run_embed crc k1 {selector}; echo rc=$?'
    r = subprocess.run(["bash", "-c", cmd], cwd=tmp_path, capture_output=True, text=True,
                       env={"PATH": f"{bin_}:/usr/bin:/bin", **env})
    calls = [l.split() for l in (tmp_path / "calls.txt").read_text().splitlines()]
    return r, calls


def _arg(c, flag):
    return c[c.index(flag) + 1]


def test_embed_loop_cm_uses_whole_walltime_and_batch64(tmp_path):
    """M1: on Delta a CM call gets the remaining walltime minus the 300 s margin (twinread.embed checks it per chunk),
    not a 300 s per-call cap that reloads CONCH and MUSK every 5 min; encode batch 64."""
    r, calls = _run_loop(tmp_path, "--pair CM", [3, 0])
    assert "rc=0" in r.stdout and len(calls) == 2
    for c in calls:
        assert c[:4] == ["-m", "twinread.embed", "--pair", "CM"] and _arg(c, "--impl") == "k1"
        assert _arg(c, "--dataset") == "crc" and _arg(c, "--batch") == "64"
        assert 4800 < int(_arg(c, "--budget")) <= 5100
    loop = (NCSA / "embed_loop.sh").read_text()
    assert "MUSK_CALL_BUDGET" not in loop and "EMBED_BATCH:-64" in loop


def test_embed_loop_optional_cap_and_batch(tmp_path):
    r, calls = _run_loop(tmp_path, "--pair CM", [3, 3, 0], EMBED_CALL_CAP="300", EMBED_BATCH="32")
    assert "rc=0" in r.stdout and len(calls) == 3
    assert all(0 < int(_arg(c, "--budget")) <= 300 and _arg(c, "--batch") == "32" for c in calls)


def test_j4_has_no_unanalysed_embed():
    """R37: analyze_cpu is CM-only (R34), so J4 offers no OPEN embed that nothing would analyse."""
    j4 = (NCSA / "j4_spare.sbatch").read_text()
    assert "embed_open" not in j4 and "run_embed" not in j4
    assert 'run_pair_analysis("CM", "crc"' in (NCSA / "analyze_cpu.sbatch").read_text()
    assert "embed_open" not in (NCSA / "README.md").read_text()


def test_runbook_bundle_step_memory():
    """8b: make_bundle peaks at ~13 GB on full CRC (val + test float32); the srun asks for more than that."""
    import re
    m = re.search(r"8b\..*?--mem=(\d+)G", (NCSA / "README.md").read_text(), re.S)
    assert m and int(m.group(1)) >= 24


def test_j4_allowlist_rejects_unknown(tmp_path):
    t = (NCSA / "j4_spare.sbatch").read_text()
    assert "bash -c" not in t and "eval" not in t
    for bad in ("rm_rf", "embed_open", "distill_seed2; sbatch x", ""):
        r = subprocess.run(["bash", str(NCSA / "j4_spare.sbatch")], env={"PATH": "/usr/bin:/bin", "ACTION": bad},
                           capture_output=True, text=True, cwd=tmp_path)
        assert r.returncode == 2 and "unknown ACTION" in r.stderr


def _trainer_default_backbone():
    import sys
    sys.path.insert(0, str(NCSA.parent / "scripts"))
    import train_student
    return train_student.parse_args(["--pair", "CM", "--dataset", "crc"]).backbone


def _download_default_backbones():
    import ast
    tree = ast.parse((NCSA / "download_weights.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--backbones"):
            return next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "default")
    raise AssertionError("--backbones argument not found")


def test_backbone_defaults_match_trainer():
    """Delta must train the architecture the local runs and the latency harness used (the trainer's default)."""
    want = _trainer_default_backbone()
    assert want == "mobilenetv4_conv_small"
    j3 = (NCSA / "j3_distill.sbatch").read_text()
    j4 = (NCSA / "j4_spare.sbatch").read_text()
    assert re.search(r"^BACKBONE=\$\{BACKBONE:-(\S+)\}", j3, re.M).group(1) == want
    assert re.search(r"^BACKBONE=\$\{BACKBONE:-(\S+)\}", j4, re.M).group(1) == want
    for t in (j3, j4):
        assert '--backbone "$BACKBONE"' in t
        assert not re.search(r"--backbone (?!\"\$BACKBONE\")", t)  # no hard-coded backbone anywhere
    assert _download_default_backbones() == [want]
    readme = (NCSA / "README.md").read_text()
    assert f"--backbones {want}" in readme and "mobilenetv3" not in readme
    for p in NCSA.iterdir():
        if p.is_file():
            assert "mobilenetv3" not in p.read_text(), p.name


# ---- final review fixes (C1, I1, I2, I3, M1, M2, HF login) -------------------------------------------------------
ALL_SBATCH = sorted(NCSA.glob("*.sbatch"))


def _runbook():
    t = (NCSA / "README.md").read_text()
    return t[t.index("## Runbook"):t.index("## Local verification")]


def test_logs_dir_created_before_first_sbatch():
    """C1: Slurm opens --output=logs/... before the script runs and does not create logs/; setup and the runbook must."""
    out_dirs = set()
    for p in ALL_SBATCH:
        out = _directives(p)["output"]
        assert not out.startswith("/"), p.name
        out_dirs.add(out.rsplit("/", 1)[0])
    assert out_dirs == {"logs"}
    setup = (NCSA / "setup_login.sh").read_text()
    assert "TWINREAD_HOME=${TWINREAD_HOME:-$HOME/twinread}" in setup  # the dir the runbook rsyncs to (~/twinread/)
    assert re.search(r'^mkdir -p "\$TWINREAD_HOME/logs"', setup, re.M)
    rb = _runbook()
    assert ":~/twinread/`" in rb
    first_sbatch = rb.index("sbatch ")
    assert "mkdir -p ~/twinread/logs" in rb[:first_sbatch]
    for job in ("stage_crc", "j1_check"):  # the first two submissions also create it inline
        assert f"cd ~/twinread && mkdir -p logs && sbatch ncsa/{job}.sbatch" in rb, job
    assert "mkdir -p logs && sbatch ncsa/stage_crc.sbatch" in (NCSA / "stage_data.sh").read_text()


def test_setup_creates_logs_dir(tmp_path):
    """Run just setup_login.sh's directory lines (no module/pip) against a fake HOME."""
    lines = [l for l in (NCSA / "setup_login.sh").read_text().splitlines()
             if l.startswith(("TWINREAD_HOME=", "mkdir -p"))]
    subprocess.run(["bash", "-c", "\n".join(lines)], env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, check=True)
    assert (tmp_path / "twinread" / "logs").is_dir()


def test_j1_asserts_build_and_runs_kernel_smoke():
    """I1: load_ext returns None on failure, so J1 asserts the module (with camera_aug) and records build=failed + errors;
    then the GPU K1/K2 tests and a short K1 bench write under results/delta/kernels/."""
    j1 = (NCSA / "j1_check.sbatch").read_text()
    assert 'm = k.load_ext("8.0"); assert m is not None and hasattr(m, "camera_aug"), k.EXT_ERRORS' in j1
    assert "BUILD=failed" in j1 and "errors=" in j1 and 'echo "$STATUS" > results/ncsa/j1_build_status.txt' in j1
    assert "TWINREAD_GPU_TESTS=1" in j1 and "-m pytest -m gpu tests/test_k1.py tests/test_k2.py" in j1
    assert "TWINREAD_REQUIRE_WEIGHTS=1" in j1
    assert "bench/bench_k1.py --part gpu --batch 64 --out" in j1 and "KD=results/delta/kernels" in j1
    assert 'STATUS=ok; else STATUS=failed' in j1 and '[ "$TESTS" = passed ]' in j1
    j4 = (NCSA / "j4_spare.sbatch").read_text()
    assert 'assert m is not None and hasattr(m, "camera_aug")' in j4
    src =(NCSA.parent / "bench" / "bench_k1.py").read_text()
    assert '"--out"' in src and '"--batch"' in src


def test_j1_build_status_logic(tmp_path):
    """The status/record lines of J1, run with a fake python: build ok + tests failing -> status failed, rc != 0."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python").write_text('#!/bin/bash\ncase "$*" in *pytest*) exit 1;; esac\nexit 0\n')
    (bin_ / "python").chmod(0o755)
    for name in ("nvidia-smi", "module"):
        (bin_ / name).write_text("#!/bin/bash\nexit 0\n")
        (bin_ / name).chmod(0o755)
    body = (NCSA / "j1_check.sbatch").read_text().replace("module load pytorch-conda/2.8", "true")
    (tmp_path / "j1.sh").write_text(body)
    r = subprocess.run(["bash", str(tmp_path / "j1.sh")], cwd=tmp_path, capture_output=True, text=True,
                       env={"PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(tmp_path), "TWINREAD_HOME": str(tmp_path),
                            "SLURM_JOB_ID": "42"})
    assert r.returncode != 0
    assert (tmp_path / "results" / "ncsa" / "j1_build_status.txt").read_text().strip() == "failed"
    (bin_ / "python").write_text("#!/bin/bash\nexit 0\n")
    r = subprocess.run(["bash", str(tmp_path / "j1.sh")], cwd=tmp_path, capture_output=True, text=True,
                       env={"PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(tmp_path), "TWINREAD_HOME": str(tmp_path),
                            "SLURM_JOB_ID": "43"})
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "results" / "ncsa" / "j1_build_status.txt").read_text().strip() == "ok"


def test_delta_outputs_never_overwrite_committed_results():
    """I2/I3: J3, J4 and the analysis job write under results/delta/, never to the committed results/{CM,student}/."""
    j3 = (NCSA / "j3_distill.sbatch").read_text()
    j4 = (NCSA / "j4_spare.sbatch").read_text()
    an = (NCSA / "analyze_cpu.sbatch").read_text()
    assert "OUT=results/delta/student/cm_crc" in j3 and '--out "$OUT"' in j3
    assert "--seed 2 --out results/delta/student/cm_crc_seed2" in j4
    assert "OUT=results/delta/CM/crc" in an
    for name, t in (("j3", j3), ("j4", j4), ("analyze", an)):
        code = "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("#"))
        assert not re.search(r"results/(CM|student)/", code), name
        assert 'results_dir=Path("results")' not in code, name
    for p in ALL_SBATCH:
        assert "train_student.py" not in p.read_text() or "--out" in p.read_text(), p.name


def test_runbook_rsync_and_copy_back():
    """I2: ship only code; copy back only results/delta/ and results/ncsa/."""
    rb = _runbook()
    ship = next(l for l in rb.splitlines() if "rsync -av --exclude" in l)
    for ex in (".worktrees", "bundles", "results", "build", "data", ".git"):
        assert f"--exclude {ex} " in ship, ex
    assert "results/CM" not in ship
    back = [l.strip(" `") for l in rb.splitlines() if "rsync -av <user>@" in l]
    assert back == ["rsync -av <user>@login.delta.ncsa.illinois.edu:~/twinread/results/delta/ ./results/delta/",
                    "rsync -av <user>@login.delta.ncsa.illinois.edu:~/twinread/results/ncsa/ ./results/ncsa/"]
    assert "~/twinread/results/ ./results/" not in rb


def test_analysis_only_on_delta():
    """I3/R34: no 'analyse locally' alternative; Delta analysis writes results/delta/CM/<ds>/."""
    t = (NCSA / "README.md").read_text()
    assert "Alternative without a Delta CPU job" not in t and "run the same call locally" not in t
    assert "only** supported full-data analysis route" in t
    assert 'results_dir=Path("results/delta")' in t and "results/delta/CM/crc" in t
    an = (NCSA / "analyze_cpu.sbatch").read_text()
    assert "run the same call locally" not in an and "ONLY supported full-data analysis route" in an


def test_make_bundle_delta_flags(tmp_path):
    """I3: make_bundle gets --data-dir and --targets-dir (alias of --targets); defaults unchanged; banks from data-dir."""
    import sys
    import numpy as np
    sys.path.insert(0, str(NCSA.parent / "scripts"))
    import make_bundle
    d = make_bundle.parse_args([])
    assert d.data_dir is None and d.targets is None and d.out is None and d.pair == "OPEN"
    a = make_bundle.parse_args(["--pair", "CM", "--targets-dir", "results/delta/CM/crc", "--data-dir", str(tmp_path)])
    assert a.targets == "results/delta/CM/crc" and a.data_dir == str(tmp_path)
    assert make_bundle.parse_args(["--targets", "x"]).targets == "x"
    (tmp_path / "emb" / "crc").mkdir(parents=True)
    for i, r in enumerate(("conch", "musk")):
        np.save(tmp_path / "emb" / "crc" / f"bank_{r}.npy", np.full((2, 3), i, np.float32))
    b = make_bundle._banks(tmp_path, "crc", ("conch", "musk"))
    assert b["A"].max() == 0 and b["B"].min() == 1
    rb = _runbook()
    assert ("make_bundle.py --pair CM --dataset crc --student results/delta/student/cm_crc --targets-dir "
            "results/delta/CM/crc --data-dir data --out bundles/cm_delta") in rb


def test_hf_login_not_token_in_command():
    """Use the stored `hf auth login`, never HF_TOKEN=... on a command line."""
    for p in (NCSA / "README.md", NCSA / "download_weights.py", NCSA.parent / "README.md"):
        assert "HF_TOKEN=" not in p.read_text(), p.name
    assert "hf auth login" in (NCSA / "README.md").read_text()
    assert "HF_TOKEN" not in (NCSA / "download_weights.py").read_text()
