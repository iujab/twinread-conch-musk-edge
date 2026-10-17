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
    assert "run_embed CM crc" in j2 and "reference" in j2 and "embed_loop.sh" in j2
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
    assert "TARGETS=${TARGETS:-results/CM/crc}" in j3
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
    assert "run_pair_analysis(\"CM\", \"crc\"" in t and 'results_dir=Path("results")' in t
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


def _run_loop(tmp_path, pair, rcs):
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
    cmd = f'source {NCSA / "embed_loop.sh"}; T0=$(date +%s); WALL=5400; run_embed {pair} crc k1; echo rc=$?'
    r = subprocess.run(["bash", "-c", cmd], cwd=tmp_path, capture_output=True, text=True,
                       env={"PATH": f"{bin_}:/usr/bin:/bin"})
    calls = [l.split() for l in (tmp_path / "calls.txt").read_text().splitlines()]
    return r, calls


def test_embed_loop_caps_musk_budget(tmp_path):
    """MUSK probe (ms_aug) chunks are slow: every CM call gets --budget <= 300 (exit 3 -> rerun with fresh budget)."""
    r, calls = _run_loop(tmp_path, "CM", [3, 3, 0])
    assert "rc=0" in r.stdout and len(calls) == 3
    for c in calls:
        assert c[:4] == ["-m", "twinread.embed", "--pair", "CM"] and "--impl" in c and c[c.index("--impl") + 1] == "k1"
        assert 0 < int(c[c.index("--budget") + 1]) <= 300


def test_embed_loop_open_budget_uncapped(tmp_path):
    r, calls = _run_loop(tmp_path, "OPEN", [0])
    assert "rc=0" in r.stdout and len(calls) == 1
    assert int(calls[0][calls[0].index("--budget") + 1]) > 300  # remaining walltime minus the 300 s margin


def test_j4_allowlist_rejects_unknown(tmp_path):
    t = (NCSA / "j4_spare.sbatch").read_text()
    assert "bash -c" not in t and "eval" not in t
    for bad in ("rm_rf", "embed_open; sbatch x", ""):
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
