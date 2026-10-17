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


@pytest.mark.parametrize("p", [*SBATCH, NCSA / "setup_login.sh", NCSA / "stage_data.sh", NCSA / "stage_crc.sbatch", NCSA / "embed_loop.sh"], ids=lambda p: p.name)
def test_bash_syntax(p):
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_compat_noop_on_transformers_v4(monkeypatch):
    import twinread.compat as compat
    monkeypatch.setattr(compat, "_transformers_major", lambda: 4)
    called = []
    monkeypatch.setitem(__import__("sys").modules, "conch.open_clip_custom", None)  # import would raise if reached
    assert compat.patch_conch() is None and not called


@pytest.mark.parametrize("p", [*SBATCH, NCSA / "stage_crc.sbatch"], ids=lambda p: p.name)
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
    assert "die()" in j3 and "j1_build_status.txt" in j3 and "AUG=torch" in j3 and "data/emb/crc/" in j3
    assert "pretrained=True" in j3


def test_embed_loop_guards():
    loop = (NCSA / "embed_loop.sh").read_text()
    assert "--budget" in loop and "margin=300" in loop and "stalls" in loop and "-lt 300" in loop


def test_j4_allowlist_rejects_unknown(tmp_path):
    t = (NCSA / "j4_spare.sbatch").read_text()
    assert "bash -c" not in t and "eval" not in t
    for bad in ("rm_rf", "embed_open; sbatch x", ""):
        r = subprocess.run(["bash", str(NCSA / "j4_spare.sbatch")], env={"PATH": "/usr/bin:/bin", "ACTION": bad},
                           capture_output=True, text=True, cwd=tmp_path)
        assert r.returncode == 2 and "unknown ACTION" in r.stderr
