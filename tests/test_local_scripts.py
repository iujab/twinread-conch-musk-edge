"""scripts/run_cm_local.sh, run against a fake python and no GPU lock (nothing is embedded or analysed)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_cm_local.sh"


def _fake_py(tmp_path, rcs):
    log = tmp_path / "calls.txt"
    fake = tmp_path / "python"
    fake.write_text("#!/bin/bash\n"
                    f'n=$(wc -l < {log} 2>/dev/null || echo 0); printf "%s " "$@" | tr "\\n" " " >> {log}; echo >> {log}\n'
                    f'rcs=({" ".join(map(str, rcs))}); exit ${{rcs[$n]:-0}}\n')
    fake.chmod(0o755)
    return fake, log


def _run(tmp_path, args, rcs, **env):
    fake, log = _fake_py(tmp_path, rcs)
    r = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "PY": str(fake), "LOCK": "", **env})
    calls = log.read_text().splitlines() if log.exists() else []
    return r, calls


def test_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_embed_default_readers_conch_musk(tmp_path):
    r, calls = _run(tmp_path, ["embed", "crc"], [3, 0, 0, 0, 0])  # first call: budget used -> rerun
    assert r.returncode == 0, r.stderr
    readers = [c.split("--readers ")[1].split()[0] for c in calls]
    assert readers == ["conch", "conch", "musk", "conch", "musk"]  # aligned (conch rerun once, musk), then probe
    kinds = [c.split("--kinds ")[1].split()[0] for c in calls]
    assert kinds == ["aligned"] * 3 + ["probe"] * 2
    for c in calls:
        b = int(c.split("--budget ")[1].split()[0])
        assert b <= (300 if "--readers musk" in c else 480)
        assert "--dataset crc" in c and "--impl reference" in c


def test_embed_reader_list_parameter(tmp_path):
    r, calls = _run(tmp_path, ["embed", "cam", "musk"], [0, 0])
    assert r.returncode == 0 and len(calls) == 2 and all("--readers musk" in c for c in calls)


def test_embed_stops_on_error(tmp_path):
    r, calls = _run(tmp_path, ["embed", "crc"], [1])
    assert r.returncode == 1 and len(calls) == 1


def test_analyze_set_e_stops_at_first_failure(tmp_path):
    bad, ok = tmp_path / "bad", tmp_path / "ok"
    bad.mkdir()
    ok.mkdir()
    r, calls = _run(bad, ["analyze"], [1, 0], RESULTS=str(bad / "res"))
    assert r.returncode != 0 and len(calls) == 1  # cam failed: crc is not attempted
    r, calls = _run(ok, ["analyze"], [0, 0], RESULTS=str(ok / "res"))
    assert r.returncode == 0 and len(calls) == 2 and str(ok / "res") in calls[0]


def test_usage_error():
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "PY": "/bin/false", "LOCK": ""})
    assert r.returncode == 2 and "usage" in r.stderr
