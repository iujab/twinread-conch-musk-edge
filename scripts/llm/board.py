#!/usr/bin/env python3
"""PC-side helpers for the UNO Q: run from this PC, stdlib only.

  board.py status                    temperature, boots, leftover processes, models on the board
  board.py push FILE                 chunked copy into data/llm on the board, under the lock, with cool-downs and a sha256 check
  board.py rm NAME                   delete one model from the board
  board.py sync                      copy scripts/llm + example_results.json to the board
  board.py run PLAN.json [--only X]  run bench_on_board.py for each plan entry; between entries waits for the board to cool

Rules enforced: every load-bearing step holds ~/.config/twinread/board.lock only for its hot window; nothing starts above
the temperature limit; at most 1 GB of models on the board; an unexpected reboot stops the plan and saves the last samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
SSH = ["ssh", "-F", str(HOME / ".config/twinread/ssh_config"), "-o", "BatchMode=yes", "uno-q"]
LOCK = str(HOME / ".config/twinread/board.lock")
REMOTE = "twinread-conch-musk-edge"
RUNS = REPO / "results/llm/runs"
MAX_MODEL_BYTES = 1_000_000_000
TEMP_CMD = "cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1"


def ssh(cmd: str, **kw):
    return subprocess.run(SSH + [cmd], capture_output=True, text=True, **kw)


def locked_ssh(cmd: str, wait: int = 1200, **kw):
    return subprocess.run(["flock", "-w", str(wait), LOCK] + SSH + [cmd], capture_output=True, text=True, **kw)


def hot_c() -> float:
    r = ssh(TEMP_CMD, timeout=60)
    return int(r.stdout.strip()) / 1000.0


def boot_id() -> str:
    return ssh("cat /proc/sys/kernel/random/boot_id", timeout=60).stdout.strip()


def wait_cool(limit_c: float, poll_s: int = 30) -> list:
    """Wait outside the lock until the board is below limit_c. Returns the cooling curve. Polls gently."""
    curve = []
    t0 = time.monotonic()
    while True:
        h = hot_c()
        curve.append({"t": round(time.monotonic() - t0), "hot_c": h})
        if h < limit_c:
            return curve
        print(f"  cooling: {h:.1f} C, waiting for < {limit_c} C", flush=True)
        time.sleep(poll_s)


def status(_a) -> None:
    r = ssh("date -u; uptime; " + TEMP_CMD + "; journalctl --list-boots --no-pager | tail -6; "
            f"du -ch ~/{REMOTE}/data/llm/*.gguf 2>/dev/null; ls -la ~/{REMOTE}/device/llm/bin 2>/dev/null; "
            "df -h / | tail -1; free -m | sed -n 2,3p; systemctl is-active twinread-stopgap twinread.service | tr '\\n' ' '")
    print(r.stdout, r.stderr)
    print(llm_processes())


def llm_processes() -> str:
    """List LLM or benchmark processes by reading /proc, skipping this probe itself so that it cannot match its own command line."""
    probe = ("import glob,os\n"
             "me=os.getpid();rows=[]\n"
             "for p in glob.glob('/proc/[0-9]*/cmdline'):\n"
             "    pid=int(p.split('/')[2])\n"
             "    if pid==me: continue\n"
             "    try: c=open(p,'rb').read().replace(b'\\0',b' ').decode(errors='replace')\n"
             "    except OSError: continue\n"
             "    if c.startswith('python3 -I -') or c.startswith('bash -c'): continue\n"
             "    if 'llama-server' in c or 'bench_on_board' in c or 'vision_loop' in c: rows.append((pid,c[:120]))\n"
             "print('llm/bench processes on board:',rows if rows else 'none')\n")
    return subprocess.run(SSH + ["python3 -I -"], input=probe, capture_output=True, text=True, timeout=60).stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def push(a) -> None:
    src = Path(a.file)
    size = src.stat().st_size
    name = src.name
    rdir = f"{REMOTE}/data/llm"
    r = ssh(f"mkdir -p ~/{rdir}; cat ~/{rdir}/*.gguf 2>/dev/null | wc -c; test -f ~/{rdir}/{name} && stat -c %s ~/{rdir}/{name} || echo 0", timeout=120)
    held, already = (int(x) for x in r.stdout.split())
    if already == size:
        print(f"{name}: already on board")
    else:
        if held - already + size > MAX_MODEL_BYTES:
            sys.exit(f"refusing: {held} bytes on board + {size} would exceed the {MAX_MODEL_BYTES} byte limit")
        offset = 0
        ro = ssh(f"stat -c %s ~/{rdir}/{name}.part 2>/dev/null || echo 0", timeout=60)
        offset = int(ro.stdout.strip() or 0)
        chunk = a.chunk_mb * 1_000_000
        while offset < size:
            for c in wait_cool(a.max_start_c):
                pass
            n = min(chunk, size - offset)
            t0 = time.monotonic()
            with open(src, "rb") as f:
                f.seek(offset)
                data = f.read(n)
            p = subprocess.run(["flock", "-w", "1200", LOCK] + SSH + [f"cat >> ~/{rdir}/{name}.part"], input=data)
            if p.returncode:
                sys.exit(f"push failed rc={p.returncode}")
            offset += n
            dt = time.monotonic() - t0
            print(f"{name}: {offset/1e6:.0f}/{size/1e6:.0f} MB, {n/1e6/dt:.1f} MB/s, hot {hot_c():.1f} C", flush=True)
    if already != size:
        want = sha256(src)
        got = locked_ssh(f"sha256sum ~/{rdir}/{name}.part | cut -d' ' -f1").stdout.strip()
        if got != want:
            sys.exit(f"sha256 mismatch: board {got}, local {want}")
        ssh(f"mv ~/{rdir}/{name}.part ~/{rdir}/{name}")
        print(f"{name}: verified sha256 {want[:16]}…")


def rm(a) -> None:
    r = ssh(f"rm -f ~/{REMOTE}/data/llm/{a.name} ~/{REMOTE}/data/llm/{a.name}.part; du -ch ~/{REMOTE}/data/llm/*.gguf 2>/dev/null | tail -3")
    print(r.stdout, r.stderr)


def sync(_a) -> None:
    ssh(f"mkdir -p ~/{REMOTE}/scripts/llm ~/{REMOTE}/results/llm ~/{REMOTE}/data/llm/run_logs ~/{REMOTE}/device/llm/bin")
    files = [str(p) for p in (REPO / "scripts/llm").glob("*.py")]
    subprocess.run(["scp", "-q", "-F", str(HOME / ".config/twinread/ssh_config")] + files + [f"uno-q:{REMOTE}/scripts/llm/"], check=True)
    subprocess.run(["scp", "-q", "-F", str(HOME / ".config/twinread/ssh_config"), str(REPO / "results/llm/example_results.json"),
                    f"uno-q:{REMOTE}/results/llm/"], check=True)
    print("synced")


def board_models() -> dict:
    r = ssh(f"cd ~/{REMOTE}/data/llm && stat -c '%n %s' *.gguf 2>/dev/null", timeout=60)
    return {ln.rsplit(' ', 1)[0]: int(ln.rsplit(' ', 1)[1]) for ln in r.stdout.splitlines() if ln.strip()}


def find_local(name: str) -> Path:
    for d in (REPO / "data/llm", REPO / "data/llm/gen"):
        if (d / name).exists():
            return d / name
    sys.exit(f"{name} not found on this PC")


def ensure_model(name: str, upcoming: list, chunk_mb: int = 200) -> None:
    """Have this model on the board without exceeding 1 GB: delete files no later run needs, then push."""
    local = find_local(name)
    held = board_models()
    if held.get(name) == local.stat().st_size:
        return
    need = local.stat().st_size
    for other in sorted(held, key=lambda n: n in upcoming):
        if sum(held.values()) + need <= MAX_MODEL_BYTES:
            break
        ssh(f"rm -f ~/{REMOTE}/data/llm/{other}")
        print(f"  removed {other} from the board to make room", flush=True)
        del held[other]
    if sum(held.values()) + need > MAX_MODEL_BYTES:
        sys.exit(f"cannot fit {name} within {MAX_MODEL_BYTES} bytes: board holds {held}")
    push(argparse.Namespace(file=str(local), chunk_mb=chunk_mb, max_start_c=58.0))


def fetch_run(name: str):
    r = ssh(f"cat ~/{REMOTE}/data/llm/run_logs/{name}.json", timeout=120)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def one_line(d: dict) -> str:
    bits = [d["name"], f"load {d.get('load_s') or 0:.1f}s"]
    for q in d.get("requests", []):
        if q.get("gen_tps"):
            bits.append(f"[{q['prompt_n']}+{q.get('cache_n') or 0}c tok {q['prompt_tps']:.1f} pp/s, ttft {q['ttft_s'] or 0:.1f}s, "
                        f"{q['gen_n']} tok {q['gen_tps']:.2f} t/s, {q['words']}w, {q['total_s']:.0f}s]")
    bits.append(f"rss {d.get('peak_server_rss_mb')} anon {d.get('peak_server_anon_mb')}MB avail>={d.get('min_avail_mb')}MB")
    bits.append(f"peak {d.get('peak_hot_c')}C window {d.get('window_s') or 0:.0f}s")
    if d.get("aborted"):
        bits.append(f"ABORTED {d['aborted']}")
    if d.get("error"):
        bits.append(f"ERROR {d['error']}")
    return " | ".join(bits)


def run_plan(a) -> None:
    plan = json.loads(Path(a.plan).read_text())
    RUNS.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    for entry in plan["runs"]:
        name = entry["name"]
        if only and name not in only:
            continue
        out = RUNS / f"{name}.json"
        if out.exists() and not a.force:
            print(f"skip {name}: already measured", flush=True)
            continue
        args = {**plan.get("defaults", {}), **entry.get("args", {})}
        if str(args.get("model", "")).startswith("data/llm/"):
            later = [str({**plan.get("defaults", {}), **e.get("args", {})}.get("model", "")).rsplit("/", 1)[-1]
                     for e in plan["runs"] if e["name"] != name and not (RUNS / f"{e['name']}.json").exists()]
            ensure_model(args["model"].rsplit("/", 1)[-1], later)
        argv = ["--name", name]
        for k, v in args.items():
            flag = "--" + k.replace("_", "-")
            if v is True:
                argv.append(flag)
            elif v is not False and v is not None:
                argv.append(f"{flag}={v}")  # the = form, because values such as --no-mmap start with dashes
        curve = wait_cool(a.start_c)
        boot0 = boot_id()
        remote = f"cd ~/{REMOTE} && python3 -I scripts/llm/bench_on_board.py " + " ".join(f"'{x}'" for x in argv)
        print(f"== {name} (hot {curve[-1]['hot_c']:.1f} C)", flush=True)
        t0 = time.time()
        r = locked_ssh(remote, wait=1800)
        t1 = time.time()
        boot1 = boot_id()
        d = fetch_run(name) or {"name": name, "error": "no result file on the board", "ssh_rc": r.returncode, "stderr": r.stderr[-500:]}
        d["driver"] = {"cooldown_before": curve, "boot_id_before": boot0, "boot_id_after": boot1, "lock_held_s": round(t1 - t0, 1),
                       "ssh_rc": r.returncode, "stdout": r.stdout[-300:], "stderr": r.stderr[-300:]}
        out.write_text(json.dumps(d, indent=1))
        print("  " + one_line(d), flush=True)
        if boot0 != boot1:
            samples = ssh(f"tail -n 20 ~/{REMOTE}/data/llm/run_logs/{name}.samples.jsonl").stdout
            (RUNS / f"{name}.REBOOT_last_samples.jsonl").write_text(samples)
            boots = ssh("journalctl --list-boots --no-pager | tail -6").stdout
            (RUNS / f"{name}.REBOOT_boots.txt").write_text(boots)
            print(f"UNEXPECTED REBOOT during {name}; saved the last samples. Stopping the plan.", flush=True)
            sys.exit(5)
        if r.returncode == 3:
            print("  too hot at start: will not retry automatically", flush=True)
        time.sleep(a.gap)
    print("plan finished", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=status)
    p = sub.add_parser("push")
    p.add_argument("file")
    p.add_argument("--chunk-mb", type=int, default=150)
    p.add_argument("--max-start-c", type=float, default=58.0)
    p.set_defaults(fn=push)
    p = sub.add_parser("rm")
    p.add_argument("name")
    p.set_defaults(fn=rm)
    sub.add_parser("sync").set_defaults(fn=sync)
    p = sub.add_parser("run")
    p.add_argument("plan")
    p.add_argument("--only", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--start-c", type=float, default=52.0, help="start each hot window below this")
    p.add_argument("--gap", type=int, default=60, help="minimum seconds between windows, outside the lock")
    p.set_defaults(fn=run_plan)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    os.chdir(REPO)
    main()
