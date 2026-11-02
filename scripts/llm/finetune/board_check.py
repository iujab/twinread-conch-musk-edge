#!/usr/bin/env python3
"""One short window on the UNO Q with the chosen GGUF. Run from the PC. Holds ~/.config/twinread/board.lock through scripts/llm/board.py's helpers.

Builds requests from held-out snapshots with device/explain_prompt.py, copies the two stdlib modules and the board-side script, waits for a cool
board, runs on_board_explain.py (at most 170 s, abort at 72 C, llama-server flags from device/llm/README.md), scores the answers with
explain_check, and records standby and leftover processes afterwards. twinread.service is not stopped, restarted or commanded.
Writes results/llm/finetune/board_check.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(REPO / "scripts/llm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import board  # noqa: E402
import explain_check as ec  # noqa: E402
import explain_prompt as ep  # noqa: E402
from generate import items  # noqa: E402

SCP = ["scp", "-q", "-F", str(Path.home() / ".config/twinread/ssh_config")]
RDIR = board.REMOTE


def state():
    py = ("import json,urllib.request\n"
          "d=json.load(urllib.request.urlopen('http://127.0.0.1:8080/state',timeout=5))\n"
          "print(json.dumps({'status':d.get('status'),'model_loaded':d['controls']['model_loaded'],'camera_open':d['controls']['camera_open']}))\n")
    r = subprocess.run(board.SSH + ["python3 -I -"], input=py, capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-200:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="file name in data/llm on the board")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(REPO / "results/llm/finetune/board_check.json"))
    a = ap.parse_args()
    its = items("held_out")
    pick = []
    for sit, aud in (("agree", "doctor"), ("disagree", "user"), ("review", "doctor")):
        pick.append(next(i for i in its if i["situation"] == sit and i["audience"] == aud))
    spec = {"prewarm": ep.prewarm_request(), "requests": [{"id": i["id"], "body": ep.chat_request(i["snapshot"], i["audience"], seed=a.seed + i["position"])} for i in pick]}
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "explain_requests.json"
        req.write_text(json.dumps(spec))
        board.ssh(f"mkdir -p ~/{RDIR}/scripts/llm/finetune ~/{RDIR}/device", timeout=60)
        subprocess.run(SCP + [str(Path(__file__).parent / "on_board_explain.py"), str(req), f"uno-q:{RDIR}/scripts/llm/finetune/"], check=True)
    subprocess.run(SCP + [str(REPO / "device/explain_prompt.py"), str(REPO / "device/explain_check.py"), f"uno-q:{RDIR}/device/"], check=True)
    # The contract has to import and agree on the board's own Python.
    probe = ("import sys,hashlib;sys.path.insert(0,'device');import explain_prompt as e,explain_check as c;"
             "print(sys.version.split()[0],hashlib.sha256(e.SYSTEM.encode()).hexdigest(),e.PROMPT_VERSION)")
    imp = board.ssh(f"cd ~/{RDIR} && python3 -I -c \"{probe}\"", timeout=60)
    before = state()
    curve = board.wait_cool(52.0)
    boot0 = board.boot_id()
    print("start", curve[-1], before, flush=True)
    t0 = time.time()
    r = board.locked_ssh(f"cd ~/{RDIR} && python3 -I scripts/llm/finetune/on_board_explain.py --model data/llm/{a.model} --requests scripts/llm/finetune/explain_requests.json", wait=1800, timeout=300)
    held = time.time() - t0
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - keep what the board said
        d = {"error": "no JSON from the board", "stdout": r.stdout[-500:], "stderr": r.stderr[-500:]}
    for q, item in zip(d.get("requests", []), pick):
        q.update({"situation": item["situation"], "audience": item["audience"], "user_turn": spec["requests"][pick.index(item)]["body"]["messages"][1]["content"]})
        c = ec.check_snapshot(q["text"], item["snapshot"])
        q["passed"], q["failed_checks"], q["invented_claim"] = c["passed"], [k for k, v in c["checks"].items() if not v], c["invented_claim"]
    d.update({"model": a.model, "board_python_import": imp.stdout.strip() or imp.stderr[-200:], "system_sha256_expected": __import__("hashlib").sha256(ep.SYSTEM.encode()).hexdigest(),
              "state_before": before, "lock_held_s": round(held, 1), "boot_same": board.boot_id() == boot0, "ssh_rc": r.returncode,
              "state_after": state(), "llm_processes_after": board.llm_processes(), "hot_after_c": board.hot_c(),
              "service_active": board.ssh("systemctl is-active twinread.service").stdout.strip()})
    Path(a.out).write_text(json.dumps(d, indent=1))
    print(json.dumps({k: v for k, v in d.items() if k != "temps"}, indent=1))


if __name__ == "__main__":
    main()
