#!/usr/bin/env python3
"""Runs ON the board, stdlib only: one short window with the chosen GGUF, using exactly the runtime flags and the prompt contract.

Reads a request file made on the PC, starts llama-server, prewarms, streams the answers, samples temperature every 2 s, aborts at the
limit or the deadline, and always stops its own server. Prints one JSON document. Touches nothing else on the board.
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--requests", required=True)
ap.add_argument("--max-seconds", type=float, default=170)
ap.add_argument("--abort-c", type=float, default=72.0)
ap.add_argument("--port", type=int, default=1790)
a = ap.parse_args()
PORT = a.port
spec = json.load(open(a.requests))


def hot_c():
    vals = []
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            vals.append(int(open(p).read()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(vals)


out = {"start_c": hot_c(), "requests": [], "aborted": None, "temps": []}
if out["start_c"] > 58:
    out["aborted"] = "too hot to start"
    print(json.dumps(out))
    raise SystemExit(3)
t_start = time.monotonic()
stop = threading.Event()
abort = threading.Event()


def watch():
    while not stop.is_set():
        c = hot_c()
        out["temps"].append([round(time.monotonic() - t_start, 1), c])
        if c >= a.abort_c:
            out["aborted"] = f"thermal {c} C"
            abort.set()
            return
        if time.monotonic() - t_start > a.max_seconds:
            out["aborted"] = "max_seconds"
            abort.set()
            return
        time.sleep(2)


env = dict(os.environ, OMP_WAIT_POLICY="passive")
srv = subprocess.Popen(["nice", "-n", "10", "device/llm/bin/llama-server", "-m", a.model, "-c", "1024", "-t", "3", "-tb", "3", "-np", "1", "--host", "127.0.0.1", "--port", str(PORT),
                        "--no-webui", "--cache-ram", "0", "--ctx-checkpoints", "0", "--reasoning", "off"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=True)
threading.Thread(target=watch, daemon=True).start()


def post(body, timeout=170):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


try:
    t0 = time.monotonic()
    while True:
        if abort.is_set() or srv.poll() is not None:
            raise RuntimeError("server did not come up")
        try:
            if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()).get("status") == "ok":
                break
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(0.25)
    out["load_s"] = round(time.monotonic() - t0, 2)
    t0 = time.monotonic()
    with post(spec["prewarm"]) as r:
        d = json.loads(r.read())
    out["prewarm_s"] = round(time.monotonic() - t0, 2)
    out["prewarm_prompt_n"] = d.get("usage", {}).get("prompt_tokens")
    for item in spec["requests"]:
        if abort.is_set() or a.max_seconds - (time.monotonic() - t_start) < 60:
            break
        t0 = time.monotonic()
        ttft, text, timings, finish = None, "", None, None
        with post(item["body"]) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                timings = ev.get("timings") or timings
                ch = (ev.get("choices") or [{}])[0]
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.monotonic() - t0
                text += piece
                finish = ch.get("finish_reason") or finish
                if abort.is_set():
                    break
        out["requests"].append({"id": item["id"], "text": text, "ttft_s": round(ttft, 2) if ttft else None, "total_s": round(time.monotonic() - t0, 2), "finish_reason": finish, "words": len(text.split()),
                                "gen_tps": round(timings["predicted_per_second"], 2) if timings else None, "gen_n": timings.get("predicted_n") if timings else None,
                                "prompt_n": timings.get("prompt_n") if timings else None, "cache_n": timings.get("cache_n") if timings else None,
                                "prompt_tps": round(timings["prompt_per_second"], 2) if timings else None})
except Exception as e:  # noqa: BLE001 - always report, always clean up
    out["error"] = str(e)
finally:
    stop.set()
    try:
        os.killpg(srv.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        srv.wait(10)
    except subprocess.TimeoutExpired:
        os.killpg(srv.pid, signal.SIGKILL)
        srv.wait(5)
    out["window_s"] = round(time.monotonic() - t_start, 1)
    out["peak_c"] = max(c for _, c in out["temps"]) if out["temps"] else out["start_c"]
    out["end_c"] = hot_c()
    print(json.dumps(out))
