#!/usr/bin/env python3
"""Benchmark llama-server on the UNO Q itself. Standard library only; runs on the board and drives 127.0.0.1.

For one model + thread count this records: load time, prompt and generation rate, time to first token, wall time to a
100-word answer, RSS, memory left, temperature over time, idle CPU of the loaded server, prompt-cache savings across
requests with a shared system prompt, and optionally the same with the TwinRead vision model loaded (idle or reading).

Safety, enforced here and not by the caller:
  * refuses to start when the hottest thermal zone is above --temp-start-max (default 60 C)
  * kills the model when the hottest zone exceeds --temp-abort (default 72 C) and records that it did
  * kills the model after --max-seconds, so a hot window cannot overrun
  * children die with this process (PR_SET_PDEATHSIG), and every exit path reaps them
Samples go to a fsynced log under data/llm/run_logs, which survives a power loss and shows what the load was at that moment.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import http.client
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PR_SET_PDEATHSIG = 1
TICK = os.sysconf("SC_CLK_TCK")
HOST, PORT = "127.0.0.1", 1790


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def zones():
    out = {}
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        t = read(z + "/temp")
        if t and t.strip().lstrip("-").isdigit():
            out[(read(z + "/type") or z).strip()] = int(t) / 1000.0
    return out


def hottest():
    return max(zones().values())


def meminfo():
    m = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        m[k] = int(v.split()[0]) if v.split() else 0
    return m


def proc_status(pid):
    d = {}
    for line in (read(f"/proc/{pid}/status") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSwap", "Threads"):
            d[k] = int(v.split()[0])
    return d


def proc_ticks(pid):
    s = read(f"/proc/{pid}/stat")
    if not s:
        return None
    rest = s[s.rfind(")") + 2:].split()
    return int(rest[11]) + int(rest[12])  # utime + stime


def cpu_total():
    f = (read("/proc/stat") or "cpu 0 0 0 0 0 0 0 0").splitlines()[0].split()[1:]
    v = [int(x) for x in f]
    return sum(v), v[3] + v[4]  # total, idle + iowait


def cpu_freqs_mhz():
    return [int(read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", "0") or 0) // 1000 for i in range(4)]


def all_procs():
    out = {}
    for p in glob.glob("/proc/[0-9]*"):
        pid = int(p.rsplit("/", 1)[1])
        s = read(p + "/stat")
        if not s:
            continue
        comm = s[s.find("(") + 1:s.rfind(")")]
        rest = s[s.rfind(")") + 2:].split()
        out[pid] = (comm, int(rest[11]) + int(rest[12]))
    return out


def baseline(seconds=5.0):
    """What is already running: per-process CPU and RSS, including the always-on TwinRead app and the kiosk browser."""
    a, ta = all_procs(), time.monotonic()
    c0 = cpu_total()
    time.sleep(seconds)
    b, tb = all_procs(), time.monotonic()
    c1 = cpu_total()
    rows = []
    for pid, (comm, ticks) in b.items():
        d = ticks - a.get(pid, (comm, ticks))[1]
        st = proc_status(pid)
        cmd = (read(f"/proc/{pid}/cmdline") or "").replace("\0", " ")
        rows.append({"pid": pid, "comm": comm, "cpu_pct": round(100.0 * d / TICK / (tb - ta), 1),
                     "rss_mb": round(st.get("VmRSS", 0) / 1024, 1), "cmd": cmd[:90], "is_app": "uno_q_app" in cmd})
    total, idle = c1[0] - c0[0], c1[1] - c0[1]
    m = meminfo()
    twinread = [r for r in rows if r["is_app"]]
    return {"seconds": seconds, "system_busy_pct_of_4_cores": round(100.0 * (total - idle) / max(1, total), 1),
            "mem_available_mb": round(m["MemAvailable"] / 1024), "mem_total_mb": round(m["MemTotal"] / 1024),
            "swap_used_mb": round((m["SwapTotal"] - m["SwapFree"]) / 1024),
            "loadavg": (read("/proc/loadavg") or "").split()[:3], "hottest_c": hottest(),
            "top_cpu": sorted(rows, key=lambda r: -r["cpu_pct"])[:5], "top_rss": sorted(rows, key=lambda r: -r["rss_mb"])[:6],
            "twinread_app": twinread, "boot_id": (read("/proc/sys/kernel/random/boot_id") or "").strip()}


class Sampler(threading.Thread):
    """1 Hz log of temperature, frequency, memory and CPU. Also the watchdog: it trips on heat or on the time limit."""

    def __init__(self, log_path, temp_abort, max_seconds, on_trip):
        super().__init__(daemon=True)
        self.log = open(log_path, "a", buffering=1)
        self.temp_abort, self.max_seconds, self.on_trip = temp_abort, max_seconds, on_trip
        self.t0 = time.monotonic()
        self.samples = []
        self.pids = {}  # name -> pid
        self.stop_flag = threading.Event()
        self.tripped = None
        self.phase = "start"

    def run(self):
        last = cpu_total()
        n = 0
        while not self.stop_flag.is_set():
            z = zones()
            hot = max(z.values())
            now = cpu_total()
            dt, di = now[0] - last[0], now[1] - last[1]
            last = now
            m = meminfo()
            rec = {"t": round(time.monotonic() - self.t0, 1), "mono": time.monotonic(), "wall": time.time(), "phase": self.phase,
                   "hot_c": hot, "cpuss_c": max(v for k, v in z.items() if "cpuss" in k),
                   "busy_pct": round(100.0 * (dt - di) / max(1, dt), 1), "mhz": cpu_freqs_mhz(),
                   "avail_mb": round(m["MemAvailable"] / 1024), "swap_used_mb": round((m["SwapTotal"] - m["SwapFree"]) / 1024)}
            for name, pid in list(self.pids.items()):
                st = proc_status(pid)
                rec[name + "_rss_mb"] = round(st.get("VmRSS", 0) / 1024, 1)
                rec[name + "_anon_mb"] = round(st.get("RssAnon", 0) / 1024, 1)
            self.samples.append(rec)
            self.log.write(json.dumps(rec) + "\n")
            n += 1
            if n % 5 == 0:
                self.log.flush()
                os.fsync(self.log.fileno())
            if self.tripped is None:
                if hot > self.temp_abort:
                    self.tripped = {"reason": "thermal", "hot_c": hot, "t": rec["t"], "phase": self.phase}
                elif rec["t"] > self.max_seconds:
                    self.tripped = {"reason": "max_seconds", "hot_c": hot, "t": rec["t"], "phase": self.phase}
                if self.tripped:
                    self.on_trip()
            self.stop_flag.wait(1.0)
        self.log.flush()
        os.fsync(self.log.fileno())


def child_setup():
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    os.setsid()


def http_json(method, path, body=None, timeout=30):
    c = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    data = json.dumps(body) if body is not None else None
    c.request(method, path, data, {"Content-Type": "application/json"} if data else {})
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode(errors="replace"))


def explain(system, user, a, sampler, label):
    """One streamed explanation. Returns wall-clock timings measured here plus the server's own numbers."""
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": a.max_tokens, "temperature": a.temperature,
            "top_p": a.top_p, "seed": a.seed, "cache_prompt": True, "repeat_penalty": a.repeat_penalty}
    if a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.monotonic()
    temp0 = hottest()
    sampler.phase = f"{label}:prompt"
    c = http.client.HTTPConnection(HOST, PORT, timeout=400)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"label": label, "error": f"http {r.status}", "body": r.read()[:400].decode(errors="replace")}
    text, reasoning = "", ""
    ttft = t100 = t_first_any = None
    timings = usage = finish = None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        ev = json.loads(payload)
        timings = ev.get("timings") or timings
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            finish = ch.get("finish_reason") or finish
            d = ch.get("delta") or {}
            if d.get("reasoning_content"):
                reasoning += d["reasoning_content"]
                t_first_any = t_first_any or time.monotonic()
            if d.get("content"):
                if ttft is None:
                    ttft = time.monotonic() - t0
                    sampler.phase = f"{label}:gen"
                text += d["content"]
                if t100 is None and len(text.split()) >= 100:
                    t100 = time.monotonic() - t0
    c.close()
    t1 = time.monotonic()
    words = len(text.split())
    temp1 = hottest()
    out = {"label": label, "t_start": t0, "t_end": t1, "ttft_s": ttft, "total_s": t1 - t0, "t_100_words_s": t100,
           "words": words, "finish_reason": finish, "text": text, "reasoning_chars": len(reasoning),
           "think_tag_in_text": "<think>" in text or "</think>" in text, "usage": usage, "timings": timings,
           "temp_start_c": temp0, "temp_end_c": temp1}
    if timings:
        out.update({"prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"), "prompt_ms": timings.get("prompt_ms"),
                    "prompt_tps": timings.get("prompt_per_second"), "gen_n": timings.get("predicted_n"),
                    "gen_ms": timings.get("predicted_ms"), "gen_tps": timings.get("predicted_per_second")})
        if words and timings.get("predicted_n"):
            out["tokens_per_word"] = timings["predicted_n"] / words
            # Same model and rate, normalised to exactly 100 words: lets runs with different answer lengths be compared.
            out["est_100_words_s"] = (timings.get("prompt_ms", 0) / 1000.0
                                      + 100 * out["tokens_per_word"] / max(timings.get("predicted_per_second") or 1e-9, 1e-9))
    sampler.phase = "between"
    return out


def percentile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def read_stats(rows):
    ms = [r["ms"] for r in rows]
    if not ms:
        return {"n": 0}
    return {"n": len(ms), "p50_ms": round(percentile(ms, 0.5), 1), "p95_ms": round(percentile(ms, 0.95), 1),
            "max_ms": round(max(ms), 1), "mean_ms": round(sum(ms) / len(ms), 1),
            "late_over_100ms": sum(1 for r in rows if r.get("late_ms", 0) > 100),
            "over_500ms": sum(1 for m in ms if m > 500)}


def cpu_pct_over(pid, seconds):
    t0, a = time.monotonic(), proc_ticks(pid)
    c0 = cpu_total()
    time.sleep(seconds)
    b, t1 = proc_ticks(pid), time.monotonic()
    c1 = cpu_total()
    total, idle = c1[0] - c0[0], c1[1] - c0[1]
    return {"seconds": seconds, "server_cpu_pct_of_one_core": round(100.0 * (b - a) / TICK / (t1 - t0), 2),
            "system_busy_pct_of_4_cores": round(100.0 * (total - idle) / max(1, total), 1), "hot_c": hottest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", default=str(REPO / "device/llm/bin/llama-server"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--examples-file", default=str(REPO / "results/llm/example_results.json"))
    ap.add_argument("--requests", default="0,1", help="example indices, in order; each is one explanation")
    ap.add_argument("--repeat-until-s", type=float, default=0.0, help="cycle --requests until this many seconds after the window opened")
    ap.add_argument("--max-tokens", type=int, default=170)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--idle-s", type=float, default=8.0, help="measure the loaded server's idle CPU for this long")
    ap.add_argument("--post-idle-s", type=float, default=0.0, help="measure idle CPU again after the requests")
    ap.add_argument("--prewarm", action="store_true", help="evaluate the system prompt once before the first request")
    ap.add_argument("--cold", action="store_true", help="evict the model from the page cache first, as after a reboot")
    ap.add_argument("--vision", choices=["none", "idle", "loop"], default="none")
    ap.add_argument("--vision-threads", type=int, default=3)
    ap.add_argument("--vision-hz", type=float, default=2.0)
    ap.add_argument("--vision-baseline-s", type=float, default=15.0, help="reads with no LLM process, before it starts")
    ap.add_argument("--vision-python", default=str(REPO / "device/.venv/bin/python"))
    ap.add_argument("--env", default="", help="extra environment for the server, e.g. OMP_WAIT_POLICY=passive")
    ap.add_argument("--server-args", default="", help="extra llama-server arguments, e.g. '--poll 0'")
    ap.add_argument("--temp-start-max", type=float, default=60.0)
    ap.add_argument("--temp-abort", type=float, default=72.0)
    ap.add_argument("--max-seconds", type=float, default=175.0)
    ap.add_argument("--out-dir", default=str(REPO / "data/llm/run_logs"))
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path, sample_path = out_dir / f"{a.name}.json", out_dir / f"{a.name}.samples.jsonl"
    server_log = out_dir / f"{a.name}.server.log"
    vision_log = out_dir / f"{a.name}.vision.jsonl"
    for p in (sample_path, server_log, vision_log):
        p.unlink(missing_ok=True)

    result = {"name": a.name, "args": vars(a), "aborted": None, "requests": []}

    def save():
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1))
        os.replace(tmp, out_path)

    start_temp = hottest()
    result["start_hot_c"] = start_temp
    if start_temp > a.temp_start_max:
        result["aborted"] = {"reason": "too_hot_to_start", "hot_c": start_temp}
        save()
        print(json.dumps(result["aborted"]))
        return 3
    if any("llama-server" in c for c, _ in all_procs().values()):
        result["aborted"] = {"reason": "llama_server_already_running"}
        save()
        print(json.dumps(result["aborted"]))
        return 4

    examples = json.loads(Path(a.examples_file).read_text())
    system = examples["system"]
    procs = {}

    def kill_children():
        for p in procs.values():
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def on_signal(signum, _):
        result["aborted"] = result["aborted"] or {"reason": f"signal_{signum}"}
        raise SystemExit(130)
    for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    result["baseline"] = baseline(5.0)  # idle system, with the always-on app and the kiosk included
    sampler = Sampler(sample_path, a.temp_abort, a.max_seconds, kill_children)
    sampler.start()
    t_window = time.monotonic()
    try:
        if a.vision != "none":
            sampler.phase = "vision_load"
            procs["vision"] = subprocess.Popen(
                [a.vision_python, "-I", str(REPO / "scripts/llm/vision_loop.py"), "--threads", str(a.vision_threads),
                 "--hz", str(a.vision_hz), "--mode", a.vision, "--log", str(vision_log)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, preexec_fn=child_setup)
            sampler.pids["vision"] = procs["vision"].pid
            for _ in range(600):
                if vision_log.exists() and vision_log.stat().st_size:
                    break
                if procs["vision"].poll() is not None or sampler.tripped:
                    raise RuntimeError("vision process failed to start")
                time.sleep(0.1)
            result["vision_ready"] = json.loads(vision_log.read_text().splitlines()[0])
            if a.vision == "loop":
                sampler.phase = "vision_alone"
                time.sleep(a.vision_baseline_s)
        result["t_vision_alone_end"] = time.monotonic()

        if a.cold:  # same state as just after a reboot: the file's pages are not in the page cache
            fd = os.open(a.model, os.O_RDONLY)
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)

        cmd = [a.bin, "-m", a.model, "-c", str(a.ctx), "-t", str(a.threads), "-tb", str(a.threads), "-np", "1",
               "--host", HOST, "--port", str(PORT), "--no-webui", "--cache-ram", "0", "--ctx-checkpoints", "0",
               "--seed", str(a.seed)] + shlex.split(a.server_args)
        if a.nice:
            cmd = ["nice", "-n", str(a.nice)] + cmd
        env = dict(os.environ)
        for kv in shlex.split(a.env):
            k, _, v = kv.partition("=")
            env[k] = v
        result["cmd"] = shlex.join(cmd)
        sampler.phase = "llm_load"
        t_load0 = time.monotonic()
        with open(server_log, "wb") as lf:
            procs["server"] = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                               env=env, preexec_fn=child_setup)
        sampler.pids["server"] = procs["server"].pid
        while True:
            if procs["server"].poll() is not None or sampler.tripped:
                raise RuntimeError(f"server exited during load, rc={procs['server'].returncode}")
            try:
                st, _ = http_json("GET", "/health", timeout=2)
                if st == 200:
                    break
            except (OSError, http.client.HTTPException, socket.timeout):
                pass
            time.sleep(0.1)
        result["load_s"] = time.monotonic() - t_load0
        result["t_loaded"] = time.monotonic()
        result["rss_after_load"] = proc_status(procs["server"].pid)
        save()

        sampler.phase = "idle"
        if a.idle_s > 0:
            result["idle_cpu"] = cpu_pct_over(procs["server"].pid, a.idle_s)
        result["t_idle_end"] = time.monotonic()

        ex = examples["examples"]
        idx = [int(x) for x in a.requests.split(",") if x != ""]
        if idx:
            tmpl_body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": ex[idx[0]]["user"]}],
                         "add_generation_prompt": True}
            if a.no_think:
                tmpl_body["chat_template_kwargs"] = {"enable_thinking": False}
            st, tp = http_json("POST", "/apply-template", tmpl_body)
            result["rendered_prompt"] = tp.get("prompt") if isinstance(tp, dict) else str(tp)
            st, tk = http_json("POST", "/tokenize", {"content": result["rendered_prompt"], "add_special": False, "parse_special": True})
            result["rendered_prompt_tokens"] = len(tk.get("tokens", [])) if isinstance(tk, dict) else None
        if a.prewarm:
            sampler.phase = "prewarm"
            result["prewarm"] = explain(system, "", argparse.Namespace(**{**vars(a), "max_tokens": 1}), sampler, "prewarm")
        n = 0
        while idx and not sampler.tripped:
            i = idx[n % len(idx)]
            if a.repeat_until_s > 0:
                if time.monotonic() - t_window > a.repeat_until_s:
                    break
            elif n >= len(idx):
                break
            r = explain(system, ex[i]["user"], a, sampler, f"r{n}_{ex[i]['name']}")
            r["example"] = ex[i]["name"]
            r["hot_c_peak_so_far"] = max(x["hot_c"] for x in sampler.samples)
            result["requests"].append(r)
            save()
            n += 1
        if a.post_idle_s > 0 and not sampler.tripped:
            sampler.phase = "post_idle"
            result["post_idle_cpu"] = cpu_pct_over(procs["server"].pid, a.post_idle_s)
        result["rss_peak"] = proc_status(procs["server"].pid)
    except (RuntimeError, OSError, http.client.HTTPException) as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        sampler.phase = "stopping"
        result["window_s"] = time.monotonic() - t_window
        if sampler.tripped:
            result["aborted"] = sampler.tripped
        for name, p in procs.items():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and any(p.poll() is None for p in procs.values()):
            time.sleep(0.05)
        kill_children()
        for p in procs.values():
            p.wait()
        sampler.stop_flag.set()
        sampler.join(3)

        s = sampler.samples
        result["peak_hot_c"] = max((x["hot_c"] for x in s), default=None)
        result["end_hot_c"] = s[-1]["hot_c"] if s else None
        result["min_avail_mb"] = min((x["avail_mb"] for x in s), default=None)
        result["max_swap_used_mb"] = max((x["swap_used_mb"] for x in s), default=None)
        result["peak_server_rss_mb"] = max((x.get("server_rss_mb", 0) for x in s), default=None)
        result["peak_server_anon_mb"] = max((x.get("server_anon_mb", 0) for x in s), default=None)
        result["peak_vision_rss_mb"] = max((x.get("vision_rss_mb", 0) for x in s), default=None)
        result["min_mhz_seen_under_load"] = min((min(x["mhz"]) for x in s if x["busy_pct"] > 90), default=None)
        result["samples"] = [{k: x[k] for k in ("t", "phase", "hot_c", "busy_pct", "avail_mb") } for x in s]
        if vision_log.exists() and a.vision == "loop":
            rows = [json.loads(line) for line in vision_log.read_text().splitlines()]
            rows = [r for r in rows if "ms" in r]
            t_alone = result.get("t_vision_alone_end", 0)
            sl = {"alone_no_llm": read_stats([r for r in rows if r["t"] < t_alone])}
            if "t_loaded" in result:
                sl["llm_loaded_idle"] = read_stats([r for r in rows if result["t_loaded"] <= r["t"] < result.get("t_idle_end", 0)])
            reqs = result["requests"]
            during = []
            for q in reqs:
                if q.get("t_start"):
                    during += [r for r in rows if q["t_start"] <= r["t"] < q["t_end"]]
            sl["llm_busy"] = read_stats(during)
            result["vision_read_latency"] = sl
        elif vision_log.exists():
            result["vision_read_latency"] = {"note": "vision loaded, not reading"}
        err = (server_log.read_text(errors="replace") if server_log.exists() else "")
        keep = [ln for ln in err.splitlines()
                if re.search(r"system_info|CPU_REPACK|repack|model buffer size|CPU KV buffer|graph splits|n_ctx |n_ctx_seq|"
                             r"print_info: (file type|file size|n_params|arch|model type)|CPU compute buffer|error|failed|warn", ln, re.I)]
        result["server_log_extract"] = keep[:50]
        result["repack_seen"] = bool(re.search(r"CPU_REPACK|repack: *repack", err))
        result["left_running"] = [c for c, _ in all_procs().values() if "llama-server" in c]
        result["boot_id_end"] = (read("/proc/sys/kernel/random/boot_id") or "").strip()
        save()
    print(json.dumps({"name": a.name, "aborted": result["aborted"], "error": result.get("error"), "peak_hot_c": result["peak_hot_c"],
                      "window_s": round(result["window_s"], 1), "load_s": result.get("load_s"), "out": str(out_path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
