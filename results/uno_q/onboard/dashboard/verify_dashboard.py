#!/usr/bin/env python
"""On-board evidence for the dashboard. Runs on the PC and drives the real board over the tailnet: POST /control from standby,
and back to standby at the end of every window. Hot windows go under the board lock:
    flock ~/.config/twinread/board.lock .venv/bin/python verify_dashboard.py model_camera

usage: verify_dashboard.py standby|model_camera|demo|viewer|crash|reboot|summary
Writes <phase>.json next to this file. CPU is the app's own utime+stime from /proc, sampled over a window with nothing polling
it, as a percentage of ONE core (the board has four). Aborts to standby above 70 C and will not start above 60 C.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = os.environ.get("TWINREAD_URL", "http://uno-q:8080")
HOME = Path.home()
SSH = ["ssh", "-F", str(HOME / ".config/twinread/ssh_config"), "-o", "BatchMode=yes", "uno-q"]
START_LIMIT_C, ABORT_C = 60.0, 70.0

SAMPLER = r'''
import glob, json, os, subprocess, sys, time
secs = float(sys.argv[1])
pid = int(subprocess.run(["systemctl", "show", "twinread", "-p", "MainPID", "--value"], capture_output=True, text=True).stdout.strip() or 0)
hz = os.sysconf("SC_CLK_TCK")
def app():
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12])
def machine():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]
def hottest():
    out = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try: out.append(int(open(z).read()) / 1000)
        except OSError: pass
    return max(out)
a0, (t0, i0), w0, peak = app(), machine(), time.time(), hottest()
while time.time() - w0 < secs:
    time.sleep(min(2.0, max(0.05, secs - (time.time() - w0))))
    peak = max(peak, hottest())
a1, (t1, i1), w1 = app(), machine(), time.time()
st = dict(l.split(":", 1) for l in open(f"/proc/{pid}/status") if ":" in l)
print(json.dumps({"seconds": round(w1 - w0, 1), "pid": pid, "app_cpu_pct_of_one_core": round(100 * (a1 - a0) / hz / (w1 - w0), 2),
                  "machine_busy_pct_of_all_cores": round(100 * (1 - (i1 - i0) / max(1, t1 - t0)), 2),
                  "app_rss_mb": round(int(st["VmRSS"].split()[0]) / 1024, 1), "app_threads": int(st["Threads"].strip()),
                  "hottest_c_end": hottest(), "hottest_c_peak": peak, "load": open("/proc/loadavg").read().split()[:3]}))
'''

PROOF = r'''
echo "video_holders=$(fuser /dev/video* 2>/dev/null | tr -s ' ')"
for d in /sys/bus/usb/devices/*; do
  [ "$(cat $d/idVendor 2>/dev/null)" = 3564 ] || continue
  echo "usb_runtime_status=$(cat $d/power/runtime_status)"
  echo "usb_alt_settings=$(cat $d/*/bAlternateSetting 2>/dev/null | tr -s ' \n' ' ')"
done
top -bn1 -o %CPU | sed -n '8,12p' | awk '{print "top=" $9 "% " $10 "% " $12}'
'''


def ssh(cmd, stdin=None, timeout=60):
    r = subprocess.run(SSH + [cmd], input=stdin, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def sample(secs):
    return json.loads(ssh(f"python3 - {secs}", stdin=SAMPLER, timeout=secs + 30).splitlines()[-1])


def proof():
    out = {"top": []}
    for line in ssh("bash -s", stdin=PROOF).splitlines():
        k, _, v = line.partition("=")
        if k == "top":
            out["top"].append(v)
        else:
            out[k] = v.strip()
    return out


def hottest():
    return max(int(x) / 1000 for x in ssh("cat /sys/class/thermal/thermal_zone*/temp").split())


STALLS = []  # requests that timed out on the way to the board: counted and reported, because they are about the network


def state():
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(URL + "/state", timeout=5))
        except (urllib.error.URLError, TimeoutError) as e:
            STALLS.append({"at": time.strftime("%H:%M:%S", time.gmtime()), "error": str(e)})
            if attempt == 3:
                raise


def control(cmd):
    """(state, seconds the command took). HTTP errors come back as {"error": ...}."""
    req = urllib.request.Request(URL + "/control", json.dumps(cmd).encode(), {"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r), round(time.perf_counter() - t, 3)
    except urllib.error.HTTPError as e:
        return json.load(e), round(time.perf_counter() - t, 3)


def to_standby():
    """Every window ends here, and it is what an abort does."""
    st, _ = control({"source": "camera", "feed_paused": True, "model_paused": True})
    return st


def until(cond, timeout=15.0, every=0.25):
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        v = cond()
        if v:
            return round(time.perf_counter() - t, 3), v
        time.sleep(every)
    return None, None


def brief(st):
    c = st["controls"]
    return {"status": st["status"], "frame_label": st["frame_label"], "model_loaded": c["model_loaded"], "camera_open": c["camera_open"],
            "call": (st.get("call") or {}).get("label"), "reads": st["reads"], "camera_fps": st["camera"]["fps"], "latency_ms": st.get("latency_ms")}


class TooHot(Exception):
    pass


def guard(rep):
    t = hottest()
    rep.setdefault("temps_c", []).append(t)
    if t > ABORT_C:
        raise TooHot(f"{t} C")
    return t


def hot_window(fn):
    """Refuse a warm start; whatever happens, end in standby and say how it ended."""
    def run():
        rep = {"simulated": False, "url": URL, "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
        t = hottest()
        rep["temp_at_start_c"] = t
        if t > START_LIMIT_C:
            rep["refused"] = f"board at {t} C, limit {START_LIMIT_C} C"
            return rep
        before = state()
        assert before["status"] == "standby", before["status"]
        rep["standby_before"] = brief(before)
        try:
            fn(rep)
        except TooHot as e:
            rep["aborted_too_hot"] = str(e)
        finally:
            rep["standby_after"] = brief(to_standby())
            rep["proof_after"] = proof()
            rep["temp_at_end_c"] = hottest()
        return rep
    return run


def phase_standby():
    st = state()
    rep = {"simulated": False, "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "state": brief(st),
           "note": "No browser on the board and nothing polling the app during the window.",
           "window": sample(30), "proof": proof(), "enabled": ssh("systemctl is-enabled twinread"), "active": ssh("systemctl is-active twinread"),
           "defaults": ssh("cat /etc/default/twinread")}
    return rep


@hot_window
def phase_model_camera(rep):
    steps = rep["steps"] = {}
    # Both on, from standby, each by its own command.
    st, dt_model = control({"model_paused": False})
    steps["start_model"] = {"post_s": dt_model, "state": brief(st)}
    st, dt_cam = control({"feed_paused": False})
    t_first, st = until(lambda: (lambda s: s if s.get("call") else None)(state()))
    steps["start_camera"] = {"post_s": dt_cam, "first_call_s": t_first, "state": brief(st)}
    guard(rep)
    time.sleep(3)
    steps["both_on"] = {"window": sample(20), "proof": proof(), "state": brief(state())}
    guard(rep)

    st, dt = control({"model_paused": True})
    steps["pause_model"] = {"post_s": dt, "state": brief(st), "call_is_none": st["call"] is None}
    time.sleep(2)
    steps["model_off_camera_on"] = {"window": sample(20), "proof": proof(), "state": brief(state())}
    guard(rep)

    st, dt = control({"model_paused": False})
    t_call, st = until(lambda: (lambda s: s if s.get("call") else None)(state()))
    steps["resume_model"] = {"post_s": dt, "first_call_s": t_call, "state": brief(st)}

    st, dt = control({"feed_paused": True})
    p = proof()
    steps["pause_camera"] = {"post_s": dt, "state": brief(st), "proof_immediately_after": p}
    time.sleep(1.5)
    reads0 = state()["reads"]
    steps["camera_off_model_on"] = {"window": sample(15), "proof": proof()}
    st = state()
    steps["camera_off_model_on"]["state"] = brief(st)
    steps["camera_off_model_on"]["reads_during_window"] = st["reads"] - reads0  # a held frame is read once, before the window
    guard(rep)

    st, dt = control({"feed_paused": False})
    t_open, _ = until(lambda: state()["controls"]["camera_open"])
    t_call, st = until(lambda: (lambda s: s if s.get("call") and s["frame_label"] == "Live" else None)(state()))
    steps["resume_camera"] = {"post_s": dt, "open_s": t_open, "first_call_s": t_call, "state": brief(st), "proof": proof()}
    guard(rep)

    to_standby()
    time.sleep(1)
    steps["back_to_standby"] = {"window": sample(15), "proof": proof(), "state": brief(state())}


@hot_window
def phase_demo(rep):
    st, dt = control({"source": "demo", "model_paused": False, "feed_paused": True, "demo_index": 0})
    total = st["demo"]["total"]
    rep["start_post_s"] = dt
    rows = []
    for i in range(total):
        st, _ = control({"demo_index": i})
        assert st["call"] is None and st["demo"]["index"] == i  # no decision from the slide before
        _, s = until(lambda: (lambda x: x if x.get("call") else None)(state()), timeout=5, every=0.05)
        rows.append({"index": i, "file": s["demo"]["file"], "truth": s["truth"], "label": s["call"]["label"], "match": s["truth_match"],
                     "status": s["status"], "reasons": s["reasons"], "prob": round(s["call"]["prob"], 4), "labelA": s["call"]["labelA"], "labelB": s["call"]["labelB"],
                     "latency_ms": round(s["latency_ms"], 1), "led": s["led"], "led_state": s["led_state"]})
    rep["slides"] = rows
    rep["summary"] = {"slides": total, "match_true_label": sum(r["match"] for r in rows), "agree": sum(r["status"] == "agree" for r in rows),
                      "review": sum(r["status"] == "review" for r in rows), "no_tissue": sum(r["status"] == "no_tissue" for r in rows),
                      "review_and_wrong": sum(r["status"] == "review" and not r["match"] for r in rows),
                      "agree_and_wrong": sum(r["status"] == "agree" and not r["match"] for r in rows),
                      "latency_ms_median": sorted(r["latency_ms"] for r in rows)[total // 2], "selection": "fixed seed, not filtered by model output (data/demo_slides/manifest.json)"}
    guard(rep)

    # A held slide is read once.
    r0 = state()["reads"]
    time.sleep(5)
    rep["held_slide_reads_in_5s"] = state()["reads"] - r0

    # Auto-advance, including the wrap from the last slide to the first.
    st, _ = control({"demo_index": total - 2, "demo_interval": 2})
    st, _ = control({"feed_paused": False})
    seen, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < 7.5:
        s = state()
        if not seen or seen[-1]["index"] != s["demo"]["index"]:
            seen.append({"t": round(time.perf_counter() - t0, 2), "index": s["demo"]["index"]})
        time.sleep(0.2)
    rep["auto_advance_2s"] = seen
    guard(rep)

    # What a running deck costs at the default pace.
    control({"demo_interval": 4})
    rep["running_4s_pace"] = {"window": sample(20), "proof": proof(), "state": brief(state())}
    guard(rep)
    # The model off with the slides still running.
    control({"model_paused": True})
    rep["running_model_off"] = {"window": sample(10), "state": brief(state())}


def phase_viewer():
    """Real Chromium on the PC, viewing over the tailnet: what a viewer costs the app, and the real screenshots."""
    import asyncio
    sys.path.insert(0, str(HERE))
    from cdp import open_page
    out = HERE / "screenshots"
    out.mkdir(exist_ok=True)
    rep = {"simulated": False, "url": URL, "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "shots": [], "viewer": {}}
    rep["temp_at_start_c"] = hottest()
    if rep["temp_at_start_c"] > START_LIMIT_C:
        rep["refused"] = "too warm"
        return rep

    async def run():
        p = await open_page()
        sizes = [(1920, 1080, False), (1280, 720, False), (1024, 600, False), (390, 844, True)]

        async def shots(name, every=False):
            for w, h, phone in (sizes if every else sizes[1:2]):
                await p.size(w, h, phone)
                await asyncio.sleep(0.5)
                f = out / f"{name}_{w}x{h}.png"
                await p.shot(f, full=phone)
                info = await p.js("({band:band.textContent, call:call.textContent, source:source.textContent, img:frame.naturalWidth, overflow:document.documentElement.scrollHeight>innerHeight})")
                rep["shots"].append({"file": f"screenshots/{f.name}", **info})

        await p.size(1280, 720)
        await p.goto(URL + "/", 1.5)
        try:
            # Standby, viewed
            rep["viewer"]["standby"] = sample(15)
            await shots("standby", every=True)
            # Live, viewed
            control({"model_paused": False, "feed_paused": False})
            until(lambda: (lambda s: s if s.get("call") else None)(state()))
            await asyncio.sleep(2)
            await shots("live", every=True)
            rep["viewer"]["both_on"] = sample(15)
            rep["viewer"]["both_on"]["state"] = brief(state())
            rep["temps_c"] = [hottest()]
            # Model paused: live video, no call
            control({"model_paused": True})
            await asyncio.sleep(1.5)
            await shots("model_paused")
            rep["viewer"]["model_off_camera_on"] = sample(10)
            # Frozen: camera closed, the model reads the held frame
            control({"model_paused": False})
            await asyncio.sleep(1.0)
            control({"feed_paused": True})
            await asyncio.sleep(1.5)
            rep["frozen_proof"] = proof()
            await shots("frozen")
            # Both off with the frame held
            control({"model_paused": True})
            await asyncio.sleep(1.0)
            await shots("held_standby")
            rep["temps_c"].append(hottest())
            # Demo slides, an Agree and a Review from the real model
            st, _ = control({"source": "demo", "model_paused": False, "feed_paused": True, "demo_index": 0})
            found = set()
            for i in range(st["demo"]["total"]):
                control({"demo_index": i})
                _, s = until(lambda: (lambda x: x if x.get("call") else None)(state()), timeout=5, every=0.05)
                if s["status"] not in found:
                    found.add(s["status"])
                    await asyncio.sleep(2.2)  # the slow poll of a held slide
                    await shots(f"demo_{s['status']}", every=True)
                if not s["truth_match"] and "mismatch" not in found:
                    found.add("mismatch")
                    await asyncio.sleep(2.2)
                    await shots("demo_mismatch")
            rep["temps_c"].append(hottest())
        finally:
            rep["console_errors"] = list(p.errors)
            rep["standby_after"] = brief(to_standby())
            await asyncio.sleep(2.5)
            await p.size(1280, 720)
            await p.goto(URL + "/", 1.5)
            await shots("standby_after_window")
            await p.close()

    asyncio.run(run())
    rep["proof_after"] = proof()
    rep["temp_at_end_c"] = hottest()
    return rep


def phase_crash():
    """kill -9 the app: systemd must bring it back by itself, in standby, without help."""
    before = state()
    pid = ssh("systemctl show twinread -p MainPID --value")
    n0 = ssh("systemctl show twinread -p NRestarts --value")
    t0 = time.perf_counter()
    ssh(f"kill -9 {pid}")
    down, _ = until(lambda: not _answers(), timeout=5, every=0.05)
    up, _ = until(_answers, timeout=30, every=0.1)
    html = urllib.request.urlopen(URL + "/", timeout=5).read()
    st = state()
    return {"simulated": False, "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "killed_pid": int(pid), "new_pid": int(ssh("systemctl show twinread -p MainPID --value")),
            "before": brief(before), "noticed_down_s": down, "back_after_s": up, "total_s": round(time.perf_counter() - t0, 2), "after": brief(st), "page_served": b"TwinRead" in html,
            "restarts": [int(n0), int(ssh("systemctl show twinread -p NRestarts --value"))], "proof": proof()}


def _answers():
    try:
        urllib.request.urlopen(URL + "/state", timeout=2).read()
        return True
    except OSError:
        return False


def phase_reboot():
    pw = (HOME / ".config/twinread/unoq-sudo-password").read_text()
    boots0 = ssh("journalctl --list-boots --no-pager")
    rep = {"simulated": False, "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "boots_before": boots0.splitlines(), "state_before": brief(state())}
    t0 = time.perf_counter()
    subprocess.run(SSH + ['sudo -S -k -p "" systemctl reboot'], input=pw, capture_output=True, text=True, timeout=30)
    down, _ = until(lambda: not _answers(), timeout=60, every=0.5)
    rep["went_down_s"] = down
    up, _ = until(_answers, timeout=240, every=1.0)
    rep["dashboard_up_after_reboot_s"] = round(time.perf_counter() - t0, 1) if up is not None else None
    st = state()
    html = urllib.request.urlopen(URL + "/", timeout=5).read()
    rep.update(state_first_answer=brief(st), page_served=b"TwinRead" in html, model_error=st["model_error"], led=st["led"], led_state=st["led_state"],
               system=st["system"])
    time.sleep(5)
    rep.update(window=sample(30), proof=proof(), state=brief(state()), enabled=ssh("systemctl is-enabled twinread"), active=ssh("systemctl is-active twinread"),
               nrestarts=ssh("systemctl show twinread -p NRestarts --value"), boots_after=ssh("journalctl --list-boots --no-pager").splitlines(),
               chromium_procs=ssh("pgrep -c chromium || true"), connectors=ssh("cat /sys/class/drm/card*-*/status").split(),
               tailscale_serve=ssh("tailscale serve status 2>&1 | head -4").splitlines(), journal=ssh("journalctl -u twinread -b --no-pager -o cat | tail -5").splitlines())
    return rep


def phase_summary():
    """The headline numbers, read back from the phase files, and whether the board runs exactly this code."""
    load = lambda n: json.loads((HERE / f"{n}.json").read_text())
    mc, de, vi, st, cr, rb = load("model_camera")["steps"], load("demo"), load("viewer"), load("standby"), load("crash"), load("reboot")
    pick = lambda w: {"app_cpu_pct_of_one_core": w["app_cpu_pct_of_one_core"], "app_rss_mb": w["app_rss_mb"], "threads": w["app_threads"]}
    temps = [mc[k]["window"]["hottest_c_peak"] for k in ("both_on", "model_off_camera_on", "camera_off_model_on", "back_to_standby")]
    temps += [de["running_4s_pace"]["window"]["hottest_c_peak"], de["running_model_off"]["window"]["hottest_c_peak"]]
    temps += [v["hottest_c_peak"] for v in vi["viewer"].values()]
    root = HERE.parents[3]
    names = ["device/uno_q_app.py", "device/camera.py", "device/demo.py", "device/twinread_runtime.py", "device/bridge.py", "device/ui/app.js", "device/ui/index.html", "device/ui/style.css"]
    board = dict(reversed(l.split()) for l in ssh("cd ~/twinread-conch-musk-edge && sha256sum " + " ".join(names)).splitlines())
    local = dict(reversed(l.split()) for l in subprocess.run(["sha256sum", *names], cwd=root, capture_output=True, text=True).stdout.splitlines())
    return {"simulated": False, "board_runs_this_code": board == local,
            "fresh_standby_no_viewer": pick(st["window"]), "after_reboot_standby_no_viewer": pick(rb["window"]),
            "both_on_no_viewer": pick(mc["both_on"]["window"]), "model_off_camera_on_no_viewer": pick(mc["model_off_camera_on"]["window"]),
            "camera_off_model_on_no_viewer": pick(mc["camera_off_model_on"]["window"]), "back_to_standby_after_model_was_loaded": pick(mc["back_to_standby"]["window"]),
            "demo_running_4s_pace_no_viewer": pick(de["running_4s_pace"]["window"]), "demo_running_model_off_no_viewer": pick(de["running_model_off"]["window"]),
            "standby_with_one_viewer": pick(vi["viewer"]["standby"]), "both_on_with_one_viewer": pick(vi["viewer"]["both_on"]),
            "model_off_camera_on_with_one_viewer": pick(vi["viewer"]["model_off_camera_on"]),
            "camera_holders": {"both_on": mc["both_on"]["proof"]["video_holders"], "right_after_pause": mc["pause_camera"]["proof_immediately_after"]["video_holders"],
                               "camera_off_model_on": mc["camera_off_model_on"]["proof"]["video_holders"], "after_resume": mc["resume_camera"]["proof"]["video_holders"]},
            "camera_usb_runtime_status": {"both_on": mc["both_on"]["proof"]["usb_runtime_status"], "camera_off_model_on": mc["camera_off_model_on"]["proof"]["usb_runtime_status"]},
            "timings_s": {"start_model_first_time": mc["start_model"]["post_s"], "start_camera_to_first_call": mc["start_camera"]["first_call_s"], "pause_model": mc["pause_model"]["post_s"],
                          "resume_model": mc["resume_model"]["post_s"], "pause_camera": mc["pause_camera"]["post_s"], "resume_camera_to_open": mc["resume_camera"]["open_s"]},
            "held_frame_reads_in_15s": mc["camera_off_model_on"]["reads_during_window"], "held_slide_reads_in_5s": de["held_slide_reads_in_5s"], "demo": de["summary"],
            "auto_advance_2s": de["auto_advance_2s"], "peak_temp_during_windows_c": max(temps), "crash": {k: cr[k] for k in ("back_after_s", "total_s", "after", "page_served", "restarts")},
            "reboot": {k: rb[k] for k in ("dashboard_up_after_reboot_s", "state_first_answer", "page_served", "enabled", "active", "nrestarts", "chromium_procs", "boots_after")}}


def main():
    phase = sys.argv[1]
    rep = globals()[f"phase_{phase}"]()
    rep["network_stalls"] = STALLS
    (HERE / f"{phase}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
