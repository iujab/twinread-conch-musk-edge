#!/usr/bin/env python3
"""Runs ON THE BOARD, with Wi-Fi off, so there is no SSH to it: it logs to a file for collecting afterwards. Stdlib only, run by root.

  offline_check.py radio-off LOG   turn the radio off, check, explain, back to standby. A restore must be armed first (see below).
  offline_check.py boot LOG        the same checks during a boot that came up with the radio already off.

Turning the radio back on is not this script's job: scripts/uno_q/verify_explainer.py arms it first, and it survives this script failing.
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8080"
ACTIVE = ("starting", "preparing", "thinking", "writing", "checking")
SLIDE = 1  # a Confident slide


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except subprocess.TimeoutExpired:
        return "timed out"


def network():
    return {"default_route": sh("ip route show default"), "wlan0": sh("ip -4 -br addr show wlan0"), "radio": sh("nmcli radio wifi"), "can_reach_outside": sh("ping -c1 -W1 1.1.1.1 >/dev/null 2>&1 && echo yes || echo no"),
            "tailscale_backend": sh("tailscale status --json 2>/dev/null | python3 -c \"import sys,json; print(json.load(sys.stdin).get('Self',{}).get('Online'))\"")}


def state():
    return json.load(urllib.request.urlopen(URL + "/state", timeout=4))


def control(cmd):
    req = urllib.request.Request(URL + "/control", json.dumps(cmd).encode(), {"Content-Type": "application/json"}, method="POST")
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, **json.load(e)}


def main():
    mode, log = sys.argv[1], sys.argv[2]
    rep = {"simulated": False, "mode": mode, "kernel": sh("uname -r"), "boot_id": sh("cat /proc/sys/kernel/random/boot_id"), "uptime_at_start_s": float(open("/proc/uptime").read().split()[0]), "events": []}

    def note(what, **kw):
        rep["events"].append({"uptime_s": float(open("/proc/uptime").read().split()[0]), "what": what, **kw})
        with open(log, "w") as f:
            json.dump(rep, f, indent=1)

    try:
        if mode == "radio-off":
            note("radio_off_command", output=sh("nmcli radio wifi off; nmcli radio wifi"))
            time.sleep(3)
        note("network_before", **network())
        t = time.monotonic()
        while time.monotonic() - t < 120:
            try:
                s = state()
                break
            except OSError:
                time.sleep(1)
        else:
            note("page_never_answered")
            return
        note("page_answers", status=s["status"], model_loaded=s["controls"]["model_loaded"], camera_open=s["controls"]["camera_open"], explainer=s["explain"]["server"], after_s=round(time.monotonic() - t, 1),
             html_has_title=b"TwinRead" in urllib.request.urlopen(URL + "/", timeout=4).read())
        rep["service"] = {"active": sh("systemctl is-active twinread"), "enabled": sh("systemctl is-enabled twinread"), "restarts": sh("systemctl show twinread -p NRestarts --value")}
        control({"source": "demo", "feed_paused": True, "model_paused": False, "demo_index": SLIDE})
        t = time.monotonic()
        while time.monotonic() - t < 15 and not state().get("call"):
            time.sleep(0.2)
        s = state()
        note("decision", status=s["status"], label=(s.get("call") or {}).get("label"))
        s = control({"explain": "doctor"})
        note("explain_pressed", job=(s.get("explain") or {}).get("job") and {"phase": s["explain"]["job"]["phase"], "source": s["explain"]["job"]["source"]}, error=s.get("error"))
        t = time.monotonic()
        seen = set()
        job = None
        while time.monotonic() - t < 110:
            job = state()["explain"]["job"]
            if job and job["phase"] not in seen:
                seen.add(job["phase"])
                note("phase", phase=job["phase"], default_route=sh("ip route show default"))
            if job is None or job["phase"] not in ACTIVE:
                break
            time.sleep(1)
        note("answer", job=job, seconds=round(time.monotonic() - t, 1))
        note("network_after_answer", **network())
    except Exception as e:  # noqa: BLE001 - the log is the only way anyone hears about this
        note("failed", error=repr(e))
    finally:
        try:
            control({"explain_stop": True})
            control({"explain_close": True})
            control({"source": "camera", "feed_paused": True, "model_paused": True})
            sh("pkill -x llama-server")
            time.sleep(3)
            s = state()
            note("standby", status=s["status"], model_loaded=s["controls"]["model_loaded"], camera_open=s["controls"]["camera_open"], explainer=s["explain"]["server"], llama=sh("pgrep -x llama-server"))
        except Exception as e:  # noqa: BLE001
            note("standby_failed", error=repr(e))
        note("done")


if __name__ == "__main__":
    main()
