#!/usr/bin/env python3
"""Print the report's tables from results/llm/runs/*.json and save them as results/llm/report_tables.md.

Derived numbers are computed here so they trace back to the runs: speedup ratios, "time for 100 words" with the shared system
prompt cached, and the sustained-heat curves. Also writes results/llm/report_numbers.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "results/llm/runs"


def load(name):
    p = RUNS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def reqs(d):
    return [q for q in d.get("requests", []) if q.get("gen_tps")]


def stats(d):
    rs = reqs(d)
    cold = next((q for q in rs if not q.get("cache_n")), None)
    cached = [q for q in rs if q.get("cache_n")]
    out = {"gen": round(median(q["gen_tps"] for q in rs), 2) if rs else None,
           "pp_cold": round(cold["prompt_tps"], 1) if cold else None, "pp_cached": round(median(q["prompt_tps"] for q in cached), 1) if cached else None,
           "ttft_cold": round(cold["ttft_s"], 1) if cold else None, "ttft_cached": round(median(q["ttft_s"] for q in cached), 1) if cached else None,
           "rss": d.get("peak_server_rss_mb"), "anon": d.get("peak_server_anon_mb"), "peak": d.get("peak_hot_c"), "start": d.get("start_hot_c"),
           "load": round(d.get("load_s") or 0, 1), "window": round(d.get("window_s") or 0), "min_avail": d.get("min_avail_mb"),
           "aborted": (d.get("aborted") or {}).get("reason")}
    tpw = [q["tokens_per_word"] for q in rs if q.get("tokens_per_word")]
    out["tpw"] = round(median(tpw), 2) if tpw else None
    if cached and out["gen"] and out["tpw"]:
        out["s100_cached"] = round(out["ttft_cached"] + 100 * out["tpw"] / out["gen"], 1)
        out["s100_cached_tpw150"] = round(out["ttft_cached"] + 150 / out["gen"], 1)  # same 1.5 tokens per word for every run
    if cold and out["gen"] and out["tpw"]:
        out["s100_cold"] = round(out["ttft_cold"] + 100 * out["tpw"] / out["gen"], 1)
    return out


def table(title, names, cols):
    lines = [f"### {title}", "", "| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    rows = {}
    for n in names:
        d = load(n)
        if not d:
            lines.append(f"| {n} | not run |" + " |" * (len(cols) - 1))
            continue
        s = stats(d)
        rl = d.get("vision_read_latency") or {}
        for key, tag in (("alone_no_llm", "read_alone"), ("llm_loaded_idle", "read_idle"), ("llm_busy", "read_busy")):
            b = rl.get(key) or {}
            s[tag + "_p50"], s[tag + "_p95"], s[tag + "_max"] = b.get("p50_ms"), b.get("p95_ms"), b.get("max_ms")
        s["reads_late"] = (rl.get("llm_busy") or {}).get("late_over_100ms")
        s["vision_rss"] = d.get("peak_vision_rss_mb") or None
        rows[n] = s
        lines.append(f"| {n} | " + " | ".join("" if s.get(c) is None else str(s.get(c)) for c in cols) + " |")
    return lines + [""], rows


def main() -> None:
    out, nums = [], {}
    speed = ["gen", "pp_cold", "ttft_cold", "ttft_cached", "s100_cold", "s100_cached", "tpw", "rss", "anon", "peak", "window", "aborted"]
    sets = [
        ("Quantization at 4 threads, no vision", ["p0_q25_q4km_t4", "q25_q4_0_speedonly_t4", "q25_q8_0_speedonly_t4", "p0_q3_q4km_t4", "q3_q4_0_real_t4",
                                                  "q3_q4_0_embq8_t4", "q3_q4_0_embq4_t4", "q3_q4km_embq8_t4", "q3_q4km_embq4_t4", "q3_q5_0_embq8_speedonly_t4",
                                                  "q3_q8_0_speedonly_t4", "q35_q4_0_real_t4"], speed),
        ("Thread count, no vision", ["q3_q4_0_embq8_t4", "th_q3_q40e8_t3", "th_q3_q40e8_t2", "p0_q25_q4km_t4", "th_q25_q4km_t3", "th_q25_q4km_t2",
                                     "x_q3_t4_passive_alone", "x_q3_t3_passive_alone"], speed),
        ("Binaries, Qwen3 Q4_0-embQ8, 3 threads, no vision", ["th_q3_q40e8_t3", "bin_noomp_t3", "bin_noomp_poll0_t3", "bin_official_b11960_t3", "bin_official_b11960_t3_again", "bin_omp_t3_again", "bin_generic_t3", "bin_nolto_t3", "bin_gcc14_t3", "bin_gcc14a53_t3", "x_q3_t3_nommap", "x_q3_t3_c640_ub128"], speed),
        ("Tandem, vision loaded and idle", ["tv_q3_t3_idle", "tv_q3_t4_idle", "tv_q25_t3_idle"],
         ["gen", "ttft_cold", "ttft_cached", "s100_cached", "rss", "vision_rss", "min_avail", "peak", "window"]),
        ("Tandem, vision reading at 2 Hz", ["tl_q3_t4_n0", "tl_q3_t4_n10", "tl_q3_t4_n19", "tl_q3_t4_n10_passive", "tl_q3_t3_n0", "tl_q3_t3_n10", "tl_q3_t3_n10_passive",
                                            "tl_q3_t3_n10_noomp_poll0", "tl_q3_t2_n0", "tl_q3_t2_n10", "tl_q25_t3_n0", "tl_q25_t3_n10", "tl_q25_t2_n10"],
         ["gen", "ttft_cold", "ttft_cached", "s100_cached_tpw150", "read_alone_p50", "read_idle_p50", "read_busy_p50", "read_busy_p95", "read_busy_max", "reads_late",
          "rss", "vision_rss", "min_avail", "peak", "window", "aborted"]),
        ("Final configuration on the chosen binary (GCC 14 static, nice 10, OMP_WAIT_POLICY=passive, repeat_penalty 1.1, system prompt prewarmed)",
         ["final2_t3_vision2hz_a", "final2_t3_vision2hz_b", "final2_t4_vision2hz", "final2_t3_alone", "final2_t3_cold_no_prewarm_vision_idle",
          "final2_fallback_q25_t3_vision2hz", "final2_fallback_q25_t3_alone"],
         ["gen", "ttft_cold", "ttft_cached", "s100_cached", "s100_cached_tpw150", "read_alone_p50", "read_idle_p50", "read_busy_p50", "read_busy_p95", "read_busy_max",
          "rss", "anon", "vision_rss", "min_avail", "peak", "load", "window", "aborted"]),
    ]
    for title, names, cols in sets:
        lines, rows = table(title, names, cols)
        out += lines
        nums[title] = rows
    heat = ["heat_q3_q40e8_t2_repeat", "heat_q3_q40e8_t3_repeat", "heat_q3_q40e8_t4_repeat", "heat_tandem_t3_n10_repeat", "heat_tandem_t4_n10_repeat", "heat_vision_only_baseline",
            "final2_heat_t3_vision2hz_repeat", "final2_heat_t4_vision2hz_repeat"]
    out += ["### Sustained heat (explanations back to back until the window closes)", "",
            "| run | start C | peak C | end C | window s | explanations | C at 30/60/90/120/150 s | aborted |", "|---|---|---|---|---|---|---|---|"]
    nums["heat"] = {}
    for n in heat:
        d = load(n)
        if not d:
            out.append(f"| {n} | not run | | | | | | |")
            continue
        sm = d["samples"]

        def at(t):
            near = min(sm, key=lambda x: abs(x["t"] - t))
            return near["hot_c"] if abs(near["t"] - t) < 3 else None
        curve = [at(t) for t in (30, 60, 90, 120, 150)]
        ab = (d.get("aborted") or {}).get("reason")
        nums["heat"][n] = {"start": d["start_hot_c"], "peak": d["peak_hot_c"], "end": d["end_hot_c"], "window": round(d["window_s"]), "n": len(d["requests"]), "curve": curve, "aborted": d.get("aborted")}
        out.append(f"| {n} | {d['start_hot_c']} | {d['peak_hot_c']} | {d['end_hot_c']} | {round(d['window_s'])} | {len(d['requests'])} | {curve} | {ab or ''} |")
    out.append("")
    idle = {}
    for n in ("cold_q3_q40e8_t3", "verbose_q3_q40e8_load_only", "tv_q3_t3_idle"):
        d = load(n)
        if d:
            idle[n] = {"load_s": round(d.get("load_s") or 0, 2), "cold": d["args"].get("cold"), "idle_cpu": d.get("idle_cpu"), "post_idle_cpu": d.get("post_idle_cpu")}
    nums["idle_and_load"] = idle
    all_runs = sorted(RUNS.glob("*.json"))
    n_llm = 0
    boots = set()
    for p in all_runs:
        d = json.loads(p.read_text())
        n_llm += 1
        for k in ("boot_id_before", "boot_id_after"):
            if d.get("driver", {}).get(k):
                boots.add(d["driver"][k])
    nums["windows"] = {"bench_windows": n_llm, "distinct_boot_ids_seen": len(boots),
                       "aborted": {json.loads(p.read_text())["name"]: json.loads(p.read_text()).get("aborted") for p in all_runs if json.loads(p.read_text()).get("aborted")},
                       "reboot_during_window": [json.loads(p.read_text())["name"] for p in all_runs
                                                if json.loads(p.read_text()).get("driver", {}).get("boot_id_before") != json.loads(p.read_text()).get("driver", {}).get("boot_id_after")]}
    loaded = sorted(((p.stat().st_mtime, json.loads(p.read_text())) for p in all_runs), key=lambda x: x[0])
    gaps = []
    for (_, a), (_, b) in zip(loaded, loaded[1:]):
        c = b.get("driver", {}).get("cooldown_before") or []
        if a.get("end_hot_c") and c:
            gaps.append({"after": a["name"], "end_c": a["end_hot_c"], "first_poll_c": c[0]["hot_c"], "polls_needed": len(c)})
    nums["cooldown"] = {"note": "first_poll_c is read after the driver's gap of 30 to 60 s outside the lock, or after a model push when one was needed",
                        "median_first_poll_c": median(g["first_poll_c"] for g in gaps), "max_first_poll_c": max(g["first_poll_c"] for g in gaps),
                        "median_end_c": median(g["end_c"] for g in gaps)}
    base = [d["baseline"] for _, d in loaded if d.get("baseline")]
    with_k = [b["mem_available_mb"] for b in base if any(x["comm"].startswith("chrom") for x in b["top_rss"])]
    no_k = [b["mem_available_mb"] for b in base if not any(x["comm"].startswith("chrom") for x in b["top_rss"])]
    apps = [x for b in base for x in b.get("twinread_app", [])]
    nums["memory_baseline"] = {"mem_total_mb": base[-1]["mem_total_mb"], "avail_with_kiosk_runs": with_k, "avail_no_kiosk_median": median(no_k), "n_no_kiosk": len(no_k),
                               "kiosk_cost_est_mb": median(no_k) - median(with_k) if with_k else None,
                               "app_rss_mb_range": [min(x["rss_mb"] for x in apps), max(x["rss_mb"] for x in apps)], "app_cpu_pct_max": max(x["cpu_pct"] for x in apps),
                               "app_cpu_pct_median": median(x["cpu_pct"] for x in apps), "system_busy_median_pct": median(b["system_busy_pct_of_4_cores"] for b in base),
                               "max_swap_used_mb_any_run": max((d.get("max_swap_used_mb") or 0) for _, d in loaded)}
    nums["hot_windows"] = {"longest_window_s": round(max((d.get("window_s") or 0) for _, d in loaded), 1),
                           "longest_lock_hold_s": max((d.get("driver", {}).get("lock_held_s") or 0) for _, d in loaded),
                           "hottest_c_any_run": max((d.get("peak_hot_c") or 0) for _, d in loaded),
                           "slowest_clock_mhz_under_load": min((d.get("min_mhz_seen_under_load") or 99999) for _, d in loaded),
                           "think_leaks": [d["name"] for _, d in loaded if any(q.get("reasoning_chars") or q.get("think_tag_in_text") for q in d.get("requests", []))],
                           "left_running_any_run": [d["name"] for _, d in loaded if d.get("left_running")]}
    (REPO / "results/llm/report_tables.md").write_text("\n".join(out))
    (REPO / "results/llm/report_numbers.json").write_text(json.dumps(nums, indent=1))
    print("\n".join(out))
    print(json.dumps(nums["windows"], indent=0), json.dumps(idle, indent=0))


if __name__ == "__main__":
    main()
