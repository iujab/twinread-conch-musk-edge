#!/usr/bin/env python3
"""Turn results/llm/runs/*.json into results/llm/summary.json and a readable results/llm/summary.md.

Every number the report quotes comes from these two files, and they contain only what the board measured. Rates and times
are taken from each run's requests: "cold" is the first request after loading (nothing cached), "cached" is a later request
that reused the shared system prompt.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "results/llm/runs"


def rnd(x, n=1):
    return None if x is None else round(x, n)


def row(d: dict) -> dict:
    reqs = [q for q in d.get("requests", []) if q.get("gen_tps")]
    cold = next((q for q in reqs if not q.get("cache_n")), None)
    cached = next((q for q in reqs if q.get("cache_n")), None)
    a = d.get("args", {})
    rl = d.get("vision_read_latency") or {}
    gen = [q["gen_tps"] for q in reqs]
    return {
        "run": d["name"], "model": Path(a.get("model", "?")).name, "threads": a.get("threads"), "nice": a.get("nice"),
        "bin": Path(a.get("bin", "?")).name, "vision": a.get("vision"), "env": a.get("env"), "server_args": a.get("server_args"),
        "load_s": rnd(d.get("load_s")), "cold_load": a.get("cold"),
        "pp_tps": rnd(median([q["prompt_tps"] for q in reqs if q.get("prompt_tps")])) if reqs else None,
        "gen_tps": rnd(median(gen), 2) if gen else None,
        "cold_prompt_n": cold and cold.get("prompt_n"), "cold_ttft_s": rnd(cold and cold.get("ttft_s")),
        "cold_100w_est_s": rnd(cold and cold.get("est_100_words_s")),
        "cached_prompt_n": cached and cached.get("prompt_n"), "cached_n": cached and cached.get("cache_n"),
        "cached_ttft_s": rnd(cached and cached.get("ttft_s")), "cached_100w_est_s": rnd(cached and cached.get("est_100_words_s")),
        "tokens_per_word": rnd(median([q["tokens_per_word"] for q in reqs if q.get("tokens_per_word")]), 2) if reqs else None,
        "peak_rss_mb": d.get("peak_server_rss_mb"), "peak_anon_mb": d.get("peak_server_anon_mb"),
        "peak_vision_rss_mb": d.get("peak_vision_rss_mb") or None, "min_avail_mb": d.get("min_avail_mb"),
        "max_swap_used_mb": d.get("max_swap_used_mb"), "peak_hot_c": d.get("peak_hot_c"), "start_hot_c": d.get("start_hot_c"),
        "window_s": rnd(d.get("window_s")), "idle_cpu_pct": (d.get("idle_cpu") or {}).get("server_cpu_pct_of_one_core"),
        "post_idle_cpu_pct": (d.get("post_idle_cpu") or {}).get("server_cpu_pct_of_one_core"),
        "baseline_busy_pct": (d.get("baseline") or {}).get("system_busy_pct_of_4_cores"),
        "baseline_avail_mb": (d.get("baseline") or {}).get("mem_available_mb"),
        "baseline_app_cpu_pct": sum(x["cpu_pct"] for x in (d.get("baseline") or {}).get("twinread_app", [])) or 0.0,
        "baseline_app_rss_mb": sum(x["rss_mb"] for x in (d.get("baseline") or {}).get("twinread_app", [])) or 0.0,
        "repack_seen": d.get("repack_seen"), "min_mhz_under_load": d.get("min_mhz_seen_under_load"),
        "aborted": d.get("aborted"), "error": d.get("error"), "reboot": (d.get("driver", {}).get("boot_id_before") != d.get("driver", {}).get("boot_id_after")),
        "read_alone_p50": (rl.get("alone_no_llm") or {}).get("p50_ms"), "read_alone_p95": (rl.get("alone_no_llm") or {}).get("p95_ms"),
        "read_idle_p50": (rl.get("llm_loaded_idle") or {}).get("p50_ms"), "read_idle_p95": (rl.get("llm_loaded_idle") or {}).get("p95_ms"),
        "read_busy_p50": (rl.get("llm_busy") or {}).get("p50_ms"), "read_busy_p95": (rl.get("llm_busy") or {}).get("p95_ms"),
        "read_busy_max": (rl.get("llm_busy") or {}).get("max_ms"), "read_busy_n": (rl.get("llm_busy") or {}).get("n"),
        "read_busy_late": (rl.get("llm_busy") or {}).get("late_over_100ms"),
        "think_leak": any(q.get("reasoning_chars") or q.get("think_tag_in_text") for q in reqs),
    }


COLS = ["run", "threads", "nice", "vision", "load_s", "pp_tps", "gen_tps", "cold_ttft_s", "cached_ttft_s", "cold_100w_est_s", "cached_100w_est_s",
        "peak_rss_mb", "peak_anon_mb", "min_avail_mb", "peak_hot_c", "window_s", "read_alone_p50", "read_idle_p50", "read_busy_p50", "read_busy_p95"]


def main() -> None:
    rows = []
    texts = {}
    for p in sorted(RUNS.glob("*.json")):
        d = json.loads(p.read_text())
        rows.append(row(d))
        texts[d["name"]] = [{"example": q.get("example"), "words": q.get("words"), "gen_n": q.get("gen_n"), "finish": q.get("finish_reason"),
                             "text": q.get("text")} for q in d.get("requests", [])]
    (REPO / "results/llm/summary.json").write_text(json.dumps({"rows": rows, "outputs": texts}, indent=1))
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in COLS) + " |")
    (REPO / "results/llm/summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(re.sub(r"\s+\|", " |", ln) for ln in lines))


if __name__ == "__main__":
    main()
