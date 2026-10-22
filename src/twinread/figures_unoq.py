"""Slide-ready figures for the on-board Arduino UNO Q run, from results/uno_q/onboard/*.json only.

    PYTHONPATH=src $BENCH/.venv/bin/python -m twinread.figures_unoq            # writes figures/uno_q/
    PYTHONPATH=src $BENCH/.venv/bin/python -m twinread.figures_unoq --only fig10

Sources: latency.json (device/bench_backbones.py run on the board), stage_timing_cm.json, testset_score.json,
int8_kernel_check.json and board_info.json (scripts/uno_q/), plus results/student/model_sizes.json. Style and palette
come from twinread.figures. Colour is emphasis only: the UNO Q run / CM student is violet, comparators (x86, the OPEN
student) are grey, and every mark carries a direct label. Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from twinread import figures as F

SRC = F.RES / "uno_q" / "onboard"
OUT = F.FIG / "uno_q"
C = F.C
VIO, GREY, DARK = C["student"], C["grey"], C["shared"]


def _j(name: str) -> dict:
    return json.loads((SRC / name).read_text())


def _save(fig, name) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=F.DPI)
    plt.close(fig)
    return p


def _lat(rows, backbone, threads) -> dict:
    return next(r for r in rows if r["backbone"] == backbone and r["threads"] == threads and "p50_ms" in r)


# ---------------------------------------------------------------------------------------------------------------
def fig9_unoq_system() -> Path:
    info, lat, st, sc = _j("board_info.json"), _j("latency.json"), _j("stage_timing_cm.json"), _j("testset_score.json")
    size = json.loads((F.RES / "student" / "model_sizes.json").read_text())["students"]["CM"]
    ms = {s["stage"]: s["p50_ms"] for s in st["stages"]}
    cm1, cm4, cm = _lat(lat["rows"], "student_cm", 1), _lat(lat["rows"], "student_cm", 4), sc["pairs"]["CM"]
    soc = next(c.split(",")[1].upper() for c in info["compatible"] if c.startswith("qcom,qrb"))

    fig = plt.figure(figsize=(F.W, 7.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=None, fs=9.4, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc,
                                    ec=ec or "#d5d4ce", lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["ink"], linespacing=1.25,
                fontweight=weight)

    def ms_label(x, y, text, bold=False):
        ax.text(x, y, text, ha="center", va="top", fontsize=9, color=C["ink"] if bold else C["ink2"],
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=C["ink2"], lw=1.3))

    def lane(y, h, title, status, col):
        ax.add_patch(FancyBboxPatch((1, y), 98, h, boxstyle="round,pad=0,rounding_size=1.5", fc=col, ec="none"))
        ax.text(2.2, y + h - 1.2, title, fontsize=10.5, fontweight="bold", va="top", color=C["ink"])
        ax.text(97.8, y + h - 1.2, status, fontsize=9, va="top", ha="right", color=C["ink2"], style="italic")

    ax.text(1.2, 69.3, f"TwinRead runs offline on the Arduino UNO Q: {cm4['p50_ms']:.1f} ms per tile, "
            f"{cm['int8_bal_acc_board']:.4f} balanced accuracy", ha="left", va="top", fontsize=14.5,
            fontweight="bold", color=C["ink"])

    # hero numbers, all measured on the board
    tiles = [(f"{cm4['p50_ms']:.1f} ms", f"int8 student, one 224-px tile\n4 threads, p50 (1 thread: {cm1['p50_ms']:.1f} ms)"),
             (f"{cm['int8_bal_acc_board']:.4f}", f"balanced accuracy, CM int8\nall {cm['board_run']['n']:,} "
                                                 "CRC-VAL-HE-7K tiles"),
             (f"{cm['board_run']['tiles_per_s']:.1f} tiles/s", f"full runtime incl. tissue check\n"
                                                             f"{cm['board_run']['n']:,} tiles in "
                                                             f"{cm['board_run']['seconds']:.0f} s"),
             (f"{size['int8_onnx_bytes'] / 1e6:.1f} MB", f"int8 ONNX, {size['params_total'] / 1e6:.2f} M params\n"
                                                         f"+{cm4['rss_mb']:.0f} MB RAM while running, offline")]
    for i, (num, lab) in enumerate(tiles):
        x = 1.5 + i * 24.5
        ax.add_patch(FancyBboxPatch((x, 52.5), 23, 10.5, boxstyle="round,pad=0,rounding_size=1.2", fc=C["note"],
                                    ec="none"))
        ax.add_patch(FancyBboxPatch((x, 52.5), 0.7, 10.5, boxstyle="square,pad=0", fc=VIO, ec="none"))
        ax.text(x + 2.2, 61.6, num, fontsize=19, fontweight="bold", va="top", color=C["ink"])
        ax.text(x + 2.2, 55.6, lab, fontsize=9, va="center", color=C["ink2"], linespacing=1.3)

    # Linux side
    lane(21.5, 29, f"Linux side · Qualcomm {soc}, {info['cpu_count']}× Cortex-A53 @ {info['cpu_max_khz'] / 1e6:.1f} GHz, "
                   f"{info['mem_total_mib'] / 1024:.1f} GiB · Debian 13 · onnxruntime {info['onnxruntime']} (CPU)",
         "", "#f0f8f2")
    ax.text(88.8, 24.6, f"one app frame = {st['total_p50_ms']:.0f} ms (p50)\nincl. "
            f"{ms['tissue fraction again (UI)'] + ms['UI frame (box + JPEG)']:.1f} ms of UI work", ha="right",
            va="center", fontsize=9, color=C["ink2"], style="italic", linespacing=1.3)
    box(2.5, 30, 10.5, 10, "frame\n(replay PNG;\nUSB camera\nin the demo)", fs=9)
    ms_label(7.75, 29.2, f"{ms['frame read (PNG decode)']:.1f} ms")
    box(16, 30, 12, 10, "crop to 224 px\n+ tissue check\n(numpy)", fs=9)
    ms_label(22, 29.2, f"{ms['crop to 224 px'] + ms['tissue check']:.1f} ms")
    arrow(13, 35, 16, 35)
    # the student: trunk + four heads
    ax.add_patch(FancyBboxPatch((31, 24.5), 33, 21.5, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#ece9fb",
                                ec=VIO, lw=1.6))
    ax.text(47.5, 45.2, f"int8 student · ONNX QDQ · {size['int8_onnx_bytes'] / 1e6:.1f} MB", ha="center", va="top",
            fontsize=10, fontweight="bold", color=C["ink"])
    ax.text(47.5, 42.6, "runs as 46 QLinearConv · 10 QLinearAdd · 6 QGemm", ha="center", va="top",
            fontsize=8.6, color=C["ink2"])
    box(32.5, 28, 13, 10.5, f"MobileNetV4-\nconv-small trunk\n{size['params_trunk'] / 1e6:.2f} M params\n"
                            "input 3×224×224", fs=8.8)
    heads = ["hA 512-d · CONCH space", "hB 1024-d · MUSK space", "fused logits · 9 classes", "review logit"]
    for k, h in enumerate(heads):
        y = 36.3 - k * 2.85
        box(49, y, 13.5, 2.1, h, fs=8.3)
        arrow(45.5, 33.25, 49, y + 1.05)
    ax.text(55.75, 27.3, f"heads {size['params_heads'] / 1e6:.2f} M params", ha="center", va="top", fontsize=8.3,
            color=C["ink2"])
    ms_label(47.5, 24.0, f"{ms['int8 student (onnxruntime)']:.1f} ms (4 threads) + {ms['uint8 -> float NCHW']:.1f} ms "
                         "input prep", bold=True)
    arrow(28, 35, 31, 35)
    box(67, 30, 12.5, 10, "linear probes\npA, pB, pF\n+ device\nreview rule", fs=9)
    ms_label(73.25, 29.2, f"{ms['probes + softmax'] + ms['review rule']:.1f} ms")
    arrow(64, 35, 67, 35)
    box(83, 31.5, 14.5, 7, "AGREE / REVIEW\n/ NO TISSUE", fc="#d6f0e5", ec=C["F"], fs=9.2, weight="bold")
    box(83, 41.5, 14.5, 4.8, "web UI · 127.0.0.1:8080", fs=9)
    arrow(79.5, 35, 83, 35); arrow(90.25, 38.5, 90.25, 41.5)

    # MCU side
    lane(2.5, 16.5, "MCU side · STM32U585 (Arm Cortex-M33, Zephyr)",
         "sketch flashed over USB-C by OpenOCD on the Linux side", "#eef4fb")
    ax.text(2.5, 11.2, "Bridge verified on hardware:\nset_state() delivered; an unknown\nmethod is refused "
                       "(negative control)", fontsize=9, va="center", color=C["ink2"], linespacing=1.35)
    box(36, 6.5, 22, 8, "arduino-router\nmsgpack RPC over a unix socket", fs=9)
    led = ms["LED busy (Bridge RPC)"] + ms["LED result (Bridge RPC)"]
    ms_label(47, 6.0, f"{led:.1f} ms per frame (2 calls)")
    box(62, 6.5, 15, 8, "twinread_led\nsketch\nset_state()", fs=9)
    box(81, 6.5, 16.5, 8, "8×13 LED matrix\n✓  ?  –  icons", fs=9)
    arrow(58, 10.5, 62, 10.5); arrow(77, 10.5, 81, 10.5)
    # status to the MCU: an elbow through the gap between the lanes
    ax.plot([90.25, 90.25, 47, 47], [31.5, 20.25, 20.25, 16.5], color=C["ink2"], lw=1.3, solid_joinstyle="round")
    arrow(47, 16.6, 47, 14.6)

    ax.text(50, 0.4, "All numbers measured on the board (scripts/uno_q/, device/bench_backbones.py); CM = the student "
                     "distilled from CONCH + MUSK. " + F.DISCLAIMER, ha="center", va="bottom", fontsize=8.8,
            color=C["ink2"])
    return _save(fig, "fig9_unoq_system.png")


# ---------------------------------------------------------------------------------------------------------------
def fig10_unoq_speed() -> Path:
    lat, st = _j("latency.json"), _j("stage_timing_cm.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(F.W, 5.1), gridspec_kw={"width_ratios": [1, 1.45]}, layout="constrained")
    F._layout(fig, 0.17, 0.88)
    th = [1, 2, 4]
    cm = [_lat(lat["rows"], "student_cm", t)["p50_ms"] for t in th]
    op = [_lat(lat["rows"], "student_open", t)["p50_ms"] for t in th]
    x = np.arange(len(th))
    for vals, dx, col in ((cm, -0.19, VIO), (op, 0.19, GREY)):
        a.bar(x + dx, vals, 0.36, color=col, edgecolor="white", lw=1.5)
        for xi, v in zip(x + dx, vals):
            a.text(xi, v + 0.9, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=C["ink"])
    a.set_xticks(x, [f"{t} thread{'s' if t > 1 else ''}" for t in th])
    a.set_ylabel("ms per tile (p50)")
    a.set_ylim(0, max(cm + op) * 1.18)
    F._grid(a)
    a.legend(handles=[Line2D([], [], marker="s", ls="", ms=10, color=VIO, label="CM student (CONCH + MUSK)"),
                      Line2D([], [], marker="s", ls="", ms=10, color=GREY, label="OPEN student")],
             loc="upper right", fontsize=9.4)
    a.set_title(f"(a) int8 student, one tile: {cm[0]:.1f} → {cm[-1]:.1f} ms\n({cm[0] / cm[-1]:.1f}× with 4 threads)",
                loc="left", fontsize=11.5, fontweight="bold")

    ms = {s["stage"]: s["p50_ms"] for s in st["stages"]}
    total = st["total_p50_ms"]
    big = {"tissue check": "tissue check (inside infer)", "tissue fraction again (UI)": "tissue check again (for UI)",
           "int8 student (onnxruntime)": "int8 student (onnxruntime)", "frame read (PNG decode)": "frame read (PNG)"}
    rows = [(lab, ms[k]) for k, lab in big.items()]
    rows.append(("LED via Bridge RPC (2 calls)", ms["LED busy (Bridge RPC)"] + ms["LED result (Bridge RPC)"]))
    other = [k for k in ms if k not in big and not k.startswith("LED")]
    rows.append(("other: crop, float, probes,\nreview rule, JPEG", sum(ms[k] for k in other)))
    rows.sort(key=lambda r: -r[1])
    col = lambda lab: VIO if lab.startswith("int8") else (DARK if lab.startswith("tissue") else GREY)
    y = np.arange(len(rows))[::-1]
    b.barh(y, [v for _, v in rows], 0.62, color=[col(lab) for lab, _ in rows], edgecolor="white", lw=1.5)
    for yy, (_, v) in zip(y, rows):
        b.text(v + 0.4, yy, f"{v:.1f} ms  ({100 * v / total:.0f}%)", va="center", fontsize=9.5, color=C["ink"])
    b.set_yticks(y, [lab for lab, _ in rows], fontsize=9.6)
    b.set_xlim(0, max(v for _, v in rows) * 1.45)
    b.set_xlabel("ms per app frame (p50 of 300 frames, CM bundle)")
    F._grid(b, "x")
    model = ms["int8 student (onnxruntime)"]
    tissue = ms["tissue check"] + ms["tissue fraction again (UI)"]
    b.set_title(f"(b) one app frame = {total:.0f} ms: the CNN is {100 * model / total:.0f}%,\n"
                f"the numpy tissue check, run twice, {100 * tissue / total:.0f}%", loc="left", fontsize=11.5,
                fontweight="bold")
    F._title(fig, "On the UNO Q the network is not the bottleneck: "
                  f"{cm[-1]:.1f} ms per tile, {total:.0f} ms per app frame")
    spread = max(r["p90_ms"] - r["p50_ms"] for r in lat["rows"] if "p50_ms" in r)
    F._foot(fig, ["Measured on the board: Arduino UNO Q, 4× Cortex-A53 @ 2.0 GHz, onnxruntime 1.31 CPU, int8 QDQ bundles. (a) device/bench_backbones.py: batch 1,",
                  f"50 runs, each setting in a fresh process; p90 is within {spread:.1f} ms of p50. (b) scripts/uno_q/stage_timing.py calls the app's own functions",
                  "in App.step order (checked against EdgeModel.infer). Replay frames are 224-px PNGs, so a camera adds capture and a real resize.",
                  "The app loop runs one frame per second. " + F.DISCLAIMER])
    return _save(fig, "fig10_unoq_speed.png")


# ---------------------------------------------------------------------------------------------------------------
def fig11_unoq_fidelity() -> Path:
    k, sc = _j("int8_kernel_check.json"), _j("testset_score.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(F.W, 5.3), gridspec_kw={"width_ratios": [1.2, 1]}, layout="constrained")
    F._layout(fig, 0.19, 0.89)
    kinds = [("fp32 student", "fp32", "o", DARK, DARK),
             ("int8, x86 default kernels (committed)", "x86", "D", GREY, GREY),
             ("int8, x86 fused kernels off (reference)", "x86_noopt", "o", "white", VIO),
             ("int8, Arduino UNO Q", "board", "o", VIO, VIO)]
    labels, yy = [], 0
    for pair in ("cm", "open"):
        for name, key, mk, fc, ec in kinds:
            v = k[pair]["fp32_student_bal_acc"] if key == "fp32" else k[pair]["bal_acc"][key]
            a.axhline(-yy, color=C["grid"], lw=0.8, zorder=0)
            a.plot(v, -yy, marker=mk, ms=9 if mk == "o" else 8, mfc=fc, mec=ec, mew=1.8, ls="", zorder=3)
            a.text(v + 0.0016, -yy, f"{v:.4f}", va="center", fontsize=9.3, color=C["ink"],
                   fontweight="bold" if key == "board" else "normal")
            labels.append((-yy, f"{pair.upper()} · {name}"))
            yy += 1
        yy += 0.6
    a.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=9.4)
    a.set_xlim(0.895, 0.952)
    a.set_xlabel("balanced accuracy, CRC-VAL-HE-7K (7,180 tiles)")
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    a.set_title("(a) the board matches the int8 reference", loc="left", fontsize=11.5, fontweight="bold")

    rows = []
    for pair in ("cm", "open"):
        rows.append((f"{pair.upper()} · x86 default", k[pair]["x86_vs_x86_noopt"]["pred_mismatches"], GREY))
        rows.append((f"{pair.upper()} · UNO Q", k[pair]["board_vs_x86_noopt"]["pred_mismatches"], VIO))
    y = np.array([4.0, 3.0, 1.4, 0.4])
    n = sc["pairs"]["CM"]["board_run"]["n"]
    b.barh(y, [r[1] for r in rows], 0.72, color=[r[2] for r in rows], edgecolor="white", lw=1.5)
    for yy, (_, v, _) in zip(y, rows):
        b.text(v + 2, yy, f"{v} tiles ({100 * v / n:.1f}%)", va="center", fontsize=9.5, color=C["ink"])
    b.set_yticks(y, [r[0] for r in rows], fontsize=9.6)
    b.set_xlim(0, max(r[1] for r in rows) * 1.5)
    b.set_xlabel(f"tiles (of {n:,})")
    F._grid(b, "x")
    b.set_title("(b) tiles whose call differs\nfrom the int8 reference", loc="left", fontsize=11.5, fontweight="bold")

    F._title(fig, "The UNO Q runs the int8 student as specified; the x86 int8 scores were a kernel artefact")
    kap = {p: (sc["pairs"][p]["board_device_flag"]["kappa"], sc["pairs"][p]["committed_x86_device_flag"]["kappa"])
           for p in ("CM", "OPEN")}
    F._foot(fig, [f"Same model and input bytes on both machines (SHA-256). x86 = {k['host']['cpu']} (AVX2, no VNNI), "
                  f"onnxruntime {k['host']['onnxruntime']} with its default fused int8 kernels.",
                  "Reference = the same x86 with graph optimisations off, so QDQ runs as dequantise → fp32 op → quantise. Source: scripts/uno_q/.",
                  f"Review flag vs the teachers' flag (kappa): CM {kap['CM'][0]:.2f} on the board vs {kap['CM'][1]:.2f} on x86, OPEN {kap['OPEN'][0]:.2f} vs "
                  f"{kap['OPEN'][1]:.2f}; the bundle thresholds were calibrated on x86-kernel outputs.",
                  F.DISCLAIMER])
    return _save(fig, "fig11_unoq_fidelity.png")


FIGS = {"fig9": fig9_unoq_system, "fig10": fig10_unoq_speed, "fig11": fig11_unoq_fidelity}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help=f"subset of {list(FIGS)}")
    a = ap.parse_args()
    for key, f in FIGS.items():
        if not a.only or key in a.only:
            print(f())
