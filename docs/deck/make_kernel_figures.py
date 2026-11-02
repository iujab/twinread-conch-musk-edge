"""Graphs for the dependent-kernel slide. Reads results/kernels/*.json and writes docs/deck/assets/kernel_*.png plus
kernel_figures.json (the derived numbers the slide text uses, so figure and slide cannot disagree).

    ../../../.venv/bin/python docs/deck/make_kernel_figures.py        # from anywhere; paths resolve from this file

Times are measured (CUDA events). DRAM traffic and memory floors are calculated from tensor sizes and the measured
memcpy rate, because Nsight hardware counters are admin-only on this machine.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "assets"
INK, MUTED, GRID, BASE, OURS = "#1F1A2E", "#625E70", "#E8E6EE", "#8C8898", "#5C2FB0"
SIZE = (3.89, 3.45)  # inches, the slot on the slide, so font sizes below are true slide point sizes
STAGES = ["warp", "HED stain", "gamma", "colour", "moiré", "glare", "defocus", "white bal.", "noise", "JPEG"]
N_FUSED_LAUNCH_1 = 9  # k2_fused: warp through noise. k2_jpeg is launch 2 (csrc/k2_camera_aug.cu).

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": INK, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})


def traffic(k2):
    """Bytes through DRAM for image data only: per-stage cumulative, one launch per stage vs the fused kernel."""
    b, (h, w) = k2["batch"], k2["hw"]
    u8 = b * h * w * 3
    f32 = u8 * 4
    n = len(STAGES)
    # One launch per stage: read the previous result and write this one. Input and output are uint8, the rest float32.
    per_stage = [u8 + f32] + [2 * f32] * (n - 2) + [f32 + u8]
    # Fused: launch 1 reads the input at its first stage and writes float32 at its last; launch 2 reads that, writes uint8.
    fused = [u8] + [0] * (N_FUSED_LAUNCH_1 - 2) + [f32] + [f32 + u8]
    cum = lambda xs: [sum(xs[: i + 1]) for i in range(len(xs))]  # noqa: E731
    return cum(per_stage), cum(fused)


def fig_traffic(unfused, fused):
    fig, ax = plt.subplots(figsize=SIZE)
    x = list(range(len(STAGES)))
    gb = lambda v: [a / 1e9 for a in v]  # noqa: E731
    ax.plot(x, gb(unfused), color=BASE, lw=2, marker="o", ms=4.5)
    ax.plot(x, gb(fused), color=OURS, lw=2.8, marker="o", ms=5)
    ax.set_xticks(x, STAGES, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Cumulative DRAM traffic (GB)")
    ax.set_ylim(0, 3.2)
    ax.set_xlim(-0.4, len(STAGES) - 0.6)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(-0.15, 2.55, f"One launch per stage\n{unfused[-1] / 1e9:.2f} GB", ha="left", va="center", fontsize=11, color=INK)
    ax.text(8.9, 0.74, f"Fused, 2 launches\n{fused[-1] / 1e9:.2f} GB", ha="right", va="center", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    fig.text(0.012, 0.965, "K2, batch 256, image data only, calculated", fontsize=11, color=MUTED, va="center")
    fig.savefig(OUT / "kernel_traffic.png")
    plt.close(fig)


def dots(path, title, rows, line_x, line_label, xmax):
    """Dot plot on a log axis: bars would imply a meaningful baseline, which a log scale does not have."""
    fig, ax = plt.subplots(figsize=SIZE)
    y = list(range(len(rows)))[::-1]
    for yi, (_, v, note, ours, left) in zip(y, rows):
        ax.plot([1, v], [yi, yi], color=OURS if ours else GRID, lw=1.6 if ours else 1.2, zorder=1)
        ax.plot(v, yi, "o", ms=9 if ours else 7.5, color=OURS if ours else BASE, zorder=3)
        # "left" here means the value sits above its dot, clear of the stem and of the reference line just to the right.
        xt, yt, ha = (v, yi + 0.42, "center") if left else (v * 1.32, yi, "left")
        ax.text(xt, yt, note, va="center", ha=ha, fontsize=10.5, fontweight="bold" if ours else "normal", color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(1, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.1 + 0.7)
    ax.set_xticks([t for t in (1, 10, 100, 1000, 10000) if t < xmax], [f"{t:,}" for t in (1, 10, 100, 1000, 10000) if t < xmax])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Time per batch (ms, log scale)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.axvline(line_x, color=INK, lw=1.2, ls=(0, (4, 3)), ymax=0.9, zorder=2)
    ax.text(line_x * 1.14, len(rows) - 0.28, line_label, va="center", ha="left", fontsize=10, color=INK)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    fig.text(0.012, 0.965, title, fontsize=11, color=MUTED, va="center")
    fig.savefig(path)
    plt.close(fig)


def main():
    k1b = json.loads((ROOT / "results/kernels/k1_bench.json").read_text())
    k2b = json.loads((ROOT / "results/kernels/k2_bench.json").read_text())
    k1p = json.loads((ROOT / "results/kernels/k1_parity.json").read_text())
    k1, k2 = k1b["gpu"]["tiles224"], k2b["gpu"]
    sev = k2["severity_1.0"]
    bytes_per_s = k1b["gpu"]["memcpy_d2d"]["GBps_read_plus_write"] * 1e9

    unfused, fused = traffic(k2)
    d = {
        "unfused_gb": unfused[-1] / 1e9, "fused_gb": fused[-1] / 1e9, "traffic_ratio": unfused[-1] / fused[-1],
        "unfused_traffic_ms": unfused[-1] / bytes_per_s * 1e3, "fused_traffic_ms": fused[-1] / bytes_per_s * 1e3,
        "k2_ms": sev["d_cuda_k2"]["ms"], "k2_launch1_ms": sev["d_cuda_k2"]["kernels_ms"]["k2_fused"],
        "k2_launch2_ms": sev["d_cuda_k2"]["kernels_ms"]["k2_jpeg"], "torch_ms": sev["c_torch_gpu"]["ms"],
        "torch_randn_ms": sev["c2_torch_gpu_randn_noise"]["ms"], "x_torch": sev["speedup_cuda_vs_torch"],
        "x_torch_randn": sev["speedup_cuda_vs_torch_randn"], "memcpy_gbps": k1b["gpu"]["memcpy_d2d"]["GBps_read_plus_write"],
        "k1_ms": k1["cuda_k1"]["ms"], "k1_floor_ms": k1["two_pass_dram_floor_ms"],
        "k1_over_floor": k1["cuda_k1"]["ms"] / k1["two_pass_dram_floor_ms"], "k1_triton_ms": k1["triton"]["ms"],
        "k1_interpolate_ms": k1["torch_interpolate"]["ms"],
        "interpolate_max_levels": max(k1["torch_interpolate"]["accuracy_first16"][r]["max_levels"] for r in ("conch", "musk")),
        "k1_bit_exact_min": min(k1p["result"]["sets"]["crc_tiles256"]["cuda"][r]["frac_bit_exact_vs_official_half"] for r in ("conch", "musk")),
    }
    assert round(fused[-1] / 1e6) == 385, "k2_bench.json bytes_note says 385 MB at minimum"
    assert abs(d["k1_over_floor"] - k1b["summary"]["tiles224"]["cuda_k1_ms_vs_two_pass_floor"]) < 1e-9
    assert d["k1_bit_exact_min"] == 1.0

    fig_traffic(unfused, fused)
    n = k2["batch"]
    ms1, ms4 = (n / k2b["cpu"][k]["images_per_s"] * 1e3 for k in ("a_numpy_single_process", "b_numpy_pool_4proc"))
    dots(OUT / "kernel_k2_time.png", "K2, batch 256, severity 1.0, measured", [
        ("numpy,\n1 process", ms1, f"{ms1:,.0f}", False, False),
        ("numpy,\n4 processes", ms4, f"{ms4:,.0f}", False, False),
        ("torch GPU ops", d["torch_ms"], f"{d['torch_ms']:.0f}  (×{d['x_torch']:.0f} slower)", False, False),
        ("torch GPU,\ncheaper noise", d["torch_randn_ms"], f"{d['torch_randn_ms']:.0f}  (×{d['x_torch_randn']:.0f} slower)", False, False),
        ("K2 fused", d["k2_ms"], f"{d['k2_ms']:.1f}", True, True),
    ], d["unfused_traffic_ms"], f"Unfused traffic alone: {d['unfused_traffic_ms']:.1f} ms", 100000)
    pil = k1b["cpu"]["tiles224"]["pil_single_process"]["seconds_median3"] * 1e3
    dots(OUT / "kernel_k1_time.png", "K1, batch 256, both readers, measured", [
        ("PIL, 1 process", pil, f"{pil:,.0f}", False, False),
        ("torch GPU\n(exact)", k1["torch_exact"]["ms"], f"{k1['torch_exact']['ms']:.0f}", False, False),
        ("torch interpolate\n(not bit-exact)", d["k1_interpolate_ms"], f"{d['k1_interpolate_ms']:.0f}", False, False),
        ("Triton,\nsame plan", d["k1_triton_ms"], f"{d['k1_triton_ms']:.2f}", False, False),
        ("K1 CUDA", d["k1_ms"], f"{d['k1_ms']:.2f}", True, False),
    ], d["k1_floor_ms"], f"Memory floor: {d['k1_floor_ms']:.2f} ms", 20000)
    (OUT / "kernel_figures.json").write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps({k: round(v, 3) for k, v in d.items()}))


if __name__ == "__main__":
    main()
