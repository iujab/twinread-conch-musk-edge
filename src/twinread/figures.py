"""Slide-ready figures from results JSON (and, for the tile gallery only, the cached test tiles in data/).

    PYTHONPATH=src .venv/bin/python -m twinread.figures            # all figures that can be built
    PYTHONPATH=src .venv/bin/python -m twinread.figures --only fig1

Every number drawn comes from results/headline.json (twinread.headline), results/**/triage_curves.json
(twinread.tilepreds) or the per-pair blindspots.csv. Figures are 1495 px wide at 130 dpi. Colours: the dataviz
reference categorical order (blue, orange, aqua first; CVD-validated), one colour per entity in every figure:
reader A = blue, reader B = orange, fused = aqua, student/K2 = violet. Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from twinread import headline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
RES = ROOT / "results"
DPI, W = 130, 11.5  # 1495 px
C = {"A": "#2a78d6", "B": "#eb6834", "F": "#1baf7a", "s4": "#6250d6", "s1": "#eda100", "s2": "#e87ba4",
     "student": "#6250d6", "grey": "#8a8984", "light": "#c9c8c2", "ink": "#0b0b0b", "ink2": "#52514e",
     "shared": "#3d3c39", "grid": "#e4e3df", "note": "#f6f5f2"}
CRC = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
DISCLAIMER = "Research prototype, not a diagnostic device."
NICE = {"conch": "CONCH", "musk": "MUSK", "pathgenclip": "PathGen-CLIP", "quiltnet_b16": "QuiltNet-B-16"}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "axes.edgecolor": "#9a9993",
                     "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False})


def _hl() -> dict:
    p = RES / "headline.json"
    return json.loads(p.read_text()) if p.exists() else headline.build(RES)


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=C["grid"], lw=0.8)
    ax.set_axisbelow(True)


def _layout(fig, bottom, top=0.93):
    fig.get_layout_engine().set(rect=(0.005, bottom, 0.99, top - bottom))


def _title(fig, text, y=0.985):
    fig.text(0.008, y, text, ha="left", va="top", fontsize=14.5, fontweight="bold", color=C["ink"])


def _foot(fig, lines, y=0.008):
    fig.text(0.008, y, "\n".join(lines), ha="left", va="bottom", fontsize=9, color=C["ink2"], linespacing=1.35)


def _save(fig, name) -> Path:
    FIG.mkdir(exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def _pts(x):
    return f"{100 * x:+.2f}"


# ---------------------------------------------------------------------------------------------------------------
def fig1_two_readers() -> Path:
    hl = _hl()["fusion"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 5.6), sharey=True, layout="constrained")
    _layout(fig, 0.1, 0.9)
    panels = (("CM_crc", "CONCH + MUSK\nCRC-VAL-HE-7K"), ("CM_cam", "CONCH + MUSK\nCamelyon17, hospital 2"),
              ("OPEN_crc", "Open pair (control)\nCRC-VAL-HE-7K"), ("OPEN_cam", "Open pair (control)\nCamelyon17, hospital 2"))
    for ax, (key, title) in zip(axes, panels):
        d = hl[key]
        rows = [("A", NICE[d["readers"]["A"]]), ("B", NICE[d["readers"]["B"]]),
                ("fused", f"fused\n({d['primary_fusion'].replace('_', ' ')})")]
        for i, (k, _) in enumerate(rows):
            v, (lo, hi) = d["bal_acc"][k], d["bal_acc_ci"][k]
            col = C["F"] if k == "fused" else C[k]
            ax.plot([i, i], [lo, hi], color=col, lw=2.4, solid_capstyle="round")
            ax.plot(i, v, "o", ms=10, color=col, mec="white", mew=2, zorder=3)
            ax.text(i + 0.14, v, f"{v:.4f}", va="center", fontsize=9.5, color=C["ink"])
        ax.set_xticks(range(3), [r[1].replace("PathGen-CLIP", "PathGen-\nCLIP").replace("QuiltNet-B-16",
                                 "QuiltNet-\nB-16") for r in rows], fontsize=9.6)
        ax.set_xlim(-0.45, 2.75)
        ax.set_ylim(0.885, 0.995)
        _grid(ax)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
        best = NICE[d["readers"][d["best_single_reader"]]]
        ci = d["delta_ci"]
        mode = "tile-level CI, anti-conservative" if "tile" in d["ci_mode"] else "slide-clustered CI"
        ax.set_xlabel(f"fused − {best}: {_pts(d['delta_vs_best_single'])} pts\n95% CI [{_pts(ci[0])}, "
                      f"{_pts(ci[1])}]\n({mode})", fontsize=9.8, linespacing=1.3)
    axes[0].set_ylabel("test balanced accuracy, 95% CI")
    _title(fig, f"Fusing CONCH and MUSK: {_pts(hl['CM_crc']['delta_vs_best_single'])} points on CRC; no gain on "
                "Camelyon17 or for the open pair")
    _foot(fig, [
        "Linear probes on frozen features. The fusion rule was chosen on validation and pre-registered before test. "
        "CRC: train on a 20k-tile NCT-CRC-HE-100K subset, test on",
        "CRC-VAL-HE-7K (7,180 tiles, no slide ids). Camelyon17-WILDS: test on unseen hospital 2 (1,500 tiles). "
        "LOCAL-SUBSET dev numbers; full-data Delta runs are pending.",
        "The CRC gain is a MUS↔STR recall trade, and validation picked the fusion rule by 1-2 tiles. " + DISCLAIMER])
    return _save(fig, "fig1_two_readers.png")


# ---------------------------------------------------------------------------------------------------------------
def _gallery_items(n=8):
    tp = RES / "cm_local/CM/crc/tile_preds.npz"
    x = ROOT / "data/crc_test_x.npy"
    if not (tp.exists() and x.exists()):
        return None
    t = np.load(tp)
    idx = json.loads((RES / "cm_local/CM/crc/review.json").read_text())["confident_shared_errors"]["indices"]
    conf = {i: min(t["pA"][i].max(), t["pB"][i].max()) for i in idx}
    picked, seen = [], {}
    for i in sorted(idx, key=lambda i: -conf[i]):  # most confident first, at most 3 per (truth, call) pair
        key = (int(t["y"][i]), int(t["pA"][i].argmax()))
        if seen.get(key, 0) < 3:
            picked.append(i)
            seen[key] = seen.get(key, 0) + 1
        if len(picked) == n:
            break
    xs = np.load(x, mmap_mode="r")
    return [(i, np.asarray(xs[i]), CRC[t["y"][i]], CRC[t["pA"][i].argmax()], float(t["pA"][i].max()),
             float(t["pB"][i].max()), int(t["flag"][i])) for i in picked]


TRIAGE = (("s1_disagree", "s1 readers disagree", C["s1"], "-", 1.7),
          ("s2_js", "s2 disagreement (JS divergence)", C["s2"], "-", 1.7),
          ("s3_fused_unc", "s3 fused reader's uncertainty", C["F"], "-", 2.6),
          ("s4", "s4 learned s1+s2+s3 (pre-registered flag)", C["s4"], "-", 2.0),
          ("base_A_own_error", "CONCH alone, own confidence → own errors", C["A"], "--", 1.7),
          ("base_B_own_error", "MUSK alone, own confidence → own errors", C["B"], "--", 1.7))


def _triage_panel(ax, d, r, title, note_xy):
    xf = np.array(d["review_fraction"])
    m = xf <= 0.30001
    for k, _, col, ls, lw in TRIAGE:
        ax.plot(100 * xf[m], np.array(d["curves"][k]["errors_caught"])[m], ls, color=col, lw=lw)
    ax.axvline(10, color=C["grey"], lw=1, ls=":")
    op = d["prereg_operating_point_10"]
    pt = (100 * op["flagged_fraction"], op["errors_caught"])
    ax.plot(*pt, "D", ms=9, color=C["s4"], mec="white", mew=1.5, zorder=5)
    ax.annotate(f"◆ pre-registered threshold\nflags {pt[0]:.1f}% of tiles (target 10%)\n"
                f"and catches {100 * pt[1]:.1f}% of errors", pt, xytext=note_xy, ha="right", va="bottom",
                fontsize=8.8, color=C["ink"], arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.8))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("tiles sent to review (%)")
    _grid(ax, "both")
    ax.set_title(f"{title}\nat 10%: s3 {100 * r['s3_fused_uncertainty']:.1f}%, s4 {100 * r['s4_learned']:.1f}%, "
                 f"s1 {100 * r['s1_disagree']:.1f}%", loc="left", fontsize=11, fontweight="bold")


def fig2_review_flag() -> Path:
    hl = _hl()
    cur = {ds: json.loads((RES / f"cm_local/CM/{ds}/triage_curves.json").read_text()) for ds in ("crc", "cam")}
    gal = _gallery_items()
    WP, HP = W * DPI, (1075 if gal else 700)
    fig = plt.figure(figsize=(W, HP / DPI))
    box = lambda x0, top, w, h: fig.add_axes((x0 / WP, 1 - (top + h) / HP, w / WP, h / HP))  # pixel layout
    a1, a2, a3 = box(78, 100, 440, 370), box(545, 100, 440, 370), box(1065, 100, 415, 370)
    _triage_panel(a1, cur["crc"], hl["fusion"]["CM_crc"]["review_caught_at_10pct"], "CRC-VAL-HE-7K", (29.5, 0.12))
    _triage_panel(a2, cur["cam"], hl["fusion"]["CM_cam"]["review_caught_at_10pct"], "Camelyon17, hospital 2",
                  (29.5, 0.03))
    a1.set_ylabel("errors caught (fraction)")
    a2.tick_params(labelleft=False)
    fig.legend([Line2D([], [], color=c, ls=ls, lw=lw) for _, _, c, ls, lw in TRIAGE], [t[1] for t in TRIAGE],
               ncol=2, loc="upper center", bbox_to_anchor=(530 / WP, 1 - 528 / HP), fontsize=9.4, handlelength=2.6,
               columnspacing=1.6)
    rows = list(csv.DictReader(open(RES / "cm_local/CM/crc/blindspots.csv")))
    cls = [r["class"] for r in rows]
    onlyA = np.array([int(r["B_only"]) for r in rows])  # only MUSK right -> only CONCH wrong
    onlyB = np.array([int(r["A_only"]) for r in rows])  # only CONCH right -> only MUSK wrong
    same = np.array([int(r["both_wrong_same_label"]) for r in rows])
    diff = np.array([int(r["both_wrong"]) for r in rows]) - same
    y = np.arange(len(cls))[::-1]
    left = np.zeros(len(cls))
    parts = ((onlyA, "only CONCH wrong", C["A"]), (onlyB, "only MUSK wrong", C["B"]),
             (diff, "both wrong, different calls", C["light"]),
             (same, "both wrong, SAME call (disagreement is blind)", C["shared"]))
    for v, _, col in parts:
        a3.barh(y, v, left=left, color=col, height=0.72, edgecolor="white", linewidth=1.5)
        left += v
    for yy, tot, s in zip(y, left, same):
        if s:
            a3.text(tot + 3, yy, f"{int(s)}", va="center", fontsize=9, color=C["ink"])
    a3.set_yticks(y, cls)
    a3.set_xlabel("test tiles where a reader is wrong")
    _grid(a3, "x")
    a3.set_xlim(0, left.max() * 1.15)
    cse = hl["fusion"]["CM_crc"]["confident_shared_errors"]
    a3.set_title(f"Shared blind spots (CRC)\n{cse['n']} confident shared errors ({100 * cse['rate']:.2f}%)",
                 loc="left", fontsize=11, fontweight="bold")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts], [p[1] for p in parts], ncol=1,
               loc="upper center", bbox_to_anchor=(1250 / WP, 1 - 528 / HP), fontsize=9.2)
    if gal:
        fig.text(12 / WP, 1 - 668 / HP, "Confident shared errors: both readers wrong with the same call, each "
                 "≥ 0.80 confident (the 8 most confident shown)", fontsize=10.8,
                 fontweight="bold", color=C["ink"], va="bottom")
        sz, gap = 176, 9
        x0 = (WP - (len(gal) * sz + (len(gal) - 1) * gap)) / 2
        for j, (i, img, truth, call, ca, cb, fl) in enumerate(gal):
            a = box(x0 + j * (sz + gap), 725, sz, sz)
            a.imshow(img)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(True)
                s.set_color(C["shared"])
                s.set_linewidth(2)
            a.set_title(f"truth {truth}\nboth said {call}", fontsize=9.6)
            a.set_xlabel(f"CONCH {ca:.2f}\nMUSK {cb:.2f}\n#{i}, {'flagged' if fl else 'NOT flagged'}", fontsize=8.6)
    _title(fig, "Review flag: the fused reader's own uncertainty does the work; adding disagreement adds nothing",
           y=1 - 10 / HP)
    _foot(fig, [
        "Curves rank the test tiles by each score and send the top k to review. s1-s4 target the fused reader's errors; "
        "the dashed baselines target each reader's own errors.",
        "On Camelyon17 the pre-registered flag (s4) clears tiles no better than CONCH alone. Its val-calibrated threshold "
        "over-flags CRC test (val is near ceiling;",
        "val → test is a distribution shift). Post-hoc reproduction of the frozen fits, matching review.json exactly. "
        "CRC CIs are tile-level.", "Gallery: 'flagged' = the pre-registered rule. " + DISCLAIMER], y=6 / HP)
    return _save(fig, "fig2_review_flag.png")


# ---------------------------------------------------------------------------------------------------------------
def _px(fig, WP, HP):
    """Pixel-coordinate helpers (origin top-left) for a fixed-size figure: axes, left-aligned titles."""
    ax = lambda x, top, w, h: fig.add_axes((x / WP, 1 - (top + h) / HP, w / WP, h / HP))
    tt = lambda x, top, text, **kw: fig.text(x / WP, 1 - top / HP, text, ha="left", va="top", fontsize=12,
                                             fontweight="bold", color=C["ink"], linespacing=1.25, **kw)
    return ax, tt


def _hbar(ax, labels, vals, cols, fmt="{:,.0f}", log=True):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="white")
    for yy, v in zip(y, vals):
        ax.text(v * 1.15 if log else v + 0.012 * max(vals), yy, fmt.format(v), va="center", fontsize=9.5,
                color=C["ink"])
    ax.set_yticks(y, labels)
    if log:
        ax.set_xscale("log")
    _grid(ax, "x")


def fig3_kernels() -> Path:
    k = _hl()["kernels"]
    WP, HP = W * DPI, 1085
    fig = plt.figure(figsize=(W, HP / DPI))
    box, tt = _px(fig, WP, HP)
    a, b = box(300, 135, 430, 320), box(1000, 135, 460, 320)
    c, d = box(300, 615, 430, 320), box(1000, 615, 460, 320)
    _hbar(a, ["PIL, 1 CPU process", "PIL DataLoader, 8 workers", "torch GPU ops (exact)", "Triton (our baseline)",
              "K1 CUDA (ours)"],
          [k["pil_single_process_tiles_per_s_224"], k["dataloader_8w_tiles_per_s_224"],
           k["k1_torch_exact_tiles_per_s_224"], k["k1_triton_tiles_per_s_224"], k["k1_cuda_tiles_per_s_224"]],
          [C["grey"]] * 4 + [C["F"]])
    a.set_xlim(100, 2e6)
    a.set_xlabel("224-px tiles/s into BOTH readers' inputs (log)")
    tt(12, 72, "K1 dual preprocess: bit-exact to the official\nCONCH and MUSK transforms")
    e = k["k1_e2e_tiles_per_s"]
    vals = [e["reference"], e["k1"], e["reference"] * k["max_e2e_speedup_if_preprocessing_were_free"]]
    _hbar(b, ["reference (PIL)", "with K1", "ceiling: forward only"], vals, [C["grey"], C["F"], C["light"]],
          fmt="{:.1f}", log=False)
    b.set_xlim(0, max(vals) * 1.2)
    b.set_xlabel("CONCH + MUSK embeddings, tiles/s")
    tt(800, 72, f"End to end ×{k['k1_e2e_speedup_vs_reference']:.2f}: the model forward is "
                f"{100 * k['model_forward_share_of_reference_e2e']:.0f}%\nof the time "
                f"(preprocessing itself ×{k['k1_preprocess_speedup_vs_pil']:.0f})")
    _hbar(c, ["numpy, 1 process", "numpy, 4 processes", "torch GPU ops", "K2 CUDA (ours)"],
          [k["k2_numpy_1proc_images_per_s"], k["k2_numpy_4proc_images_per_s"], k["k2_torch_gpu_images_per_s_sev1"],
           k["k2_cuda_images_per_s_sev1"]], [C["grey"]] * 3 + [C["s4"]])
    c.set_xlim(10, 1e6)
    c.set_xlabel("224-px images/s, severity 1.0, batch 256 (log)")
    tt(12, 552, f"K2 camera-of-screen augmentation (SIMULATION):\n≤ {k['k2_per_op_max_levels']} level per op vs "
                f"the numpy reference")
    st = k["student_train_step_images_per_s"]["CM"]
    _hbar(d, ["no augmentation", "torch augmentation", "K2 augmentation"], [st["none"], st["torch"], st["k2"]],
          [C["light"], C["grey"], C["s4"]], log=False)
    d.set_xlim(0, st["none"] * 1.2)
    d.set_xlabel("student training step, images/s (batch 128)")
    tt(800, 552, f"Student training: K2 augments ×{k['student_train_aug_speedup_k2_vs_torch']['CM']:.1f} faster "
                 f"than\ntorch, close to training with no augmentation")
    _title(fig, "Two custom CUDA kernels in the data path (RTX 5060, sm_120; sm_80 and sm_90 compile-checked)",
           y=1 - 10 / HP)
    _foot(fig, [
        "CUDA-event timing (warmup 3, median of 10) on a shared GPU, so throughput is an estimate. K1 embedding cosine "
        "vs the official transforms ≥ 0.999997 for both readers.",
        "compute-sanitizer memcheck, racecheck and synccheck: clean. K1's end-to-end gain is small here because this "
        "pipeline is batched and GPU-forward-bound;",
        "the organizers' hosted server was bound by per-image CPU preprocessing, a regime not measured here. " +
        DISCLAIMER], y=6 / HP)
    return _save(fig, "fig3_kernels.png")


# ---------------------------------------------------------------------------------------------------------------
def fig5_edge_student() -> Path:
    hl = _hl()
    s, dev = hl["student"], hl["device"]
    cm, op = s["CM"], s["OPEN"]
    t = cm["teachers_test_bal_acc"]
    WP, HP = W * DPI, 880
    fig = plt.figure(figsize=(W, HP / DPI))
    box, tt = _px(fig, WP, HP)
    a, b = box(330, 105, 410, 600), box(1010, 130, 450, 305)
    rows = [("CONCH, primary probe", t["CONCH_probe"], C["A"], "o"),
            ("MUSK, primary probe", t["MUSK_probe"], C["B"], "o"),
            ("fused CONCH + MUSK", t["fused"], C["F"], "o"),
            ("CONCH, aligned-feature probe", t["CONCH_aligned_probe"], C["A"], "s"),
            ("MUSK, aligned-feature probe", t["MUSK_aligned_probe"], C["B"], "s"),
            ("CM student fp32, clean", cm["fp32_test_bal_acc"]["clean"], C["student"], "o"),
            ("CM student fp32, camera sim 0.5", cm["fp32_test_bal_acc"]["sev0.5"], C["student"], "^"),
            ("CM student fp32, camera sim 1.0", cm["fp32_test_bal_acc"]["sev1.0"], C["student"], "v"),
            ("CM student int8 (deployed)", cm["int8_test_bal_acc"], C["student"], "D"),
            ("OPEN student fp32 (distributable)", op["fp32_test_bal_acc"]["clean"], C["grey"], "o"),
            ("OPEN student int8 (distributable)", op["int8_test_bal_acc"], C["grey"], "D")]
    y = np.arange(len(rows))[::-1].astype(float)
    y[:5] += 0.6
    y[-2:] -= 0.6
    for yy, (lab, v, col, mk) in zip(y, rows):
        a.plot([0.88, v], [yy, yy], color=C["grid"], lw=1.2, zorder=1)
        a.plot(v, yy, mk, ms=9, color=col, mec="white", mew=1.5, zorder=3)
        a.text(v + 0.0018, yy, f"{v:.4f}", va="center", fontsize=9.3)
    a.set_yticks(y, [r[0] for r in rows], fontsize=9.6)
    a.axvline(t["CONCH_probe"], color=C["A"], lw=0.8, ls=":")
    a.set_xlim(0.88, 0.972)
    a.set_xlabel("CRC-VAL-HE-7K balanced accuracy (7,180 tiles)")
    _grid(a, "x")
    tt(12, 60, "Teachers (frozen features) vs the 224-px student")
    sizes = json.loads((RES / "student/model_sizes.json").read_text())["teachers"]
    vals = [sizes["conch"]["params_vision_tower"], sizes["musk"]["params_vision_tower"], s["params"]]
    yb = np.arange(3)[::-1]
    b.barh(yb, [v / 1e6 for v in vals], color=[C["A"], C["B"], C["student"]], height=0.6, edgecolor="white")
    for yy, v in zip(yb, vals):
        b.text(v / 1e6 * 1.15, yy, f"{v / 1e6:,.1f} M" if v > 1e7 else f"{v / 1e6:.2f} M", va="center", fontsize=9.5)
    b.set_xscale("log")
    b.set_xlim(1, 3000)
    b.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    b.set_yticks(yb, ["CONCH vision tower", "MUSK vision tower", "student, both readers"])
    b.set_xlabel("parameters, millions (log)")
    _grid(b, "x")
    tt(800, 60, f"Student: {s['params_pct_of_teacher_vision_towers']:.1f}% of the two vision towers' "
                f"parameters;\nint8 ONNX file {s['int8_onnx_mb']:.1f} MB")
    fig.text(800 / WP, 1 - 530 / HP, "Arduino UNO Q on-board latency: PENDING\n(measured when the board is plugged "
             "in; only an\nx86 simulation exists so far). LED-matrix sketch:\n" +
             ("compile-verified (" + dev["mcu_fqbn"] + "), not flashed." if dev["mcu_sketch_compile_verified"]
              else "NOT compile-verified."), ha="left", va="top", fontsize=10.2, color=C["ink"], linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.7", fc="#fff4d6", ec=C["s1"]))
    _title(fig, "Edge student: most of the teachers' accuracy at about 1% of the size; int8 costs about 2 points",
           y=1 - 10 / HP)
    _foot(fig, [
        "Student = mobilenetv4_conv_small with heads for both readers' aligned embeddings, the fused class and the "
        "review flag; trained on the 20k-tile subset (CE + KD, K2 augmentation).",
        "It is fine-tuned with labels, so matching frozen-feature probes is expected, not evidence of better features. "
        "Camera severities are a SIMULATION of a camera filming a screen.",
        "The CM student derives from CC-BY-NC-ND models and is not distributed; the OPEN student is. " + DISCLAIMER],
        y=6 / HP)
    return _save(fig, "fig5_edge_student.png")


# ---------------------------------------------------------------------------------------------------------------
def fig8_system() -> Path:
    s = _hl()["student"]
    fig = plt.figure(figsize=(W, 6.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 59)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec=None, fs=9.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc,
                                    ec=ec or "#d5d4ce", lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["ink"], linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, color=C["ink2"],
                                     lw=1.3))

    def lane(y, h, title, status, col):
        ax.add_patch(FancyBboxPatch((1, y), 98, h, boxstyle="round,pad=0,rounding_size=1.5", fc=col, ec="none"))
        ax.text(2.2, y + h - 1.4, title, fontsize=11, fontweight="bold", va="top", color=C["ink"])
        ax.text(97.8, y + h - 1.4, status, fontsize=9, va="top", ha="right", color=C["ink2"], style="italic")

    lane(38, 16.5, "1  Teachers (GPU)", "done locally on a 20k-tile subset · full 100k on NCSA Delta: pending",
         "#eef4fb")
    lane(20, 16, "2  Student training (GPU)", "done locally (CM research + OPEN distributable)", "#f3f1fb")
    lane(1.5, 16.5, "3  Arduino UNO Q, offline", "app + LED sketch built; on-board run and latency: pending",
         "#f0f8f2")
    w = "white"
    box(3, 40, 11, 8, "tiles\n(uint8)", w)
    box(18, 40, 14, 8, "K1 CUDA\ndual preprocess\n(bit-exact)", "#d6f0e5", C["F"])
    box(36, 45, 12, 5.5, "CONCH\n(reader A)", "#d7e6f7", C["A"])
    box(36, 38.8, 12, 5.5, "MUSK\n(reader B)", "#fbe0d4", C["B"])
    box(52, 40, 14, 8, "linear probes\n+ fused reader", w)
    box(70, 40, 13, 8, "review signals\ns1-s4", w)
    box(87, 40, 11, 8, "triage +\nblind spots", w)
    arrow(14, 44, 18, 44); arrow(32, 45.5, 36, 47.5); arrow(32, 42.5, 36, 41.5)
    arrow(48, 47.5, 52, 45.5); arrow(48, 41.5, 52, 42.5); arrow(66, 44, 70, 44); arrow(83, 44, 87, 44)
    box(3, 22, 14, 8, "tiles + teacher\ntargets (zA, zB,\nfused, flag)", w, fs=9.2)
    box(21, 22, 16, 8, "K2 CUDA camera-\nof-screen aug.\n(SIMULATION)", "#e3dff8", C["s4"], fs=9.2)
    box(41, 22, 20, 8, f"student CNN, 224 px\nheads: hA, hB, class, review\n({s['params_millions']:.2f} M params)",
        w, fs=9.2)
    box(65, 22, 15, 8, f"ONNX → int8\n({s['int8_onnx_mb']:.1f} MB model,\nSHA256-checked)", w, fs=9.2)
    arrow(17, 26, 21, 26); arrow(37, 26, 41, 26); arrow(61, 26, 65, 26)
    arrow(59, 40, 10, 30.3)
    box(3, 3.5, 14, 8, "USB camera\n(films a screen\nin the demo)", w, fs=9.2)
    box(21, 3.5, 14, 8, "crop + tissue\ncheck", w, fs=9.2)
    box(39, 3.5, 17, 8, "int8 student\n(onnxruntime,\n4 Cortex-A53)", w, fs=9.2)
    box(60, 3.5, 16, 8, "call + AGREE /\nREVIEW flag", "#d6f0e5", C["F"], fs=9.2)
    box(80, 8.2, 18, 3.8, "local web UI (127.0.0.1)", w, fs=9)
    box(80, 3.2, 18, 3.8, "8×13 LED matrix (MCU)", w, fs=9)
    arrow(17, 7.5, 21, 7.5); arrow(35, 7.5, 39, 7.5); arrow(56, 7.5, 60, 7.5); arrow(76, 8.5, 80, 10)
    arrow(76, 6.5, 80, 5.1)
    arrow(72.5, 22, 47.5, 11.8)
    ax.text(1.2, 58.2, "TwinRead: two foundation-model readers → fused reader + review flag → offline edge student",
            ha="left", va="top", fontsize=14.5, fontweight="bold")
    ax.text(50, 0.2, DISCLAIMER + " No network at run time on the device.", ha="center", va="bottom", fontsize=9,
            color=C["ink2"])
    return _save(fig, "fig8_system.png")


FIGS = {"fig1": fig1_two_readers, "fig2": fig2_review_flag, "fig3": fig3_kernels, "fig5": fig5_edge_student,
        "fig8": fig8_system}


def build_all(only=None, out: Path | None = None) -> list[Path]:
    """Rewrite results/headline.json, then draw the figures into `out` (default figures/)."""
    global FIG
    headline.write(RES)
    if out is not None:
        FIG = Path(out)
    return [f() for k, f in FIGS.items() if not only or k in only]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help=f"subset of {list(FIGS)}")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default figures/)")
    a = ap.parse_args()
    for p in build_all(a.only, a.out):
        print(p)
