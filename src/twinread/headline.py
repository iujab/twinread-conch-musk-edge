"""results/headline.json: the numbers the README, project.json, figures and dashboard quote, computed from the source
results JSONs (never typed by hand). Derived values (ratios, percentage points, sums) carry their derivation.

    PYTHONPATH=src .venv/bin/python -m twinread.headline        # rewrites results/headline.json

Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIR_ROOTS = {"CM": "cm_local/CM", "OPEN": "open/OPEN"}
UNO_Q_PENDING = "pending: measured when the board is plugged in"


def _j(results: Path, rel: str) -> dict:
    return json.loads((results / rel).read_text())


def uno_q_latency(results: Path):
    """On-board latency from results/uno_q/backbone_latency.json, or the pending string.

    Counted as on-board only if the harness recorded ``simulated: false`` on an aarch64 host and at least one row has a
    latency (device/bench_backbones.py flags every non-aarch64 or --simulate run simulated=True); anything else, e.g. an
    x86 run saved under that name, stays pending."""
    p = Path(results) / "uno_q" / "backbone_latency.json"
    if not p.exists():
        return UNO_Q_PENDING
    try:
        d = json.loads(p.read_text())
    except ValueError:
        return UNO_Q_PENDING
    host = d.get("host") or {}
    measured = [r for r in d.get("rows", []) if "p50_ms" in r and "error" not in r]
    if d.get("simulated") is False and host.get("machine") in ("aarch64", "arm64") and measured:
        return d
    return UNO_Q_PENDING


def _pairs(results: Path, base: str) -> dict:
    """(truth -> shared call) counts of the confident shared errors, from the post-hoc triage_curves.json if present."""
    p = results / base / "triage_curves.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {"truth_to_call": d["confident_shared_error_pairs"],
            "flagged_by_prereg_rule": d["confident_shared_errors_flagged_by_prereg_rule"]}


def _pair(results: Path, pair: str, ds: str) -> dict:
    base = f"{PAIR_ROOTS[pair]}/{ds}"
    r, f, v = (_j(results, f"{base}/{n}.json") for n in ("readers", "fusion", "review"))
    sc = v["scores"]
    c10 = lambda k: sc[k]["caught"]["0.1"]
    best = f["best_single_reader"]
    return {
        "source": f"results/{base}/{{readers,fusion,review}}.json",
        "readers": {"A": r["A"]["name"], "B": r["B"]["name"]},
        "bal_acc": {"A": r["A"]["probe"]["bal_acc"], "B": r["B"]["probe"]["bal_acc"],
                    "fused": f["candidates"][f["primary"]]["bal_acc"]},
        "bal_acc_ci": {"A": r["A"]["probe"]["bal_acc_ci"], "B": r["B"]["probe"]["bal_acc_ci"],
                       "fused": f["candidates"][f["primary"]]["bal_acc_ci"]},
        "aligned_probe_bal_acc": {"A": r["A"]["aligned_probe"]["bal_acc"], "B": r["B"]["aligned_probe"]["bal_acc"]},
        "zeroshot_bal_acc": {k: r[k]["zeroshot"]["bal_acc"] for k in "AB" if "zeroshot" in r[k]},
        "primary_fusion": f["primary"], "best_single_reader": best,
        "delta_vs_best_single": f["delta_bal_acc_vs_best_single"], "delta_ci": f["delta_ci"],
        "delta_vs_best_single_pts": 100 * f["delta_bal_acc_vs_best_single"],
        "delta_ci_pts": [100 * x for x in f["delta_ci"]],
        "mcnemar_vs_best_single_p": f["mcnemar_fused_vs_best_single"]["p"],
        "ci_mode": f["ci_mode"],
        "review_caught_at_10pct": {
            "s1_disagree": c10("s1_disagree")["errors_caught"], "s2_js": c10("s2_js")["errors_caught"],
            "s3_fused_uncertainty": c10("s3_fused_unc")["errors_caught"], "s4_learned": c10("s4")["errors_caught"],
            "A_own_confidence_own_errors": c10("base_A_own_error")["errors_caught"],
            "B_own_confidence_own_errors": c10("base_B_own_error")["errors_caught"]},
        "acc_cleared_at_10pct": {"s3": c10("s3_fused_unc")["acc_cleared"], "s4": c10("s4")["acc_cleared"],
                                 "best_single_own": c10(f"base_{best}_own_error")["acc_cleared"],
                                 "A_own": c10("base_A_own_error")["acc_cleared"],
                                 "B_own": c10("base_B_own_error")["acc_cleared"]},
        "aurc": {k: sc[k]["aurc"] for k in ("s1_disagree", "s2_js", "s3_fused_unc", "s4",
                                             "base_A_own_error", "base_B_own_error")},
        "prereg_rule_test_operating_point": v["s4_val_threshold_operating_points"]["0.1"],
        "confident_shared_errors": {"n": v["confident_shared_errors"]["n"], "rate": v["confident_shared_errors"]["rate"],
                                    **_pairs(results, base)},
        "fused_error_rate": v["fused_error_rate"],
        "s1_disagree_rate": v["errors_by_flag"]["s1_disagree_rate"],
        "s1_budget_filled_in_test_order_at_10pct": 0.1 - v["errors_by_flag"]["s1_disagree_rate"],
        "s1_note": "s1 is binary: at a 10% budget every disagreement is flagged and the rest of the budget is filled "
                   "with tied (agreeing) tiles in stable test order, which is shuffled, so that part is effectively "
                   "random (review.errors_caught, argsort kind=stable)",
    }


def build(results: Path = ROOT / "results") -> dict:
    results = Path(results)
    fus = {f"{p}_{d}": _pair(results, p, d) for p in PAIR_ROOTS for d in ("crc", "cam")}
    pre_crc = _j(results, "cm_local/CM/crc/preregistration.json")
    post = _j(results, "cm_local/CM/crc/post_hoc_class_candidates_s3s4.json")
    classes = pre_crc["classes"]
    dlt = post["per_class_recall_delta_fused_minus_best_single"]

    k1b, k1e, k1p = (_j(results, f"kernels/{n}.json") for n in ("k1_bench", "k1_embed_e2e", "k1_parity"))
    k2b, k2p = _j(results, "kernels/k2_bench.json"), _j(results, "kernels/k2_parity.json")["parity"]
    san, builds = _j(results, "kernels/k2_sanitizer.json"), _j(results, "kernels/builds.json")
    tp = {p: _j(results, f"student/{d}/throughput.json")["results"] for p, d in (("CM", "cm_crc"), ("OPEN", "open_crc"))}
    sets = k1p["result"]["sets"]
    bit_exact = min(sets[s][impl][r]["frac_bit_exact_vs_official_half"]
                    for s in sets for impl in ("cuda", "triton", "torch") if impl in sets[s] for r in ("conch", "musk"))
    s = k1e["summary"]
    kernels = {
        "source": "results/kernels/{k1_bench,k1_parity,k1_embed_e2e,k2_bench,k2_parity,k2_sanitizer,builds}.json, "
                  "results/student/{cm_crc,open_crc}/throughput.json",
        "gpu": k1b["env"]["gpu"],
        "k1_bit_exact_fraction_min_over_sets_impls_readers": bit_exact,
        "k1_cuda_tiles_per_s_224": k1b["summary"]["tiles224"]["cuda_k1_tiles_per_s"],
        "k1_cuda_tiles_per_s_224_note": "k1_bench.json: input already on the GPU, both readers, batch 256 (excludes "
                                        "host-to-device copies; see k1_cuda_incl_h2d_tiles_per_s_224)",
        "k1_cuda_incl_h2d_tiles_per_s_224": k1b["gpu"]["tiles224"]["cuda_k1_incl_h2d"]["tiles_per_s"],
        "k1_bench_speedup_vs_pil_single_process": k1b["summary"]["tiles224"]["speedup_vs_pil_single_process"],
        "k1_triton_tiles_per_s_224": k1b["gpu"]["tiles224"]["triton"]["tiles_per_s"],
        "k1_torch_exact_tiles_per_s_224": k1b["gpu"]["tiles224"]["torch_exact"]["tiles_per_s"],
        "pil_single_process_tiles_per_s_224": k1b["cpu"]["tiles224"]["pil_single_process"]["tiles_per_s"],
        "dataloader_8w_tiles_per_s_224": k1b["cpu"]["tiles224"]["dataloader_8w"]["tiles_per_s"],
        "k1_e2e_tiles_per_s": s["e2e_tiles_per_s"],
        "k1_e2e_speedup_vs_reference": s["e2e_speedup_k1_vs_reference"],
        "k1_e2e_speedup_vs_dataloader_4w": s["e2e_speedup_k1_vs_dataloader_4w"],
        "k1_preprocess_speedup_vs_pil": s["preprocess_speedup_k1_vs_pil"],
        "k1_preprocess_speedup_vs_pil_source": "k1_embed_e2e.json: K1 on the GPU incl. H2D vs the official PIL "
                                               "transforms, 512 CRC tiles, both readers",
        "k1_e2e_preprocess_tiles_per_s": {"k1_gpu_incl_h2d": k1e["results"]["preprocess_k1_gpu"]["tiles_per_s"],
                                          "pil_reference_cpu": k1e["results"]["preprocess_reference_cpu"]["tiles_per_s"]},
        "model_forward_share_of_reference_e2e": s["share_of_reference_e2e"]["model_forward"],
        "preprocess_share_of_reference_e2e": s["share_of_reference_e2e"]["preprocess_cpu"],
        "max_e2e_speedup_if_preprocessing_were_free": s["max_e2e_speedup_if_preprocessing_were_free"],
        "k1_vs_reference_embedding_cosine_min": {r: k1e["results"]["k1_vs_reference_cosine_aligned"][r]["min"]
                                                 for r in ("conch", "musk")},
        "k1_context": k1e["context"],
        "k2_cuda_images_per_s_sev1": k2b["gpu"]["severity_1.0"]["d_cuda_k2"]["images_per_s"],
        "k2_torch_gpu_images_per_s_sev1": k2b["gpu"]["severity_1.0"]["c_torch_gpu"]["images_per_s"],
        "k2_numpy_1proc_images_per_s": k2b["cpu"]["a_numpy_single_process"]["images_per_s"],
        "k2_numpy_4proc_images_per_s": k2b["cpu"]["b_numpy_pool_4proc"]["images_per_s"],
        "k2_per_op_max_levels": max(v["cuda"]["max_levels"] for v in k2p["per_op"].values()),
        "k2_composition_jpeg_max_err_non_exempt_levels":
            k2p["composition_summary"]["cuda"]["decomposed"]["jpeg_max_err_non_exempt"],
        "k2_composition_exempt_block_fraction": k2p["composition_summary"]["cuda"]["decomposed"]["exempt_frac"],
        "k2_composition_worst_max_levels": k2p["composition_summary"]["cuda"]["worst_max_levels"],
        "k2_sanitizer_clean": all(t["clean"] for t in san["tools"].values()),
        "builds_compile_ok": {a: b["ok"] for a, b in builds["builds"].items()},
        "student_train_aug_images_per_s": {p: {k: tp[p][k]["aug_images_per_s"] for k in ("k2", "torch")} for p in tp},
        "student_train_aug_speedup_k2_vs_torch": {p: tp[p]["k2"]["aug_images_per_s"] / tp[p]["torch"]["aug_images_per_s"]
                                                  for p in tp},
        "student_train_step_images_per_s": {p: {k: tp[p][k]["train_step_images_per_s"] for k in ("k2", "torch", "none")}
                                            for p in tp},
        "simulation_note": "K2 output is a SIMULATION of a USB camera filming a screen.",
    }

    cm_m, cm_i = _j(results, "student/cm_crc/metrics.json"), _j(results, "student/cm_crc/int8_drift.json")
    op_m, op_i = _j(results, "student/open_crc/metrics.json"), _j(results, "student/open_crc/int8_drift.json")
    cm_f, op_f = (_j(results, f"student/{d}/int8_flag_fidelity.json")["int8_device_flag"] for d in ("cm_crc", "open_crc"))
    sz = _j(results, "student/model_sizes.json")
    cm = fus["CM_crc"]
    teachers = {"CONCH_probe": cm["bal_acc"]["A"], "MUSK_probe": cm["bal_acc"]["B"], "fused": cm["bal_acc"]["fused"],
                "CONCH_aligned_probe": cm["aligned_probe_bal_acc"]["A"],
                "MUSK_aligned_probe": cm["aligned_probe_bal_acc"]["B"]}
    stud = lambda m, i, fl: {
        "fp32_test_bal_acc": {sev: m["test"][sev]["bal_acc"] for sev in ("clean", "sev0.5", "sev1.0")},
        "fp32_val_bal_acc_clean": m["val"]["clean"]["bal_acc"],
        "fp32_hA_hB_through_aligned_probes_test": {"A": m["test"]["clean"]["probe_A_bal_acc"],
                                                   "B": m["test"]["clean"]["probe_B_bal_acc"]},
        "int8_test_bal_acc": i["bal_acc"]["test"]["int8"], "fp32_onnx_test_bal_acc": i["bal_acc"]["test"]["fp32_onnx"],
        "int8_val_bal_acc": i["bal_acc"]["val"]["int8"],
        "int8_cost_vs_fp32_test_pts": 100 * (m["test"]["clean"]["bal_acc"] - i["bal_acc"]["test"]["int8"]),
        "int8_test_label_agreement_with_fp32": i["test_drift"]["label_agreement"],
        "int8_bundle_flag_rate_test": i["flag_rate_int8"]["test_recal_thr"],
        "int8_bundle_flag_rate_note": "deployed int8 bundle with the thresholds recalibrated on int8 val outputs "
                                      "(int8_drift.json flag_rate_int8.test_recal_thr)",
        "fp32_device_flag_test_clean": {k: m["test"]["clean"]["device_flag"][k]
                                        for k in ("fraction", "student_errors_caught", "teacher_recall", "acc_cleared",
                                                  "agreement", "kappa")},
        "int8_device_flag_test_clean": {k: fl[k] for k in ("fraction", "errors_caught", "teacher_recall",
                                                           "acc_cleared", "agreement", "kappa")},
        "device_flag_note": "flag vs the teacher flag (pre-registered s4 at its val threshold; 25-26% of CRC test): "
                            "kappa, teacher_recall, agreement. The two device 'readers' are two heads on ONE shared "
                            "student backbone. int8: int8_flag_fidelity.json (device runtime, incl. no-tissue clause)"}
    student = {
        "source": "results/student/{cm_crc,open_crc}/{metrics,int8_drift,int8_flag_fidelity}.json, "
                  "results/student/model_sizes.json",
        "backbone": cm_m["backbone"],
        "CM": stud(cm_m, cm_i, cm_f) | {"teachers_test_bal_acc": teachers,
                                  "int8_below_every_teacher": cm_i["bal_acc"]["test"]["int8"] < min(teachers.values()),
                                  "distributable": False,
                                  "licence_note": "CONCH and MUSK are CC-BY-NC-ND-4.0; the CM student is not distributed"},
        "OPEN": stud(op_m, op_i, op_f) | {"distributable": True},
        "params": sz["students"]["CM"]["params_total"],
        "params_millions": sz["students"]["CM"]["params_total"] / 1e6,
        "int8_onnx_bytes": sz["students"]["CM"]["int8_onnx_bytes"],
        "int8_onnx_mb": sz["students"]["CM"]["int8_onnx_bytes"] / 1e6,
        "teacher_vision_tower_params": sz["ratios"]["CM"]["teacher_vision_towers_params_sum"],
        "teacher_vision_tower_params_millions": sz["ratios"]["CM"]["teacher_vision_towers_params_sum"] / 1e6,
        "params_pct_of_teacher_vision_towers": 100 * sz["ratios"]["CM"]["student_params_over_teacher_vision_towers"],
    }

    rig = _j(results, "rig/decode_stress.json")
    mcu = _j(results, "uno_q/mcu_compile.json")
    lat = _j(results, "uno_q/backbone_latency_x86_sim.json")
    device = {
        "source": "results/rig/decode_stress.json, results/uno_q/{mcu_compile,backbone_latency_x86_sim}.json",
        "mcu_sketch_compile_verified": bool(mcu["compile_verified"] and mcu["exit_code"] == 0),
        "mcu_fqbn": mcu["fqbn"], "mcu_flashed": False,
        "uno_q_latency": uno_q_latency(results),
        "x86_latency_table_is_simulation": bool(lat["simulated"]),
        "rig_simulated_captures": sum(r["trials"] for r in rig["runs"]),
        "rig_simulated_wrong_ids": sum(r["wrong_id"] for r in rig["runs"]),
        "rig_simulated_accepted": sum(r["ok"] for r in rig["runs"]),
        "rig_accept_rate_min": min(r["accept_rate"] for r in rig["runs"]),
        "rig_note": rig["note"],
        "real_camera_captures": 0,
    }

    return {"note": "Computed by twinread.headline from the source JSONs named in each block. LOCAL-SUBSET dev "
                    "numbers (CRC 20k-tile train subset + full CRC-VAL-HE-7K test; Camelyon17-WILDS 7.5k subset); "
                    "the full-data Delta runs are pending. Research prototype, not a diagnostic device.",
            "fusion": fus,
            "crc_fusion_fragility": {
                "source": "results/cm_local/CM/crc/post_hoc_class_candidates_s3s4.json (post hoc)",
                "per_class_recall_delta_fused_minus_musk_pts": {c: 100 * d for c, d in zip(classes, dlt)},
                "per_class_contribution_to_bal_acc_delta_pts": {c: 100 * d / len(classes) for c, d in zip(classes, dlt)},
                "mus_plus_str_contribution_pts": 100 * (dlt[classes.index("MUS")] + dlt[classes.index("STR")]) / len(classes),
                "tum_contribution_pts": 100 * dlt[classes.index("TUM")] / len(classes),
                "contribution_note": "balanced accuracy = mean of per-class recalls, so each class contributes its recall "
                                     "delta / 9; the MUS and STR swings cancel and the net gain is mostly TUM (ruling R33)",
                "val_correct_tiles": {k: v["val_correct_tiles"] for k, v in post["candidates"].items()},
                "n_val": post["n_val"],
                "test_delta_vs_musk_by_candidate": {k: v["test_delta_vs_best_single"] for k, v in post["candidates"].items()},
                "s3_s4_spearman": post["s3_s4_agreement"]["spearman"],
                "s3_s4_top10_overlap": post["s3_s4_agreement"]["top10pct_overlap"],
                "val_fused_bal_acc": pre_crc["val_balanced_accuracy"][pre_crc["primary_fusion"]]},
            "kernels": kernels, "student": student, "device": device,
            "pending": ["UNO Q on-board latency and offline demo (" + UNO_Q_PENDING + ")",
                        "real camera-of-screen captures from the rig (none recorded yet)",
                        "NCSA Delta full-data CONCH+MUSK runs (100k CRC, A100; kit in ncsa/)"]}


def write(results: Path = ROOT / "results") -> Path:
    p = Path(results) / "headline.json"
    p.write_text(json.dumps(build(results), indent=1, default=str))
    return p


if __name__ == "__main__":
    print(write())
