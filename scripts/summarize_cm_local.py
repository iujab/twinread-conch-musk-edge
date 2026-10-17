"""Print the CM local-subset headline numbers from results/cm_local/CM/<ds>/*.json (no number is typed by hand)."""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results" / "cm_local" / "CM"
f = lambda x: f"{x:.4f}"
ci = lambda v, i: f"{f(v)} [{f(i['lo'])}, {f(i['hi'])}]" + (" UNSTABLE" if i["unstable"] else "")
for ds in ("crc", "cam"):
    d = {k: json.load(open(R / ds / f"{k}.json")) for k in ("readers", "fusion", "review", "preregistration")}
    rd, fu, rv = d["readers"], d["fusion"], d["review"]
    A, B, p = rd["A"]["probe"], rd["B"]["probe"], fu["primary"]
    print(f"## {ds}  ({fu['ci_mode']})")
    print("conch probe", ci(A["bal_acc"], A["bal_acc_ci_info"]), "| musk probe", ci(B["bal_acc"], B["bal_acc_ci_info"]))
    print("zeroshot conch/musk", f(rd["A"]["zeroshot"]["bal_acc"]), f(rd["B"]["zeroshot"]["bal_acc"]))
    pc = fu["candidates"][p]
    print("best single", fu["best_single_reader"], "| fused", p, ci(pc["bal_acc"], pc["bal_acc_ci_info"]))
    i = fu["delta_ci_info"]; print("delta", f"{fu['delta_bal_acc_vs_best_single']:+.4f} [{i['lo']:+.4f}, {i['hi']:+.4f}]", "unstable" if i["unstable"] else "")
    m = fu["mcnemar_fused_vs_best_single"]; print("mcnemar vs best", m, "|", fu["mcnemar_note"])
    print("val bal acc", fu["val_balanced_accuracy"])
    for s in ("s1_disagree", "s2_js", "s3_fused_unc", "s4", "base_A", "base_B"):
        c = rv["scores"][s]; u = c.get("caught_10_ci_info", {})
        print(f"  {s:13s} caught@10% {c['caught']['0.1']['errors_caught']:.3f}", c.get("caught_10_ci"), "UNSTABLE" if u.get("unstable") else "")
    print("  s4 val-threshold operating points:", json.dumps(rv["s4_val_threshold_operating_points"]))
    print("  fused error rate", rv["fused_error_rate"], "| confident shared errors", rv["confident_shared_errors"]["n"], rv["confident_shared_errors"]["rate"])
