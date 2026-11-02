"""The talk deck: editable, within the organizers' slide limit, and in step with results/headline.json."""
import json
import re
import zipfile
from pathlib import Path

import pytest
from test_manifest import OVERCLAIMS

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "twinread_deck.pptx"


def _parts(prefix):
    with zipfile.ZipFile(DECK) as z:
        names = sorted(n for n in z.namelist() if re.fullmatch(prefix + r"\d+\.xml", n))
        return {n: z.read(n).decode("utf8") for n in names}


def _text(xml):
    return " ".join(re.findall(r"<a:t>([^<]*)</a:t>", xml)).replace("&amp;", "&")


@pytest.fixture(scope="module")
def slides():
    return [_text(x) for x in _parts("ppt/slides/slide").values()]


def test_paragraphs_have_one_properties_element():
    """pptxgenjs repeats <a:pPr> in mixed-run paragraphs. That is invalid, and it silently drops bullets."""
    for prefix in ("ppt/slides/slide", "ppt/slideLayouts/slideLayout", "ppt/notesSlides/notesSlide"):
        for name, xml in _parts(prefix).items():
            for p in re.findall(r"<a:p>.*?</a:p>", xml, flags=re.S):
                assert len(re.findall(r"<a:pPr[ />]", p)) <= 1, name


def _in_order(prefix):
    parts = _parts(prefix)
    return [parts[n] for n in sorted(parts, key=lambda n: int(re.search(r"(\d+)\.xml$", n)[1]))]


def _titles():
    return [re.sub(r"\s+", " ", _text(re.search(r'<p:sp>(?:(?!</p:sp>).)*?type="title".*?</p:sp>', x, flags=re.S)[0])).strip()
            for x in _in_order("ppt/slides/slide")]


def test_within_limit_with_native_charts_and_notes():
    assert _titles().index("References") <= 20  # the organizers' limit is on main slides
    assert len(_parts("ppt/charts/chart")) >= 5
    assert len(_parts("ppt/notesSlides/notesSlide")) == len(_parts("ppt/slides/slide"))


def test_disclaimer_is_spoken_and_in_the_readme_not_on_slides():
    phrase = "not a diagnostic device"
    for prefix in ("ppt/slides/slide", "ppt/slideLayouts/slideLayout", "ppt/slideMasters/slideMaster"):
        assert not [n for n, x in _parts(prefix).items() if phrase in x.lower()]
    assert any(phrase in _text(x) for x in _in_order("ppt/notesSlides/notesSlide"))
    assert phrase in (ROOT / "README.md").read_text().lower()


def test_headline_numbers_match_results(slides):
    hl = json.loads((ROOT / "results" / "headline.json").read_text())
    c, k, d = hl["fusion"]["CM_crc"], hl["kernels"], hl["device"]["onboard"]
    text = " ".join(slides)
    step = k["student_train_step_images_per_s"]["CM"]
    for want in (f"{100 * c['bal_acc']['fused']:.2f}%", f"{100 * c['bal_acc']['B']:.2f}%", f"{100 * c['bal_acc']['A']:.2f}%",
                 f"+{c['delta_vs_best_single_pts']:.2f}",
                 f"{100 * c['review_caught_at_10pct']['s3_fused_uncertainty']:.1f}%",
                 f"{100 * c['review_caught_at_10pct']['s1_disagree']:.1f}%",
                 f"{d['cm_int8_p50_ms_4_threads']:.1f} ms", f"{100 * d['cm_int8_bal_acc_board']:.2f}%",
                 f"×{k['k1_preprocess_speedup_vs_pil']:.0f}", f"×{k['k1_e2e_speedup_vs_reference']:.2f}",
                 f"×{k['student_train_aug_speedup_k2_vs_torch']['CM']:.1f}", f"×{step['k2'] / step['torch']:.2f}"):
        assert want in text, want


def test_dependent_kernel_slide_recomputes_from_results(slides):
    """One launch per stage: uint8 in and out, 9 float32 intermediates each written and read. Fused: 1 float32 hand-off."""
    k1 = json.loads((ROOT / "results/kernels/k1_bench.json").read_text())["gpu"]
    k2 = json.loads((ROOT / "results/kernels/k2_bench.json").read_text())["gpu"]
    u8 = k2["batch"] * k2["hw"][0] * k2["hw"][1] * 3
    unfused, fused = 2 * u8 + 18 * (4 * u8), 2 * u8 + 2 * (4 * u8)
    assert round(fused / 1e6) == 385  # k2_bench.json bytes_note
    unfused_ms = unfused / (k1["memcpy_d2d"]["GBps_read_plus_write"] * 1e9) * 1e3
    k2_ms = k2["severity_1.0"]["d_cuda_k2"]["ms"]
    assert k2_ms < unfused_ms  # the slide's claim
    over_floor = k1["tiles224"]["cuda_k1"]["ms"] / k1["tiles224"]["two_pass_dram_floor_ms"]
    text = next(t for t in slides if "Fusing dependent stages" in t)
    for want in (f"\u00d7{unfused / fused:.1f} the traffic", f"K2 takes {k2_ms:.1f} ms", f"within {round(100 * (over_floor - 1))}%"):
        assert want in text, want


def test_why_shrink_slide_matches_results(slides):
    ms = json.loads((ROOT / "results/student/model_sizes.json").read_text())["students"]["CM"]
    kc = json.loads((ROOT / "results/uno_q/onboard/int8_kernel_check.json").read_text())["cm"]
    x86 = json.loads((ROOT / "results/uno_q/backbone_latency_x86_sim.json").read_text())["rows"]
    row = {r["precision"]: r for r in x86 if r["backbone"] == "mobilenetv4_conv_small" and r["threads"] == 4}
    fp, i8 = ms["fp32_param_bytes"], ms["int8_onnx_bytes"]
    text = next(t for t in slides if "Why quantise" in t)
    for want in (f"{fp / 1e6:.1f} MB down to {i8 / 1e6:.1f} MB, \u00d7{fp / i8:.1f}",
                 f"\u00d7{row['fp32']['p50_ms'] / row['int8']['p50_ms']:.1f} faster",
                 f"\u00d7{row['fp32']['rss_mb'] / row['int8']['rss_mb']:.1f} less RAM",
                 f"{100 * kc['fp32_student_bal_acc']:.2f}% to {100 * kc['bal_acc']['board']:.2f}%",
                 f"{100 * kc['pred_agreement_with_fp32_student']['board']:.1f}% of calls"):
        assert want in text, want
    assert "x86" in text and "not timed on the board" in text  # the speed claim stays labelled
    assert text.count("Low-access clinics") == 1 and "Hurricane Maria" in text and "Mars" in text
    assert "FCC 2017" in text and "NASA" in text and "Arduino 2026" in text  # the external facts stay cited
    assert any("Why quantise" in _text(x) and "buChar" in x for x in _in_order("ppt/slides/slide"))


def test_team_hand_edits_survive_a_rebuild(slides):
    """Their edits in docs/deck/manual_edits/ are carried by the generator, which would otherwise overwrite them."""
    titles = _titles()
    for want in ("Why a second opinion matters", "CONCH and MUSK make different mistakes", "System Architecture",
                 "Implementation on Arduino UNO Q", "Future Directions"):
        assert want in titles
    text = " ".join(slides)
    for want in ("And can we make it run offline on a small device?", "out of the 505 were misclassified by both models",
                 "Combine Predictions", "Distill", "Student model", "both wrong (&gt;80% confidence)"):
        assert want in text, want
    for gone in ("Fixed in advance", "Overcome along the way", "independent readers, and a second look"):
        assert gone not in text


def test_science_comes_before_the_kernels():
    titles = _titles()
    at = lambda start: next(i for i, t in enumerate(titles) if t.startswith(start))  # noqa: E731
    assert at("Linear probes") < at("Fusion: average") < at("Fusion adds") < at("Distillation") < at("Our trained student") < at("CUDA kernels")


def test_probe_and_fusion_slides_match_results(slides):
    r = json.loads((ROOT / "results/cm_local/CM/crc/readers.json").read_text())
    f = json.loads((ROOT / "results/cm_local/CM/crc/fusion.json").read_text())
    pf = json.loads((ROOT / "docs/deck/assets/probe_fusion.json").read_text())
    probe = next(t for t in slides if "Linear probes:" in t)
    notes = " ".join(_text(x) for x in _in_order("ppt/notesSlides/notesSlide"))  # chart values are not slide text
    for k, dim in (("A", pf["probe_feature_dims"]["conch"]), ("B", pf["probe_feature_dims"]["musk"])):
        assert f"{100 * r[k]['zeroshot']['bal_acc']:.1f}" in notes and f"{100 * r[k]['probe']['bal_acc']:.1f}" in notes
        assert f"{dim:,} numbers" in probe and f"{9 * (dim + 1):,} weights" in probe
    fusion = next(t for t in slides if "Fusion: average" in t)
    b, c = f["mcnemar_fused_vs_B"]["b"], f["mcnemar_fused_vs_B"]["c"]
    assert f"fixes {b} and adds {c}" in fusion and "286 \u2192 239" in fusion and 286 - b + c == 239
    ex = pf["example"]
    assert abs((ex["conch"]["TUM"] + ex["musk"]["TUM"]) / 2 - ex["fused"]["TUM"]) < 1e-6  # the average, as the slide says
    assert ex["flagged"] and f"only {round(100 * ex['fused']['TUM'])}% sure" in fusion


def test_student_slide_matches_results(slides):
    hl = json.loads((ROOT / "results/headline.json").read_text())
    c, s, d = hl["fusion"]["CM_crc"], hl["student"], hl["device"]["onboard"]
    text = next(t for t in slides if "Our trained student" in t)
    fp = s["CM"]["fp32_test_bal_acc"]
    behind = 100 * (c["bal_acc"]["fused"] - d["cm_int8_bal_acc_board"])
    assert f"{s['params_pct_of_teacher_vision_towers']:.1f}% of the size, {behind:.1f} points behind" in text
    assert "not a like-for-like" in text and "not real captures" in text  # the caveats stay on the slide
    assert fp["clean"] < c["bal_acc"]["B"] and d["cm_int8_bal_acc_board"] < c["bal_acc"]["A"]  # "behind" is true


def test_notes_are_a_recitable_script_that_fits():
    notes = _in_order("ppt/notesSlides/notesSlide")[:_titles().index("References")]
    seconds = 0
    for x in notes:
        m = re.search(r"\[(\d+):(\d\d)\]", _text(x))
        seconds += 60 * int(m[1]) + int(m[2])
        assert "buChar" in x and x.count("<a:p>") > 3  # real bullets, one paragraph per line
    assert seconds <= 13 * 60
    assert sum("How we built it:" in _text(x) for x in notes) >= 4 and any("How it came together" in _text(x) for x in notes)


def test_no_overclaims(slides):
    notes = [_text(x) for x in _parts("ppt/notesSlides/notesSlide").values()]
    text = " ".join(slides + notes).lower()
    assert not [p for p in OVERCLAIMS if p in text]
    # x10.6 is the augmentation stage; the whole training step is x2.49
    assert not re.search(r"training[^.]{0,40}×10\.6", text)
