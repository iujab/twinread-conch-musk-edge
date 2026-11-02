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


def test_within_limit_with_native_charts_and_notes():
    assert len(_parts("ppt/slides/slide")) <= 20
    assert len(_parts("ppt/charts/chart")) >= 5
    assert len(_parts("ppt/notesSlides/notesSlide")) == len(_parts("ppt/slides/slide"))


def test_every_slide_carries_the_research_prototype_label():
    layouts = _parts("ppt/slideLayouts/slideLayout")
    labelled = [n for n, x in layouts.items() if "not a diagnostic device" in _text(x)]
    assert len(labelled) >= 3  # CONTENT, CONTENT_PLAIN, CLOSE; the cover states it on the slide
    assert "not a diagnostic device" in _parts("ppt/slides/slide")["ppt/slides/slide1.xml"]


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


def test_no_overclaims(slides):
    notes = [_text(x) for x in _parts("ppt/notesSlides/notesSlide").values()]
    text = " ".join(slides + notes).lower()
    assert not [p for p in OVERCLAIMS if p in text]
    # x10.6 is the augmentation stage; the whole training step is x2.49
    assert not re.search(r"training[^.]{0,40}×10\.6", text)
