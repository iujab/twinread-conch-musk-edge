"""The explainer's prompt contract and fact checker: fast, offline, no model."""
import ast
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "device"))
sys.path.insert(0, str(ROOT / "scripts/llm/finetune"))

import explain_check as ec  # noqa: E402
import explain_prompt as ep  # noqa: E402

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def probs(code, p):
    rest = (1.0 - p) / 8
    return [p if c == code else rest for c in CLASSES]


def snap(status="agree", label="TUM", prob=0.97, a=("TUM", 0.99), b=("TUM", 0.98), reasons=(), score=0.02, **extra):
    return {"status": status, "call": {"label": label, "prob": prob, "probsA": probs(*a), "probsB": probs(*b), "probsF": probs(label, prob),
                                       "labelA": a[0], "labelB": b[0], "review_score": score},
            "reasons": list(reasons), "tissue_fraction": 0.9, "thresholds": {"js": 0.04, "review": 0.43}, "classes": CLASSES, **extra}


AGREE = snap()
REVIEW = snap("review", "STR", 0.88, ("STR", 0.91), ("STR", 0.72), ("js", "review_head"), 0.61)
DISAGREE = snap("review", "STR", 0.87, ("MUS", 0.51), ("STR", 0.88), ("disagree", "js"), 0.43)
NO_TISSUE = {"status": "no_tissue", "call": snap()["call"], "reasons": ["no_tissue", "disagree"], "tissue_fraction": 0.07}


# -- the prompt -------------------------------------------------------------------------------------------------------
def test_stdlib_only():
    std = set(sys.stdlib_module_names) | {"__future__"}
    for f in ("explain_prompt.py", "explain_check.py"):
        tree = ast.parse((ROOT / "device" / f).read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            for n in names:
                assert n.split(".")[0] in std | {"explain_prompt", "explain_vocab"}, n


def test_system_prompt_has_no_banned_words():
    low = ep.SYSTEM.lower()
    for bad in ("reader", "research prototype", "diagnostic device", "consult"):
        assert bad not in low
    assert "Head A" in ep.SYSTEM and "Head B" in ep.SYSTEM


def test_messages_shape_and_stable_system():
    for s in (AGREE, REVIEW, DISAGREE, NO_TISSUE):
        for aud in ep.AUDIENCES:
            m = ep.build_messages(s, aud)
            assert [x["role"] for x in m] == ["system", "user"]
            assert m[0]["content"] == ep.SYSTEM


def test_each_situation_is_recognised():
    assert [ep.situation(s) for s in (AGREE, REVIEW, DISAGREE, NO_TISSUE)] == ["agree", "review", "disagree", "no_tissue"]


def test_truth_never_reaches_the_prompt():
    s = snap(truth="ADI", truth_match=False)
    text = json.dumps(ep.build_messages(s, "doctor"))
    assert "ADI" not in text and "truth" not in text.lower()
    assert ep.build_messages(s, "doctor") == ep.build_messages(snap(), "doctor")


def test_no_tissue_prompt_names_no_class_even_though_a_call_exists():
    text = ep.build_messages(NO_TISSUE, "user")[1]["content"]
    for c in CLASSES:
        assert c not in text
    assert "7%" in text and "10%" in text


def test_no_decision_raises():
    for s in ({"status": "standby", "call": None}, {"status": "reading"}, {"status": "error", "error": "x"}, {"status": "agree", "call": None}, {}):
        with pytest.raises(ep.NoDecision):
            ep.build_messages(s)


def test_bad_audience_raises():
    with pytest.raises(ValueError):
        ep.build_messages(AGREE, "nurse")


def test_numbers_are_the_dashboards():
    assert ep.pct(0.995) == 100 and ep.pct(0.5) == 50 and ep.pct(0.125) == 13  # Math.round, not banker's rounding
    f = ep.facts_from_snapshot(DISAGREE)
    assert (f["fused"]["pct"], f["head_a"]["pct"], f["head_b"]["pct"]) == (87, 51, 88)


def test_review_score_given_only_when_the_review_head_fired():
    assert ep.facts_from_snapshot(REVIEW)["review_score"] == 0.61
    assert ep.facts_from_snapshot(DISAGREE)["review_score"] is None
    assert "0.61" in ep.user_turn(ep.facts_from_snapshot(REVIEW), "doctor")
    assert "0.43" not in ep.user_turn(ep.facts_from_snapshot(DISAGREE), "doctor")


def test_every_reason_combination_seen_in_the_data_gives_a_why():
    for reasons in (["review_head"], ["js"], ["js", "review_head"], ["disagree"], ["disagree", "js"], ["disagree", "review_head"], ["disagree", "js", "review_head"]):
        a, b = (("MUS", 0.6), ("STR", 0.7)) if "disagree" in reasons else (("STR", 0.6), ("STR", 0.7))
        turn = ep.user_turn(ep.facts_from_snapshot(snap("review", "STR", 0.7, a, b, reasons, 0.5)), "doctor")
        assert "flagged for review because" in turn


def test_fused_differs_is_stated():
    s = snap("agree", "LYM", 0.52, ("NORM", 0.6), ("NORM", 0.8))
    f = ep.facts_from_snapshot(s)
    assert f["fused_differs"] and "differs" in ep.user_turn(f, "doctor")


def test_requests():
    r = ep.chat_request(AGREE, "user", seed=5)
    assert r["seed"] == 5 and r["max_tokens"] == ep.SAMPLING["max_tokens"] and r["cache_prompt"] is True
    w = ep.prewarm_request()
    assert w["messages"][0]["content"] == ep.SYSTEM and w["messages"][1]["content"] == "" and w["max_tokens"] == 1


def test_snapshot_from_result_matches_the_app_shape():
    class R:
        label, prob, review, review_score, reasons, no_tissue = "STR", 0.87, True, 0.43, ["disagree"], False
        probsA, probsB, probsF = probs("MUS", 0.51), probs("STR", 0.88), probs("STR", 0.87)
    s = ep.snapshot_from_result(R, CLASSES, tissue_fraction=0.8, thresholds={"js": 0.04, "review": 0.43})
    assert s["status"] == "review" and s["call"]["labelA"] == "MUS" and s["call"]["labelB"] == "STR"
    assert ep.situation(s) == "disagree"


# -- the checker ------------------------------------------------------------------------------------------------------
GOOD = {
    "agree": "The model read this tile as colorectal adenocarcinoma epithelium (TUM) with a combined confidence of 97%. Head A chose TUM at 99% and Head B "
             "chose TUM at 98%. The two heads agree. The tile is not flagged for review. TUM covers irregular glands lined by crowded, abnormal epithelial cells.",
    "review": "The model read this tile as cancer-associated stroma (STR) with a combined confidence of 88%. Head A chose STR at 91% and Head B chose STR at 72%. "
              "The two heads agree. The tile is flagged for review because the heads' confidence differs and the review head is unsure (score 0.61). STR covers fibrous tissue around cancer, with "
              "spindle cells and collagen.",
    "disagree": "The model read this tile as cancer-associated stroma (STR) with a combined confidence of 87%. Head A chose smooth muscle (MUS) at 51% and Head B "
                "chose STR at 88%. The two heads disagree. The tile is flagged for review because the heads chose different classes. STR covers fibrous tissue "
                "around cancer, with spindle cells and collagen.",
    "no_tissue": "The model found too little tissue in this frame, so it gave no class. Tissue is 7% of pixels, below the 10% needed. The frame may be blank, dark "
                 "or off target, so there is nothing to classify.",
}
SNAP = {"agree": AGREE, "review": REVIEW, "disagree": DISAGREE, "no_tissue": NO_TISSUE}


@pytest.mark.parametrize("sit", list(GOOD))
def test_a_correct_answer_passes(sit):
    r = ec.check_snapshot(GOOD[sit], SNAP[sit])
    assert r["passed"], r
    assert not r["invented_claim"]


def failed(text, sit):
    r = ec.check_snapshot(text, SNAP[sit])
    return {k for k, v in r["checks"].items() if not v}, r


@pytest.mark.parametrize("old,new,sit,check", [
    ("(TUM)", "(STR)", "agree", "no_wrong_class"),
    ("colorectal adenocarcinoma epithelium (TUM) with", "cancer-associated stroma (STR) with", "agree", "class_correct"),
    ("The two heads agree.", "The two heads disagree.", "agree", "heads_correct"),
    ("The two heads disagree.", "The two heads agree.", "disagree", "heads_correct"),
    ("Head A chose smooth muscle (MUS) at 51% and Head B chose STR at 88%", "Head A chose STR at 88% and Head B chose smooth muscle (MUS) at 51%", "disagree", "heads_correct"),
    ("not flagged for review", "flagged for review", "agree", "status_correct"),
    ("is flagged for review because", "is not flagged for review because", "review", "status_correct"),
    ("97%", "96%", "agree", "numbers_match"),
    ("(score 0.61)", "(score 0.71)", "review", "numbers_match"),
    ("The two heads agree.", "The two heads agree. The patient has stage II disease.", "agree", "no_forbidden"),
    ("TUM covers", "This is a research prototype, consult a doctor. TUM covers", "agree", "no_forbidden"),
    ("Head A chose TUM", "Reader A chose TUM", "agree", "no_forbidden"),
    ("TUM covers", "Smooth muscle may also be present. TUM covers", "agree", "no_wrong_class"),
])
def test_single_faults_are_caught(old, new, sit, check):
    assert old in GOOD[sit]
    bad, _ = failed(GOOD[sit].replace(old, new), sit)
    assert check in bad


def test_t2s_failure_flags_conflict():
    text = GOOD["agree"].replace("The two heads agree.", "The high combined value indicates a conflict between the heads.")
    bad, _ = failed(text, "agree")
    assert "heads_correct" in bad


def test_no_tissue_must_not_name_a_class():
    bad, r = failed(GOOD["no_tissue"] + " The model called it TUM.", "no_tissue")
    assert "no_wrong_class" in bad and r["invented_claim"]


def test_no_tissue_must_not_claim_agreement():
    bad, _ = failed(GOOD["no_tissue"] + " The heads agree.", "no_tissue")
    assert "heads_correct" in bad


@pytest.mark.parametrize("claim,cat", [
    ("The tumour is grade 2.", "grade"), ("This has a poor prognosis.", "prognosis"), ("Chemotherapy is advised.", "treatment"),
    ("The whole slide is cancer.", "whole_slide"), ("This confirms a diagnosis.", "diagnosis"), ("The tile shows crowded glands.", "image_claim"),
])
def test_invented_claims(claim, cat):
    _, r = failed(GOOD["agree"] + " " + claim, "agree")
    assert cat in r["details"]["forbidden"] and r["invented_claim"]


def test_disclaimer_fails_but_is_not_an_invented_claim():
    bad, r = failed(GOOD["agree"].replace("TUM covers", "This is not medical advice; consult a specialist. TUM covers"), "agree")
    assert "no_forbidden" in bad and not r["invented_claim"]


def test_negated_whole_slide_is_allowed():
    r = ec.check_snapshot(GOOD["agree"].replace("TUM covers", "This is not about the whole slide. TUM covers"), AGREE)
    assert "whole_slide" not in r["details"]["forbidden"]


def test_cancer_associated_stroma_does_not_count_as_tum():
    r = ec.check_snapshot(GOOD["review"], REVIEW)
    assert r["details"]["classes_named"] == ["STR"]


def test_length_and_completeness():
    bad, _ = failed("The model read this tile as TUM at 97%.", "agree")
    assert "length_ok" in bad
    bad, _ = failed(GOOD["agree"].rstrip(".") + " and then", "agree")
    assert "complete" in bad


def test_markdown_fails_format():
    bad, _ = failed(GOOD["agree"].replace("TUM covers", "**TUM** covers"), "agree")
    assert "format_ok" in bad


def test_must_be_framed_as_the_models_reading():
    bad, _ = failed(GOOD["agree"].replace("The model read this tile as", "This tile is"), "agree")
    assert "framed_as_model" in bad


def test_equivalent_number_formats_pass():
    r = ec.check_snapshot(GOOD["agree"].replace("97%", "97 percent").replace("at 99%", "at 0.99"), AGREE)
    assert r["checks"]["numbers_match"], r


def test_composer_passes_on_every_situation_and_audience():
    composer = pytest.importorskip("composer")
    rng = random.Random(0)
    for s in (AGREE, REVIEW, DISAGREE, NO_TISSUE, snap("review", "LYM", 0.5, ("NORM", 0.58), ("NORM", 0.8), ("js", "review_head"), 0.71)):
        facts = ep.facts_from_snapshot(s)
        for aud in ep.AUDIENCES:
            for _ in range(25):
                assert ec.check(composer.compose(facts, aud, rng), facts)["passed"]


def test_uniform_frame_and_missing_fraction_still_work():
    composer = pytest.importorskip("composer")
    uniform = {"status": "no_tissue", "call": None, "reasons": ["no_tissue"], "tissue_fraction": 0.4}
    missing = {"status": "no_tissue", "call": None, "reasons": ["no_tissue"]}
    assert "uniform" in ep.user_turn(ep.facts_from_snapshot(uniform), "doctor")
    assert "%" in ep.user_turn(ep.facts_from_snapshot(missing), "doctor")
    rng = random.Random(1)
    for s in (uniform, missing):
        f = ep.facts_from_snapshot(s)
        for aud in ep.AUDIENCES:
            assert ec.check(composer.compose(f, aud, rng), f)["passed"]


def test_real_app_state_shape_is_accepted():
    """A full /state document carries many other keys; only the decision is used."""
    state = {**snap(), "mode": "demo", "mode_reason": "", "source_note": "", "pair": "CM", "heads": ["Head A", "Head B"], "host": {}, "camera": {},
             "frame_label": "Slide", "has_picture": True, "model_error": None, "led": True, "led_state": "agree", "controls": {}, "demo": {"truth": "ADI"},
             "system": {}, "reads": 3, "latency_ms": 20.0, "crop_box": [0, 0, 1, 1], "n": 3}
    assert ep.build_messages(state, "doctor") == ep.build_messages(snap(), "doctor")


def test_prompt_and_checker_handle_empty_text():
    r = ec.check_snapshot("", AGREE)
    assert not r["passed"] and not r["checks"]["length_ok"]


@pytest.mark.parametrize("extra", ["It would be wise to get this looked at soon.", "Oncology follow-up would be sensible.", "A lesion is present.",
                                   "The result can be fully trusted.", "There are ten thousand cells.", "This is a serious finding."])
def test_content_outside_the_vocabulary_is_rejected(extra):
    r = ec.check_snapshot(GOOD["agree"].replace(" TUM covers", f" {extra} TUM covers"), AGREE)
    assert not r["passed"] and not r["checks"]["vocabulary_ok"], r


def test_looks_like_cancer_is_a_diagnosis_claim():
    r = ec.check_snapshot(GOOD["agree"].replace(" TUM covers", " The tile looks like cancer. TUM covers"), AGREE)
    assert not r["passed"] and "diagnosis" in r["details"]["forbidden"], r


def test_wrong_combined_percent_fails_even_if_that_number_appears_elsewhere():
    swapped = GOOD["agree"].replace("confidence of 97%", "confidence of 99%")
    r = ec.check_snapshot(swapped + " Its share was 97%.", AGREE)
    assert not r["passed"] and not r["checks"]["numbers_match"], r
    assert r["details"]["combined_stated"] == 99


def test_every_flag_reason_must_be_given():
    r = ec.check_snapshot(GOOD["review"].replace("the heads' confidence differs and ", ""), REVIEW)
    assert not r["passed"] and r["details"]["reasons_missing"] == ["js"], r
    r = ec.check_snapshot(GOOD["review"].replace(" and the review head is unsure (score 0.61)", ""), REVIEW)
    assert not r["passed"] and r["details"]["reasons_missing"] == ["review_head"], r


def test_fibrous_tissue_of_cancer_describes_stroma_not_tumour():
    r = ec.check_snapshot(GOOD["review"].replace("STR covers fibrous tissue around cancer, with spindle cells and collagen.",
                                                 "STR covers the fibrous tissue of cancer, with spindle cells and collagen."), REVIEW)
    assert r["passed"], r


def test_class_card_mentioning_cancer_is_not_a_diagnosis():
    r = ec.check_snapshot(GOOD["review"].replace("STR covers", "The STR class is"), REVIEW)
    assert "diagnosis" not in r["details"]["forbidden"] and r["passed"], r
    r = ec.check_snapshot(GOOD["review"] + " This is cancer.", REVIEW)
    assert "diagnosis" in r["details"]["forbidden"] and r["invented_claim"]


def test_tissue_just_under_the_limit_does_not_read_as_equal_to_it():
    s = {"status": "no_tissue", "call": None, "reasons": ["no_tissue"], "tissue_fraction": 0.0996}
    assert ep.facts_from_snapshot(s)["tissue_pct"] == 9
    assert "9% of pixels, below the 10% needed" in ep.user_turn(ep.facts_from_snapshot(s), "user")


@pytest.mark.parametrize("sentence", ["The model didn't find any tissue in this frame, so it gave no class.", "The tissue was too little, so the model gave no class."])
def test_no_tissue_wordings_found_in_the_held_out_reading(sentence):
    r = ec.check_snapshot(sentence + " It is 7% of the pixels, under what it needs. The frame may be blank, dark or off target.", NO_TISSUE)
    assert r["checks"]["status_correct"], r


@pytest.mark.parametrize("sit", list(SNAP))
def test_fallback_text_passes_the_checker(sit):
    text = ep.facts_text(SNAP[sit])
    r = ec.check_snapshot(text, SNAP[sit])
    assert r["passed"] or r["checks"]["length_ok"] is False and all(v for k, v in r["checks"].items() if k != "length_ok"), r
    assert "Audience" not in text and "Facts" not in text
