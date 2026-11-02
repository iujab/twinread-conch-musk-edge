#!/usr/bin/env python
"""Rule-based explanation composer: the training targets, and the deterministic control arm of the evaluation.

No LLM wrote any of this and none is called here. Facts slot into hand-written phrasing variants, clause orders and two audience
styles, so every target is correct by construction, and build_dataset.py asserts that each one also passes device/explain_check.py.

compose(facts, audience, rng) returns one paragraph. Card paraphrases say the same thing as explain_prompt.CLASS_CARDS and were
checked against the dataset's class list (Zenodo 1214456) and standard H&E histology.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "device"))
import explain_check  # noqa: E402
import explain_prompt as ep  # noqa: E402

WORDS_LO, WORDS_HI = 50, 80
NT_LO, NT_HI = 25, 60

CARD_PARAPHRASES = {
    "ADI": ["fat cells, seen as large clear round spaces with thin walls",
            "fat cells, which look like large clear round spaces with thin walls",
            "large clear round spaces with thin walls, which are fat cells"],
    "BACK": ["slide area with little or no tissue on it", "an area of the slide with little or no tissue",
             "the part of a slide that holds little or no tissue"],
    "DEB": ["fragments of dead cells and broken-down tissue", "broken-down tissue and fragments of dead cells",
            "leftover fragments of dead cells and broken-down tissue"],
    "LYM": ["dense groups of small round immune cells with dark nuclei", "small round immune cells with dark nuclei, packed in dense groups",
            "tightly packed small round immune cells that have dark nuclei"],
    "MUC": ["pale mucus outside cells, with few cells in it", "pale mucus lying outside cells, with few cells in it",
            "pale material outside cells, holding few cells"],
    "MUS": ["bundles of long spindle-shaped cells with pink cytoplasm", "long spindle-shaped cells with pink cytoplasm, arranged in bundles",
            "bundles of long thin spindle cells with pink cytoplasm"],
    "NORM": ["regular glands lined by orderly epithelium with goblet cells", "orderly epithelium with goblet cells, lining regular glands",
             "regular glands whose lining is orderly epithelium with goblet cells"],
    "STR": ["fibrous tissue around cancer, with spindle cells and collagen", "fibrous tissue with spindle cells and collagen, around cancer",
            "spindle cells and collagen forming fibrous tissue around cancer"],
    "TUM": ["irregular glands lined by crowded, abnormal epithelial cells", "crowded, abnormal epithelial cells lining irregular glands",
            "irregular glands whose lining cells are crowded and abnormal"],
}


# Same content in everyday words, for the user audience.
PLAIN_CARDS = {
    "ADI": ["fat cells, which look like large clear round spaces", "fat cells, seen as large clear round spaces"],
    "BACK": ["a part of the slide with little or no tissue on it", "an empty part of the slide with little or no tissue"],
    "DEB": ["bits of dead cells and broken-down tissue", "broken-down tissue and bits of dead cells"],
    "LYM": ["groups of small round immune cells", "tightly packed small round immune cells"],
    "MUC": ["pale mucus lying outside the cells", "pale slimy material outside the cells, with few cells in it"],
    "MUS": ["bundles of long thin muscle cells", "long thin cells packed into bundles"],
    "NORM": ["neat, regular glands in the lining of the colon", "the orderly glands of the colon lining"],
    "STR": ["fibrous tissue around cancer", "the fibrous tissue found around cancer"],
    "TUM": ["irregular glands made of crowded, abnormal cells", "crowded, abnormal cells forming irregular glands"],
}


def _call(facts, audience, rng):
    f = facts["fused"]
    c, p, name = f["code"], f["pct"], f["name"]
    if audience == "doctor":
        tag = rng.choice([f"{name} ({c})", f"{c} ({name})"])
        return rng.choice([
            f"The model read this tile as {tag} with a combined confidence of {p}%.",
            f"The model read this tile as {tag} at {p}% combined confidence.",
            f"The model's combined call for this tile is {tag} at {p}%.",
            f"The model's combined call on this tile is {tag}, with {p}% confidence.",
            f"The model read this one tile as {tag}, and the combined call carries {p}% confidence.",
            f"The model called this tile {tag}, with a combined confidence of {p}%.",
        ])
    return rng.choice([
        f"The model looked at this small square of tissue and read it as {name} ({c}), with a combined confidence of {p}%.",
        f"The model read this tile, a small square of a tissue image, as {name} ({c}), with a combined confidence of {p}%.",
        f"The model's combined call for this small tile of tissue is {name} ({c}), with {p}% confidence.",
        f"The model read this tile as {name} ({c}). Its combined confidence in that call is {p}%.",
        f"The model looked at one small tile of tissue and read it as {name}, labelled {c}, with a combined confidence of {p}%.",
    ])


def _head_tag(h, facts, audience):
    if h["code"] == facts["fused"]["code"]:
        return h["code"]
    return f"{h['name']} ({h['code']})"


def _heads(facts, audience, rng, brief=False):
    a, b = facts["head_a"], facts["head_b"]
    at, bt = _head_tag(a, facts, audience), _head_tag(b, facts, audience)
    ap, bp = a["pct"], b["pct"]
    out = [f"Head A chose {at} at {ap}% and Head B chose {bt} at {bp}%.",
           f"Head A read {at} at {ap}%, and Head B read {bt} at {bp}%.",
           f"Head A gives {at} {ap}% confidence, and Head B gives {bt} {bp}%.",
           f"Head A says {at} with {ap}% confidence; Head B says {bt} with {bp}%.",
           f"Head A picked {at} at {ap}%, while Head B picked {bt} at {bp}%."]
    if a["code"] == b["code"]:
        out += [f"Both heads chose {at}: Head A at {ap}% and Head B at {bp}%.",
                f"Both heads picked {at}, Head A with {ap}% confidence and Head B with {bp}%."]
    s = rng.choice(out)
    if audience == "user" and not brief:
        s = rng.choice(["The model has two heads that each give their own answer. ",
                        "Two parts of the model, its heads, answer separately. ",
                        "The model has two heads, and each answers on its own. "]) + s
    return s


def _agreement(facts, rng):
    if facts["fused_differs"]:
        return rng.choice(["The heads agree with each other, but the combined call differs from both.",
                           "Head A and Head B agree, yet the combined call differs from both of them.",
                           "The heads agree on the class, and the combined call is different from theirs."])
    if facts["heads_same"]:
        return rng.choice(["The two heads agree.", "The heads agree.", "Head A and Head B agree on the class.",
                           "The heads picked the same class, so they agree.", "The two heads agree on this tile."])
    return rng.choice(["The two heads disagree.", "The heads disagree.", "Head A and Head B disagree on the class.",
                       "The heads chose different classes, so they disagree.", "The two heads disagree on this tile."])


def _why_text(facts, rng):
    rs = facts["reasons"]
    parts = []
    if "disagree" in rs:
        parts.append(rng.choice(["the heads chose different classes", "the heads disagree on the class"]))
    elif "js" in rs:
        parts.append(rng.choice(["the heads' confidence differs", "the confidence of the two heads differs",
                                 "the heads' confidence levels differ"]))
    if "review_head" in rs:
        s = facts["review_score"]
        parts.append(rng.choice([f"the review head is unsure (score {s:.2f})", f"the review head is unsure, with a score of {s:.2f}",
                                 f"the review score of {s:.2f} shows the review head is unsure"]))
    return " and ".join(parts)


def _merged(facts, rng):
    """Agreement and flag in one sentence, as in the facts."""
    if facts["fused_differs"]:
        heads = "The heads agree with each other, but the combined call differs from both"
    elif facts["heads_same"]:
        heads = rng.choice(["The heads agree", "The two heads agree", "Head A and Head B agree"])
    else:
        heads = rng.choice(["The heads disagree", "The two heads disagree", "Head A and Head B disagree"])
    if facts["flagged"]:
        return f"{heads}, and the tile is flagged for review because {_why_text(facts, rng)}."
    return f"{heads}, and the tile is not flagged for review."


def _status(facts, audience, rng):
    if not facts["flagged"]:
        return rng.choice(["The tile is not flagged for review.", "It is not flagged for review.",
                           "No review flag was raised for this tile.", "The model did not flag this tile for review.",
                           "This tile is not flagged, so no second look is requested."])
    why = _why_text(facts, rng)
    return rng.choice([f"The tile is flagged for review because {why}.", f"It is flagged for review because {why}.",
                       f"This tile is flagged for a second look because {why}.", f"The tile is flagged for review: {why}.",
                       f"The model flagged this tile for review because {why}."])


def _card(facts, audience, rng):
    f = facts["fused"]
    code, name = f["code"], f["name"]
    card = rng.choice(CARD_PARAPHRASES[code])
    if audience == "doctor":
        return rng.choice([f"{code} covers {card}.", f"The {code} class covers {card}.", f"On H&E, {code} means {card}.",
                           f"The {code} class is {card}.", f"On H&E, the {code} class means {card}."])
    card = rng.choice(PLAIN_CARDS[code])
    label = name[0].upper() + name[1:]
    return rng.choice([f"{label} means {card}.", f"In a tissue image, {name} means {card}.", f"In plain words, {name} means {card}.",
                       f"{code} covers {card}."])


def _extras(facts, audience):
    out = ["This is the model's reading of one tile.", "The reading covers this one tile only.",
           "This describes one tile as the model read it."]
    if audience == "user":
        out += ["Confidence here shows how sure the model is, not whether it is right.",
                "A tile is one small square cut from a tissue image.",
                "Confidence shows how sure the model is about its answer."]
    if facts["flagged"]:
        out += ["The flag asks a person to take a second look at this tile."]
    return out


def _no_tissue(facts, audience, rng):
    t = facts["tissue_pct"]
    if facts["uniform_frame"]:
        why = rng.choice(["The frame is nearly uniform.", "This frame is nearly uniform, with almost nothing in it."])
    elif t is None:
        why = "Tissue is below the 10% needed."
    elif audience == "doctor":
        why = rng.choice([f"Tissue is {t}% of pixels, below the 10% needed.", f"Just {t}% of pixels pass the tissue test, below the 10% needed.",
                          f"Stained tissue covers {t}% of pixels, and the model needs 10%."])
    else:
        why = rng.choice([f"Stained tissue is {t}% of the pixels, and it needs 10%.",
                          f"Just {t}% of the picture looks like stained tissue, below the 10% it needs.",
                          f"Stained tissue fills {t}% of the picture, below the 10% it needs."])
    if audience == "doctor":
        first = rng.choice(["The model found too little tissue in this frame to read it.", "The model has no reading here: there is too little tissue in this frame.",
                            "The model found too little tissue in this frame."])
        then = rng.choice(["So no class was given.", "It therefore gave no class.", "So the model named no class."])
        last = rng.choice(["The frame may be blank, dark or off target.", "The frame may be blank, dark or off target, so there is nothing to classify.",
                           "It may be blank, dark or off target, which leaves nothing to classify."])
    else:
        first = rng.choice(["The model could not read this picture because there is too little tissue in it.",
                            "The model found too little tissue in this picture to read it.",
                            "The model found too little tissue here to read."])
        then = rng.choice(["So it gave no class.", "That means no class was given.", "So the model named no class at all."])
        last = rng.choice(["The picture may be blank, dark or pointed away from the tissue.", "The picture may be blank, dark or off target, so there is nothing to classify.",
                           "It may be blank, dark or off target, which leaves nothing to classify."])
    return " ".join([first, why, then, last])


def compose(facts: dict, audience: str, rng: random.Random, tries: int = 400) -> str:
    """One paragraph that passes explain_check. Raises if none is found, which would be a bug in the variants.
    Results with many facts and long class names may not fit: after half the tries the head introduction is dropped and 85 is allowed."""
    for attempt in range(tries):
        brief = attempt > tries // 2
        if facts["situation"] == "no_tissue":
            text = _no_tissue(facts, audience, rng)
            lo, hi = NT_LO, NT_HI
        else:
            call, heads, agree, status, card = (_call(facts, audience, rng), _heads(facts, audience, rng, brief), _agreement(facts, rng),
                                                _status(facts, audience, rng), _card(facts, audience, rng))
            order = rng.choice([
                [call, heads, agree, status, card], [call, heads, agree, status, card], [call, heads, agree, status, card],
                [call, heads, _merged(facts, rng), card], [call, heads, _merged(facts, rng), card], [call, heads, status, agree, card],
                [call, heads, agree, card, status],
            ])
            parts = list(order)
            extras = _extras(facts, audience)
            rng.shuffle(extras)
            while len(" ".join(parts).split()) < rng.randint(WORDS_LO, WORDS_LO + 12) and extras:
                parts.append(extras.pop())
            text = " ".join(parts)
            lo, hi = WORDS_LO, WORDS_HI + (5 if brief else 0)
        if lo <= len(text.split()) <= hi and explain_check.check(text, facts)["passed"]:
            return text
    raise RuntimeError(f"composer found no passing text for {facts}: last was {text!r}: {explain_check.check(text, facts)}")
