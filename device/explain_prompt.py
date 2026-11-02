"""The prompt contract for the on-device explainer: one decision snapshot + an audience -> OpenAI-style messages.

Stdlib only, because it runs on the board. Training and serving both use build_messages(), so what the model was trained on is
what it sees. The system prompt's bytes are the cached prefix on the board: changing a single byte costs a new prewarm and means
any fine-tuned model has to be judged again, so change PROMPT_VERSION with it.

A snapshot is the decision part of the app's /state JSON: status, call {label, prob, probsA, probsB, labelA, labelB,
review_score}, reasons, tissue_fraction. Anything else in /state is ignored, including truth and truth_match, which the LLM must
never see. Statuses that carry no decision (standby, reading, model_paused ...) raise NoDecision.
"""
from __future__ import annotations

import math

PROMPT_VERSION = 1
AUDIENCES = ("doctor", "user")
SITUATIONS = ("agree", "review", "disagree", "no_tissue")
NO_TISSUE_PCT = 10  # twinread_runtime.NO_TISSUE_FRACTION as a percentage

# NCT-CRC-HE class names, as in the dataset's own description (Zenodo 1214456) and quoted in the CONCH paper (lu2024, CRC-100K).
CLASS_NAMES = {
    "ADI": "adipose tissue",
    "BACK": "background",
    "DEB": "debris",
    "LYM": "lymphocytes",
    "MUC": "mucus",
    "MUS": "smooth muscle",
    "NORM": "normal colon mucosa",
    "STR": "cancer-associated stroma",
    "TUM": "colorectal adenocarcinoma epithelium",
}

# What each class is on H&E, and nothing beyond that. One goes in each prompt, so they stay short.
CLASS_CARDS = {
    "ADI": "fat cells, seen as large clear round spaces with thin walls.",
    "BACK": "slide area with little or no tissue on it.",
    "DEB": "fragments of dead cells and broken-down tissue.",
    "LYM": "dense groups of small round immune cells with dark nuclei.",
    "MUC": "pale mucus outside cells, with few cells in it.",
    "MUS": "bundles of long spindle-shaped cells with pink cytoplasm.",
    "NORM": "regular glands lined by orderly epithelium with goblet cells.",
    "STR": "fibrous tissue around cancer, with spindle cells and collagen.",
    "TUM": "irregular glands lined by crowded, abnormal epithelial cells.",
}
NO_TISSUE_CARD = "frame may be blank, dark or off target, so there is nothing to classify."

WHY = {
    "disagree": "the heads chose different classes",
    "js": "the heads' confidence differs",
    "review_head": "the review head is unsure",
}

AUDIENCE_LINE = {
    "doctor": "Audience: a doctor. Use histology terms.",
    "user": "Audience: someone with no medical training. Use plain everyday words.",
}

SYSTEM = (
    "You explain one result from TwinRead, a model that reads one small H&E image tile of colorectal tissue and picks one of nine "
    "tissue classes. It has two heads, Head A and Head B. Each head picks a class with a confidence. The combined call merges "
    "them. A high confidence means the model is sure; it does not mean conflict.\n"
    "You get the facts as short sentences. Rewrite them as one plain paragraph of 50 to 80 words for the audience named, in full "
    "sentences, with no lists, labels or bold. Keep every fact in the same order, including whether the heads agree and whether the "
    "tile is flagged. Copy every number and class exactly. Start with \"The model\". Add no new facts. You cannot see the image, so "
    "say what a class covers, never what the tile shows.\n"
    "Describe this one tile only. Do not mention stage, grade, prognosis, treatment, patients or the whole slide. Give no advice."
)

# Runtime sampling, used by every evaluation arm; see results/llm/finetune/. repeat_penalty stays 1.0: it also penalises digits and
# words in the facts at the end of the prompt, and at T2's 1.1 validation pass rates were lower on every prompt version tried.
SAMPLING = {"temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.0, "max_tokens": 170, "cache_prompt": True}


class NoDecision(ValueError):
    """The snapshot holds no decision to explain (standby, reading, an error, or no call)."""


def pct(x: float) -> int:
    """The dashboard's Math.round(100 * x)."""
    return int(math.floor(100.0 * float(x) + 0.5))


def snapshot_from_result(r, classes, tissue_fraction=None, thresholds=None) -> dict:
    """The /state decision for a TileResult, built the way uno_q_app.step() builds it."""
    probs_a, probs_b = list(r.probsA), list(r.probsB)
    status = "no_tissue" if r.no_tissue else ("review" if r.review else "agree")
    call = {"label": r.label, "prob": float(r.prob), "probsA": [float(p) for p in probs_a], "probsB": [float(p) for p in probs_b],
            "probsF": [float(p) for p in r.probsF], "labelA": classes[probs_a.index(max(probs_a))],
            "labelB": classes[probs_b.index(max(probs_b))], "review_score": float(r.review_score)}
    snap = {"status": status, "call": call, "reasons": list(r.reasons)}
    if tissue_fraction is not None:
        snap["tissue_fraction"] = float(tissue_fraction)
    if thresholds is not None:
        snap["thresholds"] = dict(thresholds)
    return snap


def situation(snapshot: dict) -> str:
    """One of SITUATIONS. Raises NoDecision when the snapshot has no decision."""
    status = snapshot.get("status")
    if status == "no_tissue":
        return "no_tissue"
    call = snapshot.get("call")
    if status not in ("agree", "review") or not isinstance(call, dict):
        raise NoDecision(f"no decision to explain: status={status!r}")
    if status == "agree":
        return "agree"
    if "disagree" in (snapshot.get("reasons") or []) or call.get("labelA") != call.get("labelB"):
        return "disagree"
    return "review"


def _head(code: str, prob: float) -> dict:
    if code not in CLASS_NAMES:
        raise ValueError(f"unknown class {code!r}")
    return {"code": code, "name": CLASS_NAMES[code], "pct": pct(prob)}


def facts_from_snapshot(snapshot: dict) -> dict:
    """The only things the explanation may say, as data. Also what explain_check checks against."""
    sit = situation(snapshot)
    if sit == "no_tissue":
        tf = snapshot.get("tissue_fraction")
        low = tf is None or float(tf) < NO_TISSUE_PCT / 100.0
        tissue_pct = pct(tf) if tf is not None else None
        if low and tissue_pct is not None:  # 9.96 % must not round up to "10% of pixels, below the 10% needed"
            tissue_pct = min(tissue_pct, NO_TISSUE_PCT - 1)
        return {"situation": sit, "status": "no_tissue", "flagged": True, "tissue_pct": tissue_pct,
                "uniform_frame": not low, "fused": None, "head_a": None, "head_b": None, "heads_same": None,
                "fused_differs": False, "reasons": ["no_tissue"], "review_score": None}
    call = snapshot["call"]
    a = _head(call["labelA"], max(call["probsA"]))
    b = _head(call["labelB"], max(call["probsB"]))
    f = _head(call["label"], call["prob"])
    reasons = [r for r in (snapshot.get("reasons") or []) if r in WHY]
    same = a["code"] == b["code"]
    score = round(float(call["review_score"]), 2) if "review_head" in reasons else None
    return {"situation": sit, "status": snapshot["status"], "flagged": snapshot["status"] == "review", "tissue_pct": None,
            "uniform_frame": False, "fused": f, "head_a": a, "head_b": b, "heads_same": same,
            "fused_differs": same and a["code"] != f["code"], "reasons": reasons, "review_score": score}


def _why(facts: dict) -> str:
    # As on the dashboard, a disagreement already implies the confidence differs, so "js" is not repeated next to it.
    rs = [r for r in facts["reasons"] if not (r == "js" and "disagree" in facts["reasons"])]
    parts = []
    for r in rs:
        parts.append(f"{WHY[r]} (score {facts['review_score']:.2f})" if r == "review_head" else WHY[r])
    return " and ".join(parts)


def _head_tag(h: dict, fused: dict) -> str:
    """A class other than the combined call gets its full name too, or the model guesses it (one model called MUS mucus)."""
    return h["code"] if h["code"] == fused["code"] else f"{h['name']} ({h['code']})"


def user_turn(facts: dict, audience: str) -> str:
    """Audience, facts and the one relevant class card. Short on purpose: each uncached token costs about 0.09 s on the board."""
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}")
    lines = [AUDIENCE_LINE[audience], "Facts:"]
    if facts["situation"] == "no_tissue":
        lines.append("The model found too little tissue in this frame, so it gave no class.")
        if facts["uniform_frame"]:
            lines.append("The frame is nearly uniform.")
        elif facts["tissue_pct"] is None:
            lines.append(f"Tissue is below the {NO_TISSUE_PCT}% needed.")
        else:
            lines.append(f"Tissue is {facts['tissue_pct']}% of pixels, below the {NO_TISSUE_PCT}% needed.")
        lines.append(f"The {NO_TISSUE_CARD}")
        return "\n".join(lines)
    f, a, b = facts["fused"], facts["head_a"], facts["head_b"]
    lines.append(f"The model read this tile as {f['name']} ({f['code']}) with a combined confidence of {f['pct']}%.")
    lines.append(f"Head A chose {_head_tag(a, f)} at {a['pct']}%. Head B chose {_head_tag(b, f)} at {b['pct']}%.")
    if facts["fused_differs"]:
        heads = "The heads agree with each other, but the combined call differs from both"
    elif facts["heads_same"]:
        heads = "The heads agree"
    else:
        heads = "The heads disagree"
    # One sentence, because a separate "not flagged" sentence was what the base model most often dropped.
    lines.append(f"{heads}, and the tile is flagged for review because {_why(facts)}." if facts["flagged"]
                 else f"{heads}, and the tile is not flagged for review.")
    lines.append(f"{f['code']} covers {CLASS_CARDS[f['code']]}")
    return "\n".join(lines)


def build_messages(snapshot: dict, audience: str = "doctor") -> list:
    """OpenAI-style messages for one decision. Raises NoDecision if the snapshot has none, ValueError on a bad audience."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_turn(facts_from_snapshot(snapshot), audience)}]


def facts_text(snapshot: dict) -> str:
    """The facts as plain sentences, with no LLM: correct by construction. Show this when an answer fails explain_check."""
    return " ".join(user_turn(facts_from_snapshot(snapshot), "doctor").split("\n")[2:])


def chat_request(snapshot: dict, audience: str = "doctor", seed=None, stream: bool = True) -> dict:
    """The whole POST /v1/chat/completions body."""
    body = {"messages": build_messages(snapshot, audience), "stream": stream, **SAMPLING}
    if seed is not None:
        body["seed"] = int(seed)
    return body


def prewarm_request() -> dict:
    """Send once after the server starts. It caches the system prompt, so the first click is as fast as every later one."""
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ""}], "stream": False, "max_tokens": 1,
            "temperature": 0.0, "cache_prompt": True}
