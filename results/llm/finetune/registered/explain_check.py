"""Deterministic check of a generated explanation against the facts it was given. Stdlib only, fast enough to run per answer.

check(text, facts) takes facts from explain_prompt.facts_from_snapshot and returns:
  passed          every check below holds
  checks          class_correct, status_correct, heads_correct, numbers_match, no_forbidden, no_wrong_class, length_ok, complete,
                  format_ok, framed_as_model
  invented_claim  the text asserts something outside the facts: stage, grade, prognosis, treatment, patient, whole-slide or
                  diagnosis claims, what the image shows, a class the result does not contain, or a number it does not contain
  details         what was found, for reading a failure

This is pattern matching: it catches wrong facts and banned topics in the phrasings it knows and can miss or misjudge a phrasing it
does not. results/llm/finetune/ reports how often it disagreed with a human read and with a blind judge.
"""
from __future__ import annotations

import re

LEN_RANGE = {"no_tissue": (15, 70), "other": (40, 90)}
ALLOWED_BARE = {1, 2, 9}  # "2 heads", "9 classes", "1 tile"

_I = re.IGNORECASE
_ALIASES = {
    "ADI": [r"\badipose\b", r"\bfat\b", r"\bfatty\b", r"\badipocytes?\b"],
    "BACK": [r"\bbackground\b"],
    "DEB": [r"\bdebris\b", r"\bnecro\w+"],
    "LYM": [r"\blymphocyt\w+", r"\blymphoid\b"],
    "MUC": [r"\bmucus\b", r"\bmucin\w*", r"\bmucous\b"],
    "MUS": [r"\bmuscle\b", r"\bmyocytes?\b"],
    "NORM": [r"\bnormal\b", r"\bmucosa\b"],
    "STR": [r"\bstroma\w*"],
    "TUM": [r"\btumou?rs?\b", r"\badenocarcinoma\b", r"\bcarcinoma\b", r"\bcancers?\b", r"\bcancerous\b"],
}
_CLASS_RX = {c: [re.compile(rf"\b{c}\b")] + [re.compile(p, _I) for p in ps] for c, ps in _ALIASES.items()}
# Phrases that name a class but contain another class's word, removed before looking for class mentions.
_STR_PHRASES = re.compile(r"\b(cancer|tumou?r)[- ]associated\b|\btumou?r stroma\b|\baround (the )?(cancer|tumou?r)\b|"
                          r"\bfrom cancer\b", _I)
_TUM_PHRASE = re.compile(r"\btumou?r stroma\b", _I)

_FORBIDDEN = {
    "stage": [r"\bstag(e|es|ed|ing)\b", r"\bdukes\b", r"\bT[1-4]\s?N[0-3]\b", r"\bTNM\b"],
    "grade": [r"\bgrad(e|es|ed|ing)\b", r"\b(well|moderately|poorly)[- ]differentiated\b"],
    "prognosis": [r"\bprognos\w*", r"\bsurvival\b", r"\boutcomes?\b", r"\bmetasta\w*", r"\brecurr\w*", r"\baggressive\b",
                  r"\blife expectancy\b", r"\binvasi\w*"],
    "treatment": [r"\btreat\w*", r"\bchemo\w*", r"\bradiotherapy\b", r"\bradiation\b", r"\bsurg\w*", r"\bresect\w*", r"\btherap\w*",
                  r"\bmedications?\b", r"\bdrugs?\b", r"\bbiops\w*", r"\bcolonoscop\w*", r"\bscreening\b"],
    "patient": [r"\bpatients?\b", r"\b(he|she|his|her)\b", r"\byour (body|health|colon|doctor|results?|condition)\b", r"\byou have\b",
                r"\bsymptoms?\b", r"\bmedical history\b", r"\bfamily history\b", r"\b\d+[- ]year[- ]old\b"],
    "whole_slide": [r"\b(whole|entire|full)\s+(slide|specimen|sample|biopsy|case)\b",
                    r"\bthe (slide|specimen|biopsy|sample|case)\s+(is|shows|contains|has|was)\b", r"\bacross the slide\b"],
    "diagnosis": [r"\bdiagnos\w*", r"\bmalignan\w*", r"\bbenign\b", r"\bcancerous\b",
                  r"\b(has|have|having|is|confirms?|indicates?|suggests?)\b[^.;]{0,30}\bcancer\b(?![- ]associated)"],
    "disclaimer": [r"\bprototype\b", r"\bdiagnostic device\b", r"\bmedical advice\b", r"\bconsult\w*", r"\bhealthcare\b",
                   r"\bseek\b", r"\b(see|ask|talk to|speak (to|with)) (a|your) (doctor|physician|pathologist|specialist)\b",
                   r"\bnot a substitute\b", r"\bfor (research|educational|informational) purposes\b", r"\bnot intended\b",
                   r"\bdisclaimer\b"],
    "reader": [r"\breaders?\b"],
    # The LLM never sees the image, so it can only report the model's call and what a class is in general.
    "image_claim": [r"\b(the|this)\s+(tile|image|tissue|frame|sample|slide|section|picture)\s+(clearly\s+)?(shows?|showed|contains?|contained|displays?|"
                    r"displayed|features?|reveals?|revealed|depicts?|consists?\s+of|is\s+composed\s+of|has|had)\b"],
}
_FORBIDDEN_RX = {k: [re.compile(p, _I) for p in ps] for k, ps in _FORBIDDEN.items()}
_INVENTED = ("stage", "grade", "prognosis", "treatment", "patient", "whole_slide", "diagnosis", "image_claim")
_NEGATED = re.compile(r"\b(not|no|never|rather than|without|beyond)\b[^.;]{0,25}$", _I)

_DISAGREE = re.compile(
    r"\bdisagre\w*|\b(do|does|did|could|can)\s+not\s+agree\b|\b(don'?t|doesn'?t|didn'?t)\s+agree\b|\bnot\s+agree\w*|"
    r"\bno\s+agreement\b|\black\s+of\s+agreement\b|\bneither\s+head\s+agrees?\b|\bdifferent\s+(class(es)?|tissues?|calls?|labels?|answers?)\b|"
    r"\bdiffer(s|ed)?\s+(on|about|in|over)\s+(the\s+)?(class|tissue|call|label)\w*|\bsplit\b|\bconflict\w*|\bnot\s+the\s+same\b|"
    r"\b(do|does)\s+not\s+match\b|\bmismatch\w*|\bat\s+odds\b", _I)
_NOT_DISAGREE = re.compile(r"\b(do|does|did)\s+not\s+disagree\b|\bno\s+disagreement\b", _I)
_AGREE = re.compile(
    r"\bagree(s|d|ing|ment)?\b|\bhead\s+[ab]\s+also\s+\w+|\bsame\s+(class|tissue|call|label|answer)\b|\bboth\s+(heads\s+)?(chose|choose|picked|pick|named|name|read|"
    r"called|call|gave|give|landed|selected)\b|\bconcur\w*|\bmatch(es|ed)?\b|\bunanimous\b", _I)

_NOT_FLAGGED = re.compile(
    r"\b(did|does|do)\s+not\s+flag\b(\s+\w+){0,3}?\s+for\s+(a\s+)?(review|second\s+look)\b|\b(did|does|do)\s+not\s+flag\b|"
    r"\b(is|was|were|been)?\s*not\s+(been\s+)?flagged\b(\s+for\s+(a\s+)?(review|second\s+look)\b)?|"
    r"\b(isn'?t|wasn'?t)\s+flagged\b(\s+for\s+(a\s+)?(review|second\s+look)\b)?|\bno\s+(review\s+)?flags?\b|\bunflagged\b|"
    r"\bno\s+(need\s+for\s+|call\s+for\s+)?(a\s+)?(review|second\s+look)\b|\bnot\s+(need\w*|require\w*)\s+(a\s+)?(review|second\s+look)\b|"
    r"\bdoes\s+not\s+(need|ask\s+for|call\s+for|require)\b[^.;]{0,20}\b(review|look)\b|\bnot\s+(sent|marked|held)\s+for\b|"
    r"\bneither\s+head\s+asks\s+for\b|\bwithout\s+(a\s+)?(flag|review)\b|\braised?\s+no\b|\bno\s+reason\s+to\s+(flag|review)\b|"
    r"\bnot\s+(up\s+)?for\s+review\b", _I)
_FLAGGED = re.compile(
    r"\bflag(ged|s|ging)?\b|\bfor\s+(a\s+)?(human\s+|further\s+|manual\s+|expert\s+)?review\b|\bsecond\s+look\b|\bshould\s+be\s+(reviewed|checked)\b|"
    r"\bneeds?\s+(a\s+)?(human\s+|further\s+|manual\s+|expert\s+)?(review|look)\b", _I)
_REVIEW_HEAD = re.compile(r"\breview\s+(head|score)\b", _I)
_JS = re.compile(r"\bconfidence\b[^.;]{0,40}\b(differ\w*|diverg\w*|gap|apart)\b|\b(differ\w*|diverg\w*)\b[^.;]{0,40}\bconfidence\b|"
                 r"\bdifferent\s+confidence\w*|\bconfidence\s+levels?\s+differ\w*", _I)
_NO_TISSUE = re.compile(r"\b(no|too\s+little|(not|n't)\s+(find\s+|have\s+)?enough|little|insufficient|without)\s+(\w+\s+)?tissue\b|\bnearly\s+uniform\b|"
                        r"\bbelow\s+the\s+10\b", _I)
_NO_CLASS = re.compile(r"\bno\s+(tissue\s+)?(class|call|reading|label)\b|\bno\s+class\b|\bnot\s+(be\s+)?(classif\w+|read|named|given)\b|"
                       r"\bnothing\s+to\s+(read|classify)\b|\b(could|can|did)\s*(not|n'?t)\s+(read|classify|name|give)\b|"
                       r"\bname[sd]?\s+no\b|\bgave\s+no\b|\bgives\s+no\b|\bwithout\s+(a\s+|any\s+)?class\b", _I)
_CALL_MARKER = re.compile(r"\bread\s+it\s+as\b|\bread\s+(this|the)\s+(\w+\s+)?(tile|image|frame)\b[^.;]{0,50}?\bas\b|\bcombined\s+(call|reading|result|answer|class)\b|"
                          r"\bclassif\w+\s+(this|the)\s+(\w+\s+)?tile\s+as\b|\b(tile|image)\s+was\s+(read|called|classified|labell?ed)\s+as\b", _I)
_HEAD = re.compile(r"\bhead\s+([ab])\b", _I)
_NUM = re.compile(r"(?<![\w.,])(\d+(?:\.\d+)?)(\s*(?:%|percent\b|per\s+cent\b))?(?![\w%])")
_MORE = re.compile(r"\bhead\s+([ab])\b[^.;,]{0,40}?\b(more\s+confident|more\s+sure|higher\s+confidence|less\s+confident|less\s+sure|lower\s+confidence)\b", _I)
_FORMAT = re.compile(r"\*\*|^\s*[-*•]\s|^\s*\d+[.)]\s|^#+\s|</?think>|<\|", re.M)
_SENT = re.compile(r"(?<=[.!?;])\s+")


def _mentions(text: str) -> dict:
    """{class: [start offsets]} of every class mentioned, with phrases like "cancer-associated" neutralised."""
    t = _STR_PHRASES.sub(lambda m: "STR" if "stroma" in m.group(0).lower() else " " * len(m.group(0)), text)
    out = {}
    for c, rxs in _CLASS_RX.items():
        pos = sorted(m.start() for rx in rxs for m in rx.finditer(t))
        if pos:
            out[c] = pos
    return out


def _first_class(text: str):
    m = _mentions(text)
    return min(m, key=lambda c: m[c][0]) if m else None


def _nums(text: str) -> list:
    """(value, raw string, followed by a percent sign or word) for each number."""
    return [(float(m.group(1)), m.group(1), bool(m.group(2))) for m in _NUM.finditer(text)]


def _as_pct(value: float, raw: str, has_pct: bool):
    """The number as a percentage if it reads as one: 97%, 97 percent or 0.97. A bare 97 counts too, a bare 1 does not."""
    if has_pct:
        return value
    if "." in raw:
        return round(value * 100, 6) if value <= 1 else None
    return value if value not in ALLOWED_BARE else None


def _value_ok(value, raw, has_pct, pcts, score):
    if not has_pct and "." in raw and score is not None and abs(value - score) < 0.0051:
        return True
    if not has_pct and "." not in raw and value in ALLOWED_BARE:
        return True
    return _as_pct(value, raw, has_pct) in pcts


def _sentences(text: str) -> list:
    return [s for s in _SENT.split(text) if s.strip()]


def _head_issues(text: str, facts: dict) -> list:
    """Class or number attached to the wrong head within a sentence. Only claims it can attribute; silence is not an issue here."""
    a, b = facts["head_a"], facts["head_b"]
    heads = {"a": a, "b": b}
    issues = []
    for sent in _sentences(text):
        marks = list(_HEAD.finditer(sent))
        i = 0
        while i < len(marks):
            group = [marks[i].group(1).lower()]
            j = i
            # "Head A and Head B ...": the claim that follows belongs to both.
            while j + 1 < len(marks) and re.fullmatch(r"\s*(,|and|&|as well as|plus)\s*", sent[marks[j].end():marks[j + 1].start()], _I):
                j += 1
                group.append(marks[j].group(1).lower())
            end = marks[j + 1].start() if j + 1 < len(marks) else len(sent)
            seg = sent[marks[j].end():end]
            # A combined-call clause later in the segment is not about this head.
            cut = re.search(r"\b(combined|overall|fused)\b", seg, _I)
            if cut:
                seg = seg[:cut.start()]
            cls = _first_class(seg)
            if cls is not None and any(heads[h]["code"] != cls for h in group):
                issues.append(f"class {cls} attached to head {'+'.join(group).upper()}")
            score_ = facts.get("review_score")
            pcts = [_as_pct(v, r, p) for v, r, p in _nums(seg)
                    if not (score_ is not None and not p and "." in r and abs(v - score_) < 0.0051)]
            pcts = [x for x in pcts if x is not None]
            if pcts:
                if len(group) == 1:
                    want = heads[group[0]]["pct"]
                    if pcts[0] != want:
                        issues.append(f"{pcts[0]:g}% attached to head {group[0].upper()}, expected {want}%")
                else:
                    want = [heads[h]["pct"] for h in group]
                    got = pcts[:len(want)]
                    if len(got) == 1 and len(set(want)) > 1:
                        issues.append(f"{got[0]:g}% attached to both heads, expected {want}")
                    elif len(got) == len(want) and got != want:
                        issues.append(f"{got} attached to heads {group}, expected {want}")
                    elif len(got) == 1 and got[0] != want[0]:
                        issues.append(f"{got[0]:g}% attached to both heads, expected {want}")
            i = j + 1
    for m in _MORE.finditer(text):
        h, phrase = m.group(1).lower(), m.group(2).lower()
        other = "b" if h == "a" else "a"
        more = phrase.startswith(("more", "higher"))
        if (more and heads[h]["pct"] <= heads[other]["pct"]) or (not more and heads[h]["pct"] >= heads[other]["pct"]):
            issues.append(f"head {h.upper()} called {phrase}: {heads[h]['pct']}% against {heads[other]['pct']}%")
    return issues


def check(text: str, facts: dict) -> dict:
    t = (text or "").strip().replace("’", "'").replace("‐", "-").replace("‑", "-").replace("–", "-")
    words = len(t.split())
    sit = facts["situation"]
    details = {"words": words}

    # Forbidden claims
    hits = {}
    for cat, rxs in _FORBIDDEN_RX.items():
        for rx in rxs:
            for m in rx.finditer(t):
                if cat == "whole_slide" and _NEGATED.search(t[max(0, m.start() - 40):m.start()]):
                    continue
                hits.setdefault(cat, []).append(m.group(0))
    details["forbidden"] = hits

    # Classes
    ment = _mentions(t)
    details["classes_named"] = sorted(ment)
    if sit == "no_tissue":
        allowed, class_ok = set(), not ment
    else:
        f, a, b = facts["fused"], facts["head_a"], facts["head_b"]
        allowed = {f["code"], a["code"], b["code"]}
        class_ok = f["code"] in ment
        if class_ok:  # the sentence that states the call must state the fused class
            for sent in _sentences(t):
                m = _CALL_MARKER.search(sent)
                if m:
                    first = _first_class(sent[m.end():]) or _first_class(sent[:m.start()])
                    if first is not None and first != f["code"]:
                        class_ok = False
                        details["call_stated_as"] = first
                    break
            else:
                first = min(ment, key=lambda c: ment[c][0])
                if first != f["code"]:
                    class_ok = False
                    details["first_class_named"] = first
    wrong = sorted(set(ment) - allowed)
    details["wrong_classes"] = wrong

    # Status
    unflagged_text = _NOT_FLAGGED.sub(" ", t)
    said_not = bool(_NOT_FLAGGED.search(t))
    said_flag = bool(_FLAGGED.search(unflagged_text))
    if sit == "no_tissue":
        status_ok = bool(_NO_TISSUE.search(t)) and bool(_NO_CLASS.search(t))
    elif facts["flagged"]:
        status_ok = said_flag and not said_not
        rs = facts["reasons"]
        cited = {"review_head": bool(_REVIEW_HEAD.search(t)), "js": bool(_JS.search(t)), "disagree": bool(_DISAGREE.search(t))}
        details["reasons_cited"] = [k for k, v in cited.items() if v]
        if not any(cited[r] for r in rs):
            status_ok = False
        if cited["review_head"] and "review_head" not in rs:
            status_ok = False
    else:
        status_ok = said_not and not said_flag

    # Heads
    no_dis = _NOT_DISAGREE.sub(" agree ", t)
    said_dis = bool(_DISAGREE.search(no_dis))
    said_agree = bool(_AGREE.search(_DISAGREE.sub(" ", no_dis)))
    issues = []
    if sit == "no_tissue":
        heads_ok = not said_dis and not re.search(r"\bagree\w*", t, _I)
    else:
        if facts["heads_same"]:
            heads_ok = said_agree and not said_dis
        else:
            heads_ok = said_dis and not said_agree
        issues = _head_issues(t, facts)
        heads_ok = heads_ok and not issues
    details["head_issues"] = issues

    # Numbers
    found = _nums(t)
    if sit == "no_tissue":
        pcts = {10} | ({facts["tissue_pct"]} if facts.get("tissue_pct") is not None else set())
        required = [facts["tissue_pct"]] if facts.get("tissue_pct") is not None and not facts.get("uniform_frame") else []
        score = None
    else:
        pcts = {f["pct"], a["pct"], b["pct"]}
        required = [f["pct"], a["pct"], b["pct"]]
        score = facts.get("review_score")
    unknown = [r for v, r, p in found if not _value_ok(v, r, p, pcts, score)]
    said = {_as_pct(v, r, p) for v, r, p in found}
    missing = [r for r in required if r not in said]
    details["unknown_numbers"], details["missing_numbers"] = unknown, missing

    lo, hi = LEN_RANGE["no_tissue" if sit == "no_tissue" else "other"]
    checks = {
        "class_correct": bool(class_ok),
        "status_correct": bool(status_ok),
        "heads_correct": bool(heads_ok),
        "numbers_match": not unknown and not missing,
        "no_forbidden": not hits,
        "no_wrong_class": not wrong,
        "length_ok": lo <= words <= hi,
        "complete": t[-1:] in (".", "!", "?"),
        "format_ok": not _FORMAT.search(t) and "\n\n" not in t,
        "framed_as_model": bool(re.search(r"\bmodel\b|\bTwinRead\b", t)),
    }
    invented = any(c in hits for c in _INVENTED) or bool(wrong) or bool(unknown)
    return {"passed": all(checks.values()), "checks": checks, "invented_claim": invented, "details": details}


def check_snapshot(text: str, snapshot: dict) -> dict:
    """check() for a /state snapshot. Raises explain_prompt.NoDecision if it holds no decision."""
    import explain_prompt
    return check(text, explain_prompt.facts_from_snapshot(snapshot))
