"""Task 18: judge-facing deliverables (project.json, README, figures, deck outline, dashboard, headline numbers)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("slug", "title", "tagline", "track", "description", "highlights", "models", "datasets", "images", "hardware",
        "how_to_run", "demo_port", "event_scaleup", "status", "limitations", "metrics")
TRACKS = {"pathology-core", "spine-evidence", "hardware", "crossover"}
NUM = re.compile(r"(?<![\w.])[-+−]?(\d+\.\d+)")


def _pj():
    return json.loads((ROOT / "project.json").read_text())


def _json_numbers():
    out = []

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            out.append(abs(float(o)))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for p in (ROOT / "results").rglob("*.json"):
        walk(json.loads(p.read_text()))
    return np.unique(np.array([x for x in out if np.isfinite(x)]))


def _traceable(text_number: str, nums: np.ndarray) -> bool:
    """`text_number` (e.g. '0.9555', '80.3', '1.11') equals some results number rounded to the same decimals,
    either as stored or as a percentage (x100)."""
    v, d = float(text_number), len(text_number.split(".")[1])
    tol = 0.5 * 10 ** -d + 1e-9
    return bool(np.any(np.abs(nums - v) <= tol) or np.any(np.abs(100 * nums - v) <= tol))


def test_project_json_schema():
    pj = _pj()
    for k in KEYS:
        assert k in pj, k
    assert pj["slug"] == ROOT.name
    assert pj["track"] in TRACKS
    assert pj["demo_port"] == 1777
    assert pj["status"] in ("complete", "partial")
    assert 3 <= len(pj["highlights"]) <= 6
    assert isinstance(pj["metrics"], dict) and pj["metrics"]
    assert isinstance(pj["limitations"], list) and pj["limitations"]
    assert 2 <= len(pj["images"]) <= 8
    for im in pj["images"]:
        assert (ROOT / im["path"]).exists(), im["path"]
        assert im["caption"].strip()
    lay = pj["lay"]
    for k in ("what", "why", "impressive"):
        n = len(lay[k].split())
        assert 25 <= n <= 55, (k, n)


def test_numbers_traceable():
    pj = _pj()
    nums = _json_numbers()
    texts = list(pj["highlights"]) + [f"{k} {v}" for k, v in pj["metrics"].items()]
    missing = [(m, t[:60]) for t in texts for m in NUM.findall(str(t)) if not _traceable(m, nums)]
    assert not missing, missing


def test_traceable_rejects_unknown_number():
    nums = np.array([0.9555, 0.0074])
    assert _traceable("0.9555", nums) and _traceable("0.74", nums) and _traceable("95.5", nums)
    assert not _traceable("0.97", nums)


def test_readme_has_disclaimer_and_abstract():
    txt = (ROOT / "README.md").read_text()
    low = txt.lower()
    assert "research prototype" in low and "not a diagnostic device" in low
    m = re.search(r"^## Abstract\s*\n(.*?)(?=^## )", txt, re.S | re.M)
    assert m, "README needs an '## Abstract' section before the next '## ' heading"
    words = len(m.group(1).split())
    assert 150 <= words <= 260, words
    assert txt.index("## Abstract") < 1200, "the abstract opens the README"
    for word in ("pending", "SIMULATION", "CC-BY-NC-ND"):
        assert word.lower() in low, word


def test_figures_fit_bench_rules():
    from PIL import Image
    pngs = sorted((ROOT / "figures").glob("*.png"))
    assert 2 <= len(pngs) <= 8, [p.name for p in pngs]
    for p in pngs:
        w, _ = Image.open(p).size
        assert 1200 <= w <= 1600, (p.name, w)
    listed = {im["path"] for im in _pj()["images"]}
    assert listed == {f"figures/{p.name}" for p in pngs}


def test_deck_outline_at_most_20_slides():
    txt = (ROOT / "docs" / "deck_outline.md").read_text()
    slides = re.findall(r"^### Slide (\d+)", txt, re.M)
    assert 8 <= len(slides) <= 20
    for word in ("aims", "methods", "results", "challenges", "references"):
        assert word in txt.lower(), word
    for paper in ("lu2024_conch", "xiang2025_musk"):
        assert paper in txt


def test_headline_matches_sources():
    from twinread import headline
    on_disk = json.loads((ROOT / "results" / "headline.json").read_text())
    assert on_disk == json.loads(json.dumps(headline.build(ROOT / "results"), default=str))
    assert on_disk["student"]["CM"]["int8_below_every_teacher"] is True
    assert on_disk["device"]["uno_q_latency"].startswith("pending")


def test_caught_curve_matches_errors_caught():
    from twinread import review
    from twinread.tilepreds import caught_curve
    rng = np.random.default_rng(0)
    s, c = rng.random(503), rng.random(503) > 0.1
    s[:50] = 0.5  # ties exercise the stable ordering
    curve = dict(zip((0.05, 0.1, 0.2), caught_curve(s, c, (0.05, 0.1, 0.2))))
    ref = review.errors_caught(s, c)
    for b in (0.05, 0.1, 0.2):
        assert curve[b] == pytest.approx(ref[b]["errors_caught"], abs=1e-12)
    full = caught_curve(s, c, (0.0, 1.0))
    assert full == [0.0, 1.0]


def test_triage_curves_verified():
    for ds in ("crc", "cam"):
        doc = json.loads((ROOT / "results/cm_local/CM" / ds / "triage_curves.json").read_text())
        assert doc["verification"]["all_ok"] and doc["post_hoc"]
        assert len(doc["review_fraction"]) == len(doc["curves"]["s4"]["errors_caught"])


def test_run_demo_has_smoke_mode():
    txt = (ROOT / "run_demo.sh").read_text()
    assert "--smoke" in txt


def test_app_host_serves_and_binds_localhost(tmp_path):
    from fastapi.testclient import TestClient
    from twinread import app_host
    assert app_host.HOST == "127.0.0.1" and app_host.PORT == 1777
    client = TestClient(app_host.make_app(ROOT / "results", data_dir=tmp_path))  # no tile data: viewer degrades
    r = client.get("/")
    assert r.status_code == 200
    low = r.text.lower()
    assert "research prototype" in low and "not a diagnostic device" in low and "1778/rig" in r.text
    assert client.get("/api/headline").json()["device"]["uno_q_latency"].startswith("pending")
    t = client.get("/api/tiles?dataset=crc&n=4").json()
    assert "available" in t
    assert client.get("/tile/crc/0.png").status_code in (200, 404)


# ---- Task 18 fix round 1 (review): regression guards ------------------------------------------------------------
OVERCLAIMS = ("recall trade", "runs offline on an arduino", "runs offline on a uno q", "runs on an arduino uno q",
              "runs on the uno q", "sends 10% of tiles", "make the pipeline cheap", "decoded 8,000",
              "which reader to distrust", "phone-grade", "equally strong")


def _deliverable_texts():
    from fastapi.testclient import TestClient
    from twinread import app_host
    page = TestClient(app_host.make_app(ROOT / "results")).get("/").text
    return {"README.md": (ROOT / "README.md").read_text(), "project.json": (ROOT / "project.json").read_text(),
            "deck_outline.md": (ROOT / "docs" / "deck_outline.md").read_text(), "dashboard": page,
            "figures.py": (ROOT / "src" / "twinread" / "figures.py").read_text()}


def test_no_overclaims_in_deliverables():
    bad = [(name, phrase) for name, txt in _deliverable_texts().items() for phrase in OVERCLAIMS
           if phrase in txt.lower()]
    assert not bad, bad


def test_abstract_carries_required_caveats():
    txt = (ROOT / "README.md").read_text()
    abstract = re.search(r"^## Abstract\s*\n(.*?)(?=^!\[|^## )", txt, re.S | re.M).group(1)
    for must in ("On CRC", "Camelyon17", "does not beat CONCH alone", "25.1%", "below every teacher", "x86 replay",
                 "pending", "×1.11", "tumour recall"):
        assert must in abstract, must


def test_crc_gain_decomposition_in_headline():
    hl = json.loads((ROOT / "results" / "headline.json").read_text())
    c = hl["crc_fusion_fragility"]["per_class_contribution_to_bal_acc_delta_pts"]
    assert sum(c.values()) == pytest.approx(hl["fusion"]["CM_crc"]["delta_vs_best_single_pts"], abs=1e-9)
    assert abs(hl["crc_fusion_fragility"]["mus_plus_str_contribution_pts"]) < 0.05  # the MUS/STR swings cancel
    rest = {k: v for k, v in c.items() if k not in ("MUS", "STR")}
    assert c["TUM"] == max(rest.values()) and c["TUM"] > 0.5 * sum(rest.values())  # net gain mostly tumour recall


def test_fig2_keeps_committed_figure_without_gallery(monkeypatch, tmp_path):
    from twinread import figures
    monkeypatch.setattr(figures, "_gallery_items", lambda n=8: None)
    target = ROOT / "figures" / "fig2_review_flag.png"
    before = target.read_bytes()
    monkeypatch.setattr(figures, "FIG", figures.FIG_DEFAULT)
    assert figures.fig2_review_flag() == target and target.read_bytes() == before
    monkeypatch.setattr(figures, "FIG", tmp_path)  # elsewhere it draws the gallery-less variant, labelled
    out = figures.fig2_review_flag()
    assert out.parent == tmp_path and out.exists()
