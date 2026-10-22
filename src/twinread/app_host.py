"""TwinRead host dashboard (FastAPI, 127.0.0.1:1777): headline results and a tile viewer. Read-only, local only.

    PYTHONPATH=src .venv/bin/python -m twinread.app_host [--port 1777]

- `/`                 headline tables from results/headline.json + tile viewer + figures + a link to the rig page
- `/api/headline`     results/headline.json (rebuilt from the source JSONs if missing)
- `/api/tiles`        per-tile test predictions (both readers' calls, fused call, pre-registered flag) from
                      results/cm_local/CM/<dataset>/tile_preds.npz (scripts/export_tile_preds.py)
- `/tile/<ds>/<i>.png` the test tile image from data/ (git-ignored; 404 when absent)
- `/figures/<name>`   the slide figures
The screen-capture rig page is a separate process on 127.0.0.1:1778 (rig/show_tiles.py). No outbound network calls.
Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response

ROOT = Path(__file__).resolve().parents[2]
HOST, PORT, RIG_URL = "127.0.0.1", 1777, "http://127.0.0.1:1778/rig"
CAM_TEST_HOSPITAL = 2  # same as twinread.data.CAM_TEST_HOSPITAL (not imported: keeps the dashboard light)
FILTERS = ("all", "flagged", "errors", "disagree", "shared", "missed")


class TileStore:
    """Lazy per-dataset access to tile_preds.npz and the test tile images; every miss is reported, not raised."""

    def __init__(self, results: Path, data: Path):
        self.results, self.data, self._c = Path(results), Path(data), {}

    def get(self, ds: str) -> dict:
        if ds in self._c:
            return self._c[ds]
        out = {"available": False, "reason": ""}
        base = self.results / "cm_local/CM" / ds
        tp = base / "tile_preds.npz"
        if ds not in ("crc", "cam"):
            out["reason"] = f"unknown dataset {ds!r}"
        elif not tp.exists():
            out["reason"] = f"{tp.relative_to(ROOT) if tp.is_relative_to(ROOT) else tp} missing: run " \
                            "scripts/export_tile_preds.py (needs the cached embeddings in data/)"
        else:
            t = dict(np.load(tp))
            out.update(available=True, t=t, classes=json.loads((base / "preregistration.json").read_text())["classes"],
                       shared=set(json.loads((base / "review.json").read_text())["confident_shared_errors"]["indices"]))
            out["images"] = self._images(ds)
        self._c[ds] = out
        return out

    def _images(self, ds):
        try:
            if ds == "crc":
                return np.load(self.data / "crc_test_x.npy", mmap_mode="r")
            x = np.load(self.data / "cam_x.npy", mmap_mode="r")
            return x[np.flatnonzero(np.load(self.data / "cam_h.npy") == CAM_TEST_HOSPITAL)]
        except (FileNotFoundError, ValueError):
            return None


def _rows(st: dict, filt: str) -> np.ndarray:
    t = st["t"]
    pF, y = t["pF"], t["y"]
    err = pF.argmax(1) != y
    dis = t["pA"].argmax(1) != t["pB"].argmax(1)
    m = {"all": np.ones(len(y), bool), "flagged": t["flag"].astype(bool), "errors": err, "disagree": dis,
         "shared": np.isin(np.arange(len(y)), sorted(st["shared"])), "missed": err & ~t["flag"].astype(bool)}[filt]
    idx = np.flatnonzero(m)
    return idx[np.argsort(-t["s4"][idx], kind="stable")]  # most review-worthy first


def _tile_json(st: dict, i: int) -> dict:
    t, cls = st["t"], st["classes"]
    call = lambda p: {"call": cls[int(p.argmax())], "p": round(float(p.max()), 4)}
    return {"i": int(i), "truth": cls[int(t["y"][i])], "A": call(t["pA"][i]), "B": call(t["pB"][i]),
            "F": call(t["pF"][i]), "flag": bool(t["flag"][i]), "s3": round(float(t["s3_fused_unc"][i]), 5),
            "s4": round(float(t["s4"][i]), 5), "disagree": bool(t["pA"][i].argmax() != t["pB"][i].argmax()),
            "fused_correct": bool(t["pF"][i].argmax() == t["y"][i]), "shared_blind_spot": int(i) in st["shared"],
            "image": st["images"] is not None}


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def _page(hl: dict) -> str:
    f = hl["fusion"]
    nm = {"conch": "CONCH", "musk": "MUSK", "pathgenclip": "PathGen-CLIP", "quiltnet_b16": "QuiltNet-B-16"}
    fus_rows = "".join(
        f"<tr><td>{'CONCH + MUSK' if k.startswith('CM') else 'Open pair'}</td><td>{'CRC' if k.endswith('crc') else 'Camelyon17'}"
        f"</td><td>{nm[d['readers']['A']]} {d['bal_acc']['A']:.4f}</td><td>{nm[d['readers']['B']]} {d['bal_acc']['B']:.4f}</td>"
        f"<td><b>{d['bal_acc']['fused']:.4f}</b> <span class=mut>({d['primary_fusion']})</span></td>"
        f"<td>{d['delta_vs_best_single_pts']:+.2f} pts <span class=mut>[{d['delta_ci_pts'][0]:+.2f}, "
        f"{d['delta_ci_pts'][1]:+.2f}]</span></td><td class=mut>{html.escape(d['ci_mode'])}</td></tr>"
        for k, d in f.items())
    rv_rows = ""
    fr = hl["crc_fusion_fragility"]
    for k in ("CM_crc", "CM_cam"):
        bname = nm[f[k]["readers"][f[k]["best_single_reader"]]]
        r, a, op = f[k]["review_caught_at_10pct"], f[k]["acc_cleared_at_10pct"], f[k]["prereg_rule_test_operating_point"]
        rv_rows += (f"<tr><td>{'CRC' if k.endswith('crc') else 'Camelyon17'}</td><td>{_pct(r['s1_disagree'])}</td>"
                    f"<td>{_pct(r['s2_js'])}</td><td><b>{_pct(r['s3_fused_uncertainty'])}</b></td>"
                    f"<td>{_pct(r['s4_learned'])}</td><td>{_pct(r['A_own_confidence_own_errors'])}</td>"
                    f"<td>{_pct(r['B_own_confidence_own_errors'])}</td><td>{_pct(a['s4'], 2)} vs {bname} alone "
                    f"{_pct(a['best_single_own'], 2)}</td><td>flags {_pct(op['flagged_fraction'])}, catches "
                    f"{_pct(op['errors_caught'])}</td></tr>")
    k, s, dv = hl["kernels"], hl["student"], hl["device"]
    cm = s["CM"]
    t = cm["teachers_test_bal_acc"]
    figs = sorted(p.name for p in (ROOT / "figures").glob("*.png"))
    fig_html = "".join(f'<a href="/figures/{n}" target="_blank"><img src="/figures/{n}" alt="{n}" loading="lazy"></a>'
                       for n in figs)
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>TwinRead dashboard</title>
<style>
:root{{--bg:#f6f5f2;--card:#fff;--fg:#0b0b0b;--mut:#52514e;--line:#e1e0db;--A:#2a78d6;--B:#eb6834;--F:#1baf7a;
--s4:#6250d6;--warn:#fff4d6;--warnline:#eda100;--rev:#b4231a;--ok:#1c7c3f}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#141413;--card:#1d1d1b;--fg:#f0efec;
--mut:#c3c2b7;--line:#34332f;--A:#3987e5;--B:#d95926;--F:#199e70;--s4:#9085e9;--warn:#3a2f10;--warnline:#c98500;
--rev:#ff6b60;--ok:#4cc17a}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}}
.banner{{background:#7a1010;color:#fff;text-align:center;padding:6px 12px;font-weight:700}}
header{{padding:14px 16px 4px}}h1{{margin:0;font-size:22px}}.sub{{color:var(--mut);margin:2px 0 0}}
main{{padding:8px 16px 24px;display:grid;gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;overflow-x:auto}}
h2{{margin:0 0 8px;font-size:15px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--mut);font-weight:600}}
.mut{{color:var(--mut)}}.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}}
.kv{{display:grid;grid-template-columns:auto 1fr;gap:3px 12px}}.kv b{{font-variant-numeric:tabular-nums}}
.pending{{background:var(--warn);border:1px solid var(--warnline);border-radius:8px;padding:8px 10px;margin-top:8px}}
.controls{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}}
select,button{{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}}
.tile{{border:1px solid var(--line);border-radius:8px;padding:6px;font-size:12.5px}}
.tile img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:5px;image-rendering:auto;background:#ddd}}
.tile .noimg{{width:100%;aspect-ratio:1;display:grid;place-items:center;background:var(--line);border-radius:5px;color:var(--mut)}}
.r{{display:flex;justify-content:space-between;gap:6px}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}}
.flag{{font-weight:700;border-radius:5px;padding:1px 6px;color:#fff}}.flag.rev{{background:var(--rev)}}.flag.ok{{background:var(--ok)}}
.figs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}}.figs img{{width:100%;border:1px solid var(--line);border-radius:6px}}
a{{color:var(--A)}}
</style></head><body>
<div class=banner>Research prototype, not a diagnostic device</div>
<header><h1>TwinRead: CONCH and MUSK as two independent readers</h1>
<p class=sub>Fused reader, review flag, shared blind spots, and an int8 student that runs on an offline Arduino UNO Q
(device numbers measured on the board).
LOCAL-SUBSET dev numbers (CRC 20k-tile train subset; full CRC-VAL-HE-7K test). Every number below is read from
<code>results/headline.json</code>. Screen-capture rig: <a href="{RIG_URL}">{RIG_URL}</a> (run <code>rig/show_tiles.py</code>).</p></header>
<main>
<section class=card><h2>Fused vs single readers (test balanced accuracy)</h2>
<table><tr><th>pair</th><th>dataset</th><th>reader A</th><th>reader B</th><th>fused</th><th>fused − best single, 95% CI</th><th>CI</th></tr>
{fus_rows}</table>
<p class=mut>On CRC, large opposing MUS/STR recall swings cancel (net {fr['mus_plus_str_contribution_pts']:+.2f} pts);
the net gain is mostly tumour recall ({fr['per_class_recall_delta_fused_minus_musk_pts']['TUM']:+.1f} pts, worth
{fr['tum_contribution_pts']:+.2f} pts of balanced accuracy). Validation picked the fusion rule by 1-2 tiles. The open pair is a
negative control: fusion adds nothing there.</p></section>
<section class=card><h2>Review flag (CONCH + MUSK): errors caught when 10% of tiles go to review</h2>
<table><tr><th>dataset</th><th>s1 disagree†</th><th>s2 JS</th><th>s3 fused uncertainty</th><th>s4 learned</th>
<th>CONCH alone*</th><th>MUSK alone*</th><th>accuracy of cleared tiles (s4)</th><th>pre-registered threshold on test</th></tr>
{rv_rows}</table>
<p class=mut>* each reader flagged by its own confidence, scored against its own errors. s4 (which adds disagreement)
adds nothing over s3; the value of two readers here is a better-calibrated fused reader. The val-calibrated
threshold over-flags CRC test (val near ceiling; val → test is a distribution shift).
† s1 is binary: it flags the {_pct(f['CM_crc']['s1_disagree_rate'])} of CRC tiles where the readers disagree; the other
{_pct(f['CM_crc']['s1_budget_filled_in_test_order_at_10pct'])} of the budget is filled in shuffled test order
(effectively random).</p></section>
<div class=grid2>
<section class=card><h2>Kernels ({html.escape(k['gpu'])})</h2><div class=kv>
<span>K1 dual preprocess</span><b>{k['k1_cuda_tiles_per_s_224']:,.0f} tiles/s input on GPU ({k['k1_cuda_incl_h2d_tiles_per_s_224']:,.0f} incl. H2D), {'bit-exact to the official transforms' if k['k1_bit_exact_fraction_min_over_sets_impls_readers'] == 1.0 else 'NOT bit-exact'}</b>
<span>K1 preprocessing vs PIL (end-to-end run)</span><b>×{k['k1_preprocess_speedup_vs_pil']:.0f} ({k['k1_e2e_preprocess_tiles_per_s']['k1_gpu_incl_h2d']:,.0f} vs {k['k1_e2e_preprocess_tiles_per_s']['pil_reference_cpu']:,.0f} tiles/s)</b>
<span>K1 end to end (CONCH+MUSK)</span><b>×{k['k1_e2e_speedup_vs_reference']:.2f} (forward is {_pct(k['model_forward_share_of_reference_e2e'], 0)} of the time)</b>
<span>K2 camera aug (SIMULATION)</span><b>{k['k2_cuda_images_per_s_sev1']:,.0f} images/s, ≤ {k['k2_per_op_max_levels']} level per op</b>
<span>K2 vs torch in student training</span><b>×{k['student_train_aug_speedup_k2_vs_torch']['CM']:.1f}</b>
</div><p class=mut>{html.escape(k['k1_context'])}</p></section>
<section class=card><h2>Edge student (CRC-VAL-HE-7K)</h2><div class=kv>
<span>teachers: CONCH / MUSK / fused</span><b>{t['CONCH_probe']:.4f} / {t['MUSK_probe']:.4f} / {t['fused']:.4f}</b>
<span>CM student fp32 clean / camera sim 1.0</span><b>{cm['fp32_test_bal_acc']['clean']:.4f} / {cm['fp32_test_bal_acc']['sev1.0']:.4f}</b>
<span>CM student int8 (deployed)</span><b>{cm['int8_test_bal_acc']:.4f} (below every teacher)</b>
<span>OPEN student int8 (distributable)</span><b>{s['OPEN']['int8_test_bal_acc']:.4f}</b>
<span>size</span><b>{s['params_millions']:.2f} M params ({s['params_pct_of_teacher_vision_towers']:.1f}% of the teachers' vision towers), {s['int8_onnx_mb']:.1f} MB int8</b>
<span>LED sketch</span><b>{'flashed on the board; Bridge RPC verified' if dv.get('mcu_flashed') else ('compile-verified, not flashed' if dv['mcu_sketch_compile_verified'] else 'not verified')}</b>
<span>rig (SIMULATED captures)</span><b>{dv['rig_simulated_captures']:,} trials, {dv['rig_simulated_accepted']:,} accepted, {dv['rig_simulated_wrong_ids']} wrong labels; real captures: {dv['real_camera_captures']}</b>
<span>int8 bundle review flag (CM, test)</span><b>flags {_pct(cm['int8_bundle_flag_rate_test'])} (thresholds recalibrated on int8 val)</b>
<span>int8 flag vs teacher flag (CM)</span><b>kappa {cm['int8_device_flag_test_clean']['kappa']:.2f}, recovers {_pct(cm['int8_device_flag_test_clean']['teacher_recall'])} of teacher-flagged tiles; two heads of one student</b>
</div><div class=pending>Arduino UNO Q, measured on the board: <b>{_board(dv)}</b>.</div>
<p class=mut>The CM student derives from CONCH and MUSK (CC-BY-NC-ND-4.0) and is not distributed; the OPEN student is distributable (not distributed).</p></section>
</div>
<section class=card><h2>Tile viewer: CRC test tiles, both readers' calls, the fused call and the pre-registered flag</h2>
<div class=controls><label>dataset <select id=ds><option value=crc>CRC-VAL-HE-7K</option><option value=cam>Camelyon17 hospital 2</option></select></label>
<label>show <select id=filt><option value=shared>confident shared errors</option><option value=missed>fused errors NOT flagged</option>
<option value=errors>fused errors</option><option value=disagree>readers disagree</option><option value=flagged>flagged</option>
<option value=all>all tiles</option></select></label><button id=more>more</button><span id=info class=mut></span></div>
<div id=tiles class=tiles></div>
<p class=mut>Calls are linear probes on frozen CONCH / MUSK features; fused = the pre-registered fusion rule. Sorted by s4
(most review-worthy first). Post-hoc reproduction of the frozen fits (matches review.json exactly).</p></section>
<section class=card><h2>Figures</h2><div class=figs>{fig_html}</div></section>
</main>
<script>
const $=s=>document.querySelector(s);let off=0;
function card(t,ds){{const c=(r,col,lab)=>`<div class=r><span><span class=dot style="background:var(${{col}})"></span>${{lab}}</span><b>${{r.call}} ${{r.p.toFixed(2)}}</b></div>`;
 const img=t.image?`<img src="/tile/${{ds}}/${{t.i}}.png" alt="tile ${{t.i}}">`:`<div class=noimg>image not cached</div>`;
 return `<div class=tile>${{img}}<div class=r><span>#${{t.i}} truth <b>${{t.truth}}</b></span>${{t.flag?'<span class="flag rev">REVIEW</span>':'<span class="flag ok">clear</span>'}}</div>`+
 c(t.A,'--A','CONCH')+c(t.B,'--B','MUSK')+c(t.F,'--F','fused')+`<div class="r mut"><span>s4 ${{t.s4.toFixed(3)}}</span><span>${{t.fused_correct?'fused right':'fused WRONG'}}</span></div></div>`}}
async function load(reset){{if(reset){{off=0;$('#tiles').innerHTML=''}};const ds=$('#ds').value,f=$('#filt').value;
 const r=await (await fetch(`/api/tiles?dataset=${{ds}}&filter=${{f}}&n=24&offset=${{off}}`)).json();
 if(!r.available){{$('#info').textContent='tile predictions unavailable: '+r.reason;return}}
 $('#tiles').insertAdjacentHTML('beforeend',r.tiles.map(t=>card(t,ds)).join(''));off+=r.tiles.length;
 $('#info').textContent=`${{off}} of ${{r.n_match}} tiles`}}
$('#ds').onchange=()=>load(true);$('#filt').onchange=()=>load(true);$('#more').onclick=()=>load(false);load(true);
</script></body></html>"""


def _board(dv: dict) -> str:
    ob = dv.get("onboard") or {}
    if not isinstance(dv.get("uno_q_latency"), dict) or not ob:
        return html.escape(str(dv.get("uno_q_latency")))
    return (f"int8 student {ob['cm_int8_p50_ms_4_threads']:.1f} ms per tile (4 threads); {ob['cm_int8_bal_acc_board']:.4f} "
            f"balanced accuracy (CM) on all 7,180 CRC-VAL-HE-7K tiles; {ob['cm_full_loop_tiles_per_s']:.1f} tiles/s "
            "for the full app loop")


def make_app(results: Path = ROOT / "results", data_dir: Path = ROOT / "data") -> FastAPI:
    results = Path(results)
    store = TileStore(results, Path(data_dir))
    app = FastAPI(title="TwinRead dashboard", docs_url=None, redoc_url=None)

    def headline_doc() -> dict:
        p = results / "headline.json"
        if p.exists():
            return json.loads(p.read_text())
        from twinread import headline
        return headline.build(results)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _page(headline_doc())

    @app.get("/api/headline")
    def api_headline():
        return headline_doc()

    @app.get("/api/tiles")
    def api_tiles(dataset: str = "crc", filter: str = Query("shared"), n: int = Query(24, ge=1, le=200),
                  offset: int = Query(0, ge=0)):
        st = store.get(dataset)
        if not st["available"]:
            return {"available": False, "reason": st["reason"], "tiles": []}
        if filter not in FILTERS:
            raise HTTPException(400, f"filter must be one of {FILTERS}")
        idx = _rows(st, filter)
        return {"available": True, "dataset": dataset, "filter": filter, "n_match": int(len(idx)),
                "tiles": [_tile_json(st, i) for i in idx[offset:offset + n]]}

    @app.get("/tile/{dataset}/{i}.png")
    def tile_png(dataset: str, i: int):
        st = store.get(dataset)
        imgs = st.get("images")
        if not st["available"] or imgs is None or not 0 <= i < len(imgs):
            raise HTTPException(404, "tile image not available")
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.asarray(imgs[i], np.uint8)).save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.get("/figures/{name}")
    def figure(name: str):
        p = (ROOT / "figures" / name).resolve()
        if p.parent != (ROOT / "figures").resolve() or p.suffix != ".png" or not p.exists():
            raise HTTPException(404, "no such figure")
        return FileResponse(p)

    return app


def main(argv=None):
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args(argv)
    print(f"TwinRead dashboard on http://{HOST}:{a.port}/ (research prototype, not a diagnostic device)", flush=True)
    uvicorn.run(make_app(), host=HOST, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
