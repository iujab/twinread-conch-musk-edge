// Builds docs/twinread_deck.pptx: editable text, shapes, tables and native charts, with speaker notes.
// Every TwinRead number is read from results/**/*.json at build time. The only numbers typed by hand are external facts:
// the two literature figures on the second-opinion slide (cited in its notes and on References) and the disaster, Mars and
// price figures on the why-shrink slide (cited on it). A rebuild overwrites edits made directly in the .pptx, so port them
// here; docs/deck/manual_edits/ keeps the team's hand-edited versions.
//
//   cd docs/deck && npm install && node build_deck.js
//
// docs/deck/assets/ comes from make_assets.py and make_kernel_figures.py and is committed, so this script does not need data/.
// PDF copy: export from PowerPoint, or  soffice --headless --convert-to pdf --outdir docs docs/twinread_deck.pptx
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");

const ROOT = path.resolve(__dirname, "../..");
const OUT = path.join(ROOT, "docs/twinread_deck.pptx");
const A = (f) => path.join(__dirname, "assets", f);
const J = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, f), "utf8"));

const H = J("results/headline.json");
const OV = J("results/cm_local/CM/crc/reader_error_overlap.json");
const TC = J("results/cm_local/CM/crc/triage_curves.json");
const KC = J("results/uno_q/onboard/int8_kernel_check.json");
const SM = J("results/student/cm_crc/metrics.json");
const BS = JSON.parse(fs.readFileSync(A("blindspots.json"), "utf8"));
const KF = JSON.parse(fs.readFileSync(A("kernel_figures.json"), "utf8"));
const MS = J("results/student/model_sizes.json");
const X86 = J("results/uno_q/backbone_latency_x86_sim.json");
const RD = J("results/cm_local/CM/crc/readers.json");
const FJ = J("results/cm_local/CM/crc/fusion.json");
const PRE = J("results/cm_local/CM/crc/preregistration.json");
const DRIFT = J("results/student/cm_crc/int8_drift.json");
const PF = JSON.parse(fs.readFileSync(A("probe_fusion.json"), "utf8"));

const CC = H.fusion.CM_crc, CA = H.fusion.CM_cam, OC = H.fusion.OPEN_crc, OA = H.fusion.OPEN_cam;
const K = H.kernels, S = H.student, D = H.device.onboard, FR = H.crc_fusion_fragility;

const MINUS = "−";
const pct2 = (v) => (100 * v).toFixed(2) + "%";
const pct1 = (v) => (100 * v).toFixed(1) + "%";
const int = (n) => Math.round(n).toLocaleString("en-US");
const x1 = (v) => "×" + v.toFixed(1);
const x2 = (v) => "×" + v.toFixed(2);
const x0 = (v) => "×" + Math.round(v);
const pts = (v) => (Math.abs(v) < 0.005 ? "0.00" : (v > 0 ? "+" : MINUS) + Math.abs(v).toFixed(2));
const ciTxt = (ci) => `[${pts(ci[0])}, ${pts(ci[1])}]`;

// ---- theme: hematoxylin blue and eosin pink for the two readers, their mix for the fused reader ----------------------
const THEME = {
  name: "TwinRead H and E",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1F1A2E", lt1: "FFFFFF", dk2: "241B3D", lt2: "F3F1F7",
    accent1: "2573BE", accent2: "DB4B86", accent3: "5C2FB0", accent4: "A87400", accent5: "625E70", accent6: "C9C6D2",
    hlink: "2573BE", folHlink: "5C2FB0",
  },
};
const HEX = THEME.colors;
const BASE = "8C8898"; // neutral comparison bars

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "TwinRead";
pres.author = "Yubo Wu, Noah Zhang, William Liu, Franklin Liu, Andy Chen";
pres.subject = "2026 Ashby Prize in Computational Science Hackathon";
const C = pres.SchemeColor;
const CONCH = C.accent1, MUSK = C.accent2, FUSED = C.accent3, DISAGREE = C.accent4, MUTED = C.accent5, RULE = C.accent6;
const INK = C.text1, WHITE = C.background1, TINT = C.background2, NIGHT = C.text2;

const LM = 0.6, CW = 12.13; // left margin and content width
const FOOT = "TwinRead";

const footerObjects = () => [
  { image: { path: A("motif.png"), x: LM, y: 7.03, w: 0.3, h: 0.2 } },
  { text: { text: FOOT, options: { x: LM + 0.38, y: 6.99, w: 8, h: 0.28, fontSize: 10, color: MUTED, margin: 0, valign: "middle" } } },
];
const titlePh = (o = {}) => ({
  placeholder: {
    options: { name: "title", type: "title", x: LM, y: 0.42, w: CW, h: 1.05, fontSize: 30, bold: true, color: INK, align: "left", valign: "top", margin: 0, ...o },
    text: "Slide title",
  },
});
const slideNo = { x: 12.13, y: 6.99, w: 0.6, h: 0.28, fontSize: 10, color: MUTED, align: "right", margin: 0 };

pres.defineSlideMaster({
  title: "CONTENT", background: { color: WHITE }, slideNumber: slideNo,
  objects: [titlePh(), {
    placeholder: {
      options: { name: "source", type: "body", x: LM, y: 6.38, w: CW, h: 0.52, fontSize: 12, color: MUTED, align: "left", valign: "bottom", margin: 0 },
      text: "Source",
    },
  }, ...footerObjects()],
});
pres.defineSlideMaster({
  title: "CONTENT_PLAIN", background: { color: WHITE }, slideNumber: slideNo,
  objects: [titlePh(), ...footerObjects()],
});
pres.defineSlideMaster({
  title: "COVER", background: { color: NIGHT },
  objects: [titlePh({ x: 0.68, y: 3.9, w: 6.6, h: 1.1, fontSize: 60, color: WHITE, valign: "bottom" })],
});
pres.defineSlideMaster({
  title: "CLOSE", background: { color: NIGHT }, slideNumber: { ...slideNo, color: RULE },
  objects: [titlePh({ color: WHITE, fontSize: 32 })],
});

// ---- helpers --------------------------------------------------------------------------------------------------------
const T = (s, text, x, y, w, h, o = {}) => s.addText(text, { x, y, w, h, margin: 0, fontSize: 18, color: INK, align: "left", valign: "top", isTextBox: true, ...o });
const dot = (s, x, y, d, color, name) => s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { type: "none" }, objectName: name || "marker" });
const card = (s, x, y, w, h, o = {}) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: TINT }, line: { type: "none" }, objectName: "card", ...o });
const arrow = (s, x, y, w, h, o = {}) => s.addShape(pres.shapes.LINE, { x, y, w, h, line: { color: MUTED, width: 1.75, endArrowType: "triangle" }, objectName: "arrow", ...o });
const box = (s, text, x, y, w, h, fill, color, o = {}) => s.addText(text, {
  shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x, y, w, h, fill: { color: fill }, line: { type: "none" }, color, fontSize: 18,
  align: "center", valign: "middle", margin: 4, objectName: "step", ...o,
});
const title = (s, text) => s.addText(text, { placeholder: "title" });
const source = (s, text) => s.addText(text, { placeholder: "source" });
const motif = (s, x, y, d) => { // the two readers, overlapping
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: CONCH, transparency: 12 }, line: { type: "none" }, objectName: "motif CONCH" });
  s.addShape(pres.shapes.OVAL, { x: x + d * 0.62, y, w: d, h: d, fill: { color: MUSK, transparency: 18 }, line: { type: "none" }, objectName: "motif MUSK" });
};
const CHART_TEXT = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5 };
const noGrid = { catGridLine: { style: "none" }, valGridLine: { style: "none" } };

function hBars(s, labels, values, colors, o) {
  s.addChart(pres.charts.BAR, [{ name: o.series, labels, values }], {
    x: o.x, y: o.y, w: o.w, h: o.h, barDir: "bar", chartColors: colors, invertedColors: colors, barGapWidthPct: o.gap ?? 45,
    catAxisOrientation: "maxMin", valAxisHidden: true, valAxisMinVal: o.min ?? 0, valAxisMaxVal: o.max, ...noGrid, ...CHART_TEXT,
    catAxisLabelFontSize: o.font ?? 14, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: o.font ?? 14, dataLabelFontBold: true, dataLabelColor: HEX.dk1,
    dataLabelFormatCode: o.fmt ?? "#,##0", catAxisLineShow: false, valAxisLineShow: false, showLegend: false, showTitle: false, altText: o.alt,
    layout: o.layout,
  });
}

const tableOpts = (o) => ({ fontFace: "Arial", fontSize: 15, color: HEX.dk1, border: { type: "solid", pt: 0.75, color: "E8E6EE" }, valign: "middle", margin: [0.04, 0.1, 0.04, 0.1], ...o });
const th = (text, o = {}) => ({ text, options: { bold: true, color: HEX.accent5, fill: { color: HEX.lt2 }, ...o } });

// Speaker notes as a script: bullets to read aloud, then answers for questions. The time tag comes from the word count, so
// tags and script cannot drift apart. formatNotes() turns the lines into real paragraphs and bullets after the build.
const WPM = 140;
let talkSeconds = 0;
const mmss = (sec) => `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
function script(s, { cue, say, built = [], asked = [], sources, after }) {
  const words = [...say, ...built].join(" ").split(/\s+/).length;
  const sec = Math.ceil((words / WPM) * 12) * 5; // round up to 5 s
  talkSeconds += sec;
  const dot = (x) => `\u2022 ${x}`;
  const lines = [`[${mmss(sec)}]`];
  if (cue) lines.push(cue);
  lines.push(...say.map(dot));
  if (built.length) lines.push("How we built it:", ...built.map(dot));
  if (asked.length) lines.push("If asked:", ...asked.map(dot));
  if (sources) lines.push(`Sources: ${sources}`);
  if (after) lines.push(after(talkSeconds));
  s.addNotes(lines.join("\n"));
}

// ---- 1 cover --------------------------------------------------------------------------------------------------------
pres.addSection({ title: "Talk" });
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Talk" });
  title(s, "TwinRead");
  motif(s, 0.73, 1.82, 1.92);
  T(s, "Yubo Wu, Noah Zhang, William Liu, Franklin Liu, Andy Chen", 0.73, 5.25, 6.5, 0.35, { fontSize: 16, color: WHITE, bold: true });
  T(s, "University of Illinois Urbana-Champaign", 0.73, 5.63, 6.5, 0.3, { fontSize: 14, color: RULE });
  T(s, "2026 Ashby Prize in Computational Science Hackathon", 0.73, 5.93, 6.5, 0.3, { fontSize: 14, color: RULE });
  const order = ["TUM", "STR", "LYM", "MUS", "NORM", "DEB", "MUC", "ADI", "BACK"];
  const names = { ADI: "adipose", BACK: "background", DEB: "debris", LYM: "lymphocytes", MUC: "mucus", MUS: "smooth muscle", NORM: "normal mucosa", STR: "stroma", TUM: "tumour epithelium" };
  const d = 1.62, g = 0.06, x0_ = 7.85, y0_ = 1.26;
  order.forEach((c, i) => s.addImage({
    path: A(`class_${c}.jpg`), x: x0_ + (i % 3) * (d + g), y: y0_ + Math.floor(i / 3) * (d + g), w: d, h: d,
    altText: `H&E tile from the CRC-VAL-HE-7K test set: ${names[c]}`,
  }));
  script(s, {
    say: [
      "Hello, we are team TwinRead, from the University of Illinois.",
      "Our question comes from the challenge: can foundation models make pathology more precise, more reproducible and more interpretable?",
      "These are the nine tissue types in our test set. We call each small square image a tile.",
    ],
  });
}

// ---- 2 problem ------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT_PLAIN", sectionTitle: "Talk" });
  title(s, "Why a second opinion matters");
  T(s, "75.3%", LM, 1.45, 5.6, 1.05, { fontSize: 66, bold: true });
  T(s, "of individual breast-biopsy reads matched the expert panel, and only 48% for atypia", LM, 2.55, 5.5, 1.1, { fontSize: 20 });
  T(s, "47%", 7.0, 1.45, 5.6, 1.05, { fontSize: 66, bold: true });
  T(s, "of the world’s population has little or no access to diagnostics", 7.0, 2.55, 5.5, 1.1, { fontSize: 20 });
  card(s, LM, 4.1, CW, 1.95);
  motif(s, LM + 0.4, 4.6, 0.92);
  T(s, "Can two AI models act as independent readers?", 2.75, 4.5, 9.6, 0.6, { fontSize: 26, bold: true, valign: "middle" });
  T(s, "And can we make it run offline on a small device?", 2.75, 5.1, 9.6, 0.5, { fontSize: 20, valign: "middle" });
  script(s, {
    say: [
      "Hard cases in pathology often get a second, independent read, because reads vary.",
      "In Elmore's JAMA study, individual breast biopsy reads matched the expert panel 75 percent of the time, and only 48 percent for atypia.",
      "But a second read needs a second expert, and the Lancet Commission estimates almost half the world has little or no access to diagnostics.",
      "So: can CONCH and MUSK act like two independent readers? And can we make that run offline on a small device?",
    ],
    sources: "Elmore et al., JAMA 2015, 115 pathologists, 6,900 interpretations, doi:10.1001/jama.2015.1405. Fleming et al., Lancet 2021, doi:10.1016/S0140-6736(21)00673-5.",
  });
}

// ---- 3 different errors ---------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT_PLAIN", sectionTitle: "Talk" });
  title(s, "CONCH and MUSK make different mistakes");
  dot(s, LM, 1.43, 0.24, CONCH, "CONCH marker");
  T(s, "CONCH", LM + 0.36, 1.35, 5, 0.4, { fontSize: 22, bold: true, valign: "middle" });
  T(s, "Contrastive: 1.17 M image–caption pairs", LM + 0.36, 1.77, 5.5, 0.35, { fontSize: 17, color: MUTED });
  dot(s, 7.0, 1.43, 0.24, MUSK, "MUSK marker");
  T(s, "MUSK", 7.36, 1.35, 5, 0.4, { fontSize: 22, bold: true, valign: "middle" });
  T(s, "Masked: 50 M images and 1 B text tokens", 7.36, 1.77, 5.37, 0.35, { fontSize: 17, color: MUTED });
  T(s, `At least one model was wrong on ${OV.either_wrong} of ${int(OV.n)} test images`, LM, 2.62, CW, 0.38, { fontSize: 18, bold: true });
  const barW = CW, seg = [OV.conch_only_errors, OV.both_wrong, OV.musk_only_errors], tot = OV.either_wrong;
  s.addChart(pres.charts.BAR, [
    { name: "Only CONCH wrong", labels: ["Tiles"], values: [seg[0]] },
    { name: "Both wrong", labels: ["Tiles"], values: [seg[1]] },
    { name: "Only MUSK wrong", labels: ["Tiles"], values: [seg[2]] },
  ], {
    x: LM, y: 3.05, w: barW, h: 1.4, barDir: "bar", barGrouping: "stacked", chartColors: [HEX.accent1, HEX.dk1, HEX.accent2], barGapWidthPct: 0,
    catAxisHidden: true, valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: tot, ...noGrid, ...CHART_TEXT, showValue: true, dataLabelPosition: "ctr",
    dataLabelColor: "FFFFFF", dataLabelFontSize: 26, dataLabelFontBold: true, dataLabelFormatCode: "0", showLegend: false, showTitle: false,
    dataBorder: { pt: 2, color: "FFFFFF" }, layout: { x: 0, y: 0, w: 1, h: 1 },
    altText: `Stacked bar of the ${tot} tiles at least one reader gets wrong: ${seg[0]} only CONCH, ${seg[1]} both, ${seg[2]} only MUSK.`,
  });
  let cx = LM;
  ["Only CONCH wrong", "Both wrong", "Only MUSK wrong"].forEach((lab, i) => {
    const w = (seg[i] / tot) * barW;
    T(s, lab, cx, 4.52, w, 0.35, { fontSize: 16, align: "center" });
    cx += w;
  });
  T(s, `Only ${OV.both_wrong} out of the ${OV.either_wrong} were misclassified by both models.`, LM, 5.3, CW, 0.6, { fontSize: 24, bold: true, valign: "middle" });
  script(s, {
    say: [
      "Why would two models act like two readers? They learned differently.",
      "CONCH learned by matching pathology images to captions. MUSK first learned by filling in hidden pieces of images and text.",
      `On ${int(OV.n)} colorectal test images, CONCH gets ${OV.conch_errors} wrong and MUSK ${OV.musk_errors}, but only ${OV.both_wrong} are wrong for both.`,
      "So on most of its mistakes, the other model is right. That is the gap a second reader covers.",
    ],
    asked: [
      `Each model here means a linear probe on its frozen features, explained in two slides. ${OV.both_wrong_same_call} of the ${OV.both_wrong} shared errors make the same wrong call.`,
      "This overlap was measured after the test run, from saved predictions. It describes the result and chose nothing.",
    ],
    sources: "results/cm_local/CM/crc/reader_error_overlap.json. Lu et al., Nat Med 2024. Xiang et al., Nature 2025.",
  });
}

// ---- 4 system -------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  title(s, "System Architecture");
  const head = (txt, x, y, w) => T(s, txt, x, y, w, 0.35, { fontSize: 16, bold: true, color: MUTED, valign: "middle" });
  head("1  Read twice", 2.75, 1.3, 2.8);
  head("2  Combine Predictions", 6.0, 1.65, 3);
  // row 1: centre line y = 2.75
  s.addImage({ path: A("class_TUM.jpg"), x: LM, y: 1.85, w: 1.8, h: 1.8, altText: "An example H&E tile" });
  T(s, "H&E tile", LM, 3.72, 1.8, 0.3, { fontSize: 14, color: MUTED, align: "center" });
  arrow(s, 2.43, 2.26, 0.3, 0.44, { flipV: true });
  arrow(s, 2.43, 2.8, 0.3, 0.44);
  box(s, "CONCH", 2.75, 1.75, 2.55, 0.92, CONCH, WHITE, { bold: true });
  box(s, "MUSK", 2.75, 2.83, 2.55, 0.92, MUSK, WHITE, { bold: true });
  T(s, "Linear probes on frozen features", 2.75, 3.82, 2.55, 0.5, { fontSize: 14, color: MUTED, align: "center" });
  arrow(s, 5.34, 2.26, 0.62, 0.44);
  arrow(s, 5.34, 2.8, 0.62, 0.44, { flipV: true });
  box(s, "Fused call", 6.0, 2.13, 2.55, 1.24, FUSED, WHITE, { bold: true });
  arrow(s, 8.6, 2.75, 0.6, 0);
  box(s, [
    { text: "Review flag", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "Least certain tiles go to a pathologist", options: { fontSize: 14, color: MUTED } },
  ], 9.25, 1.95, 3.48, 1.6, TINT, INK);
  // row 2
  head("3  Distill", 7.4, 3.72, 2);
  arrow(s, 7.27, 3.41, 0, 1.0);
  box(s, [
    { text: "Student model", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: `${S.params_millions.toFixed(2)} M parameters, ${S.int8_onnx_mb.toFixed(1)} MB int8`, options: { fontSize: 14, color: MUTED } },
  ], 5.75, 4.46, 3.05, 1.35, TINT, INK);
  arrow(s, 8.85, 5.13, 0.35, 0);
  box(s, [
    { text: "Arduino UNO Q", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "Offline, no GPU, USB-powered", options: { fontSize: 14, color: MUTED } },
  ], 9.25, 4.46, 3.48, 1.35, TINT, INK);
  source(s, `Train: ${int(SM.n.train)} NCT-CRC-HE-100K tiles. Test: ${int(TC.n_test)} CRC-VAL-HE-7K tiles (9 classes) and 1,500 Camelyon17 tiles from an unseen hospital.`);
  script(s, {
    say: [
      "Three steps. One: every tile is read twice, through CONCH and through MUSK.",
      "Two: we combine the predictions into one call. If that call is unsure, a review flag sends the tile to a pathologist.",
      "Three: we train our own small student model to copy both, and run it on an Arduino UNO Q.",
    ],
    built: [
      "Before reading any test label, we fixed the fusion rule and thresholds on validation data in a pre-registration file. Every test number was computed once, after that.",
    ],
    asked: [
      "PathGen-CLIP and QuiltNet, an open model pair, went through the same protocol as a negative control.",
      `The pre-registration file is dated ${PRE.created_utc.slice(0, 10)} and records the candidates, the selection rule and the prompt hashes.`,
    ],
    sources: "results/cm_local/CM/crc/preregistration.json, docs/design.md.",
  });
}

// ---- 5 linear probes: what each "read" is -----------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  const zs = { A: 100 * RD.A.zeroshot.bal_acc, B: 100 * RD.B.zeroshot.bal_acc }, pr = { A: 100 * RD.A.probe.bal_acc, B: 100 * RD.B.probe.bal_acc };
  const T_ = MS.teachers, dims = PF.probe_feature_dims, wts = PF.probe_weights;
  const mp = (v) => `${(v / 1e6).toFixed(1)} M parameters`;
  title(s, "Linear probes: one simple classifier on each model");
  const head = (txt, x, w) => T(s, txt, x, 1.2, w, 0.32, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  head("Frozen encoder", 2.65, 2.75); head("Features", 5.75, 1.75); head("Linear probe, trained", 7.85, 2.4); head("Output", 10.6, 2.13);
  s.addImage({ path: A("class_TUM.jpg"), x: LM, y: 1.55, w: 1.65, h: 1.65, altText: "An example H&E tile" });
  const rows = [[CONCH, "CONCH", T_.conch.params_vision_tower, dims.conch, wts.conch, 1.55], [MUSK, "MUSK", T_.musk.params_vision_tower, dims.musk, wts.musk, 2.44]];
  rows.forEach(([col, name, params, d, w, y], i) => {
    const h = 0.76, c = y + h / 2;
    i === 0 ? arrow(s, 2.29, c, 0.32, 2.375 - c, { flipV: true }) : arrow(s, 2.29, 2.375, 0.32, c - 2.375);
    box(s, [{ text: name, options: { bold: true, fontSize: 18, breakLine: true } }, { text: mp(params), options: { fontSize: 14 } }], 2.65, y, 2.75, h, col, WHITE);
    arrow(s, 5.43, c, 0.29, 0);
    box(s, `${int(d)} numbers`, 5.75, y, 1.75, h, TINT, INK, { fontSize: 16 });
    arrow(s, 7.53, c, 0.29, 0);
    box(s, [{ text: "One layer", options: { bold: true, fontSize: 16, breakLine: true } }, { text: `${int(w)} weights`, options: { fontSize: 14 } }], 7.85, y, 2.4, h, WHITE, INK, { line: { color: col, width: 2.25 } });
    arrow(s, 10.28, c, 0.29, 0);
    box(s, "9 tissue probabilities", 10.6, y, 2.13, h, TINT, INK, { fontSize: 15 });
  });
  T(s, "Why a linear probe", LM, 3.5, 5.8, 0.42, { fontSize: 20, bold: true, valign: "middle" });
  T(s, [
    ["Honest:", "simple enough that it can only use what the features already know"],
    ["Independent:", "each probe sees one model, so their errors stay different"],
    ["Cheap:", `${int(SM.n.train)} labelled tiles, fitted on CPU; the big models never change`],
  ].flatMap(([lead, rest], i, all) => [
    { text: lead + " ", options: { bold: true, bullet: { indent: 16 } } },
    { text: rest, options: { breakLine: i < all.length - 1 } },
  ]), LM, 3.97, 5.8, 2.3, { fontSize: 17, paraSpaceAfter: 9 });
  T(s, `Probes add ${Math.round(pr.B - zs.B)}–${Math.round(pr.A - zs.A)} points over text prompts`, 6.93, 3.5, 5.8, 0.42, { fontSize: 20, bold: true, valign: "middle" });
  T(s, "Balanced accuracy on the colorectal test set (%)", 6.93, 3.93, 5.8, 0.3, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  hBars(s, ["CONCH, text prompts", "CONCH, linear probe", "MUSK, text prompts", "MUSK, linear probe"], [zs.A, pr.A, zs.B, pr.B].map((v) => +v.toFixed(1)), [BASE, HEX.accent1, BASE, HEX.accent2], {
    x: 6.93, y: 4.22, w: 5.8, h: 2.1, max: 118, series: "Balanced accuracy (%)", fmt: "0.0", font: 15, gap: 30, layout: { x: 0.37, y: 0.02, w: 0.61, h: 0.96 },
    alt: `Balanced accuracy: CONCH ${zs.A.toFixed(1)} with text prompts and ${pr.A.toFixed(1)} with a probe; MUSK ${zs.B.toFixed(1)} with text prompts and ${pr.B.toFixed(1)} with a probe.`,
  });
  source(s, `Probe: standardised logistic regression, strength chosen on ${int(SM.n.val)} validation tiles (CONCH C = ${PRE.probe_C.A}, MUSK C = ${PRE.probe_C.B}). Text prompts: zero-shot, no training. Parameters count the vision towers.`);
  script(s, {
    say: [
      "First, what is a read? CONCH and MUSK are huge trained networks, and we never change them. We call that frozen.",
      `Each turns a tile into a list of numbers, ${int(dims.conch)} for CONCH and ${int(dims.musk)} for MUSK. Those are its features.`,
      "A linear probe is the simplest classifier on top: it weights those numbers and gives a probability for each of nine tissue types. CONCH has 90 million parameters. Its probe has under five thousand.",
      "It can only use what the features already know, and each probe sees one model, so their errors stay different.",
      `With text prompts alone, no training, the models reach ${zs.A.toFixed(1)} and ${zs.B.toFixed(1)} percent balanced accuracy, where rare tissues count as much as common ones. With our probes, ${pr.A.toFixed(1)} and ${pr.B.toFixed(1)}.`,
    ],
    built: [
      "MUSK was the slow step, about 8 tiles a second, so we saved features once and fitted probes on CPU.",
    ],
    asked: [
      `A probe is multinomial logistic regression on standardised features: 9 classes times (features + 1) weights. Strength C was chosen from 0.01, 0.1, 1 and 10 on validation balanced accuracy.`,
      `Probe features are pre-projection: CONCH ${int(dims.conch)}-d, MUSK ${int(dims.musk)}-d multi-scale. Probes on the text-aligned embeddings score ${pct1(RD.A.aligned_probe.bal_acc)} and ${pct1(RD.B.aligned_probe.bal_acc)}, so we use those as a sensitivity check and as the student's target space.`,
      "Where is the text? Zero-shot compares each tile with prompt ensembles in the shared image-text space. The prompts are frozen in prompts/ and hashed in the pre-registration. The student learns those text-aligned embeddings.",
    ],
    sources: "results/cm_local/CM/crc/readers.json, preregistration.json, src/twinread/probes.py, results/student/model_sizes.json, docs/progress.md.",
  });
}

// ---- 6 fusion: how two reads become one, with a real tile --------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  const ex = PF.example, CN = { ADI: "Adipose", BACK: "Background", DEB: "Debris", LYM: "Lymphocytes", MUC: "Mucus", MUS: "Muscle", NORM: "Normal", STR: "Stroma" };
  const other = CN[ex.other], p = (v) => Math.round(100 * v);
  title(s, "Fusion: average the two models’ probabilities");
  s.addImage({ path: A("fusion_example.jpg"), x: LM, y: 1.3, w: 1.75, h: 1.75, altText: `Colorectal test tile ${ex.tile}, labelled tumour` });
  T(s, "Real test tile. Truth: tumour.", LM, 3.1, 1.75, 0.55, { fontSize: 14, color: MUTED });
  const panel = (x, w, col, name, d, note, bold) => {
    dot(s, x, 1.34, 0.22, col, name + " marker");
    T(s, name, x + 0.32, 1.27, w - 0.32, 0.36, { fontSize: 18, bold: true, valign: "middle" });
    hBars(s, ["Tumour", other], [p(d.TUM), p(d.other)], [col === CONCH ? HEX.accent1 : col === MUSK ? HEX.accent2 : HEX.accent3, col === CONCH ? HEX.accent1 : col === MUSK ? HEX.accent2 : HEX.accent3], {
      x, y: 1.65, w, h: 1.25, max: 165, series: `${name} probability (%)`, fmt: '0"%"', font: 15, gap: 35, layout: { x: 0.32, y: 0.03, w: 0.66, h: 0.94 },
      alt: `${name}: tumour ${p(d.TUM)} percent, ${other.toLowerCase()} ${p(d.other)} percent.`,
    });
    T(s, note, x, 2.95, w, 0.7, { fontSize: 15, bold: !!bold });
  };
  panel(2.65, 2.8, CONCH, "CONCH", ex.conch, "Calls tumour");
  T(s, "+", 5.47, 1.85, 0.36, 0.6, { fontSize: 32, bold: true, align: "center", valign: "middle" });
  panel(5.85, 2.8, MUSK, "MUSK", ex.musk, `Calls ${other.toLowerCase()}`);
  T(s, "=", 8.67, 1.85, 0.36, 0.6, { fontSize: 32, bold: true, align: "center", valign: "middle" });
  panel(9.05, 3.68, FUSED, "Average", ex.fused, `Calls tumour, correctly, but only ${p(ex.fused.TUM)}% sure, so the review flag fires`, true);
  const v = FR.val_correct_tiles, d = FR.test_delta_vs_musk_by_candidate;
  T(s, "Three rules, chosen on validation only", LM, 3.82, 6.9, 0.38, { fontSize: 18, bold: true, valign: "middle" });
  const rt = (txt) => ({ text: txt, options: { align: "right" } });
  s.addTable([
    [th("Rule"), th(`Validation correct, of ${int(FR.n_val)}`, { align: "right" }), th("Test points vs MUSK", { align: "right" })],
    [{ text: "Average (chosen)", options: { bold: true } }, { text: int(v.average), options: { bold: true, align: "right" } }, { text: pts(100 * d.average), options: { bold: true, align: "right" } }],
    ["Gated neural network", rt(int(v.gated_mlp)), rt(pts(100 * d.gated_mlp))],
    ["Logistic regression on joined features", rt(int(v.concat_lr)), rt(pts(100 * d.concat_lr))],
  ], tableOpts({ x: LM, y: 4.24, w: 6.9, colW: [3.9, 1.55, 1.45], rowH: [0.62, 0.44, 0.44, 0.44] }));
  card(s, 7.8, 3.9, 4.93, 2.3);
  T(s, `${OV.musk_errors} → ${OV.fused_errors}`, 8.05, 4.0, 4.43, 0.85, { fontSize: 40, bold: true, valign: "middle" });
  T(s, `errors, compared with MUSK alone: fusion fixes ${FJ.mcnemar_fused_vs_B.b} and adds ${FJ.mcnemar_fused_vs_B.c}.`, 8.05, 4.88, 4.43, 1.2, { fontSize: 18 });
  source(s, "Tile chosen by rule: the median, by fused top probability, of the tumour tiles that one model misses and fusion gets right. Test: CRC-VAL-HE-7K.");
  script(s, {
    say: [
      "How do we combine two probes? The simplest way possible: add the probabilities and divide by two. No new parameters.",
      `Here is a real test tile, truly tumour. CONCH says tumour, ${p(ex.conch.TUM)} percent. MUSK says ${other.toLowerCase()}, ${p(ex.musk.other)} percent. The average says tumour, ${p(ex.fused.TUM)} percent. The more certain model carries the call.`,
      `And because it is only ${p(ex.fused.TUM)} percent sure, the review flag fires, so a pathologist sees this tile too.`,
      "We tested three rules on validation, and averaging won by a single tile. On test, the fancier neural network fell below MUSK alone, while the simple rule held up.",
      `Against MUSK alone, fusion fixes ${FJ.mcnemar_fused_vs_B.b} errors and adds ${FJ.mcnemar_fused_vs_B.c}: net ${OV.musk_errors - OV.fused_errors} fewer.`,
    ],
    asked: [
      "The winning rule went into the pre-registration before any test label was read, so we could not pick the best-looking rule afterwards.",
      `A one-tile margin means the choice of rule is fragile, and validation is near ceiling at ${pct2(FR.val_fused_bal_acc)}. See the backup slide.`,
      `Fused versus MUSK: McNemar p = ${FJ.mcnemar_fused_vs_B.p.toFixed(4)}, tile-level and optimistic because there are no slide IDs. All ${ex.n_rescued_flagged} of the ${ex.n_rescued} tumour tiles of this kind were flagged. This tile is the median of that group, not a best case.`,
      "The review flag: s3 = 1 minus the fused top probability. The pre-registered s4 adds disagreement signals and ranks tiles almost identically.",
    ],
    sources: "results/cm_local/CM/crc/fusion.json, reader_error_overlap.json, tile_preds.npz via docs/deck/make_assets.py, src/twinread/fusion.py, results/headline.json.",
  });
}

// ---- fusion result -------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  title(s, `Fusion adds ${pts(CC.delta_vs_best_single_pts)} points on colorectal tiles only`);
  T(s, "Balanced accuracy, colorectal test set", LM, 1.5, 3.8, 0.35, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  [[CONCH, CC.bal_acc.A, "CONCH alone"], [MUSK, CC.bal_acc.B, "MUSK alone"], [FUSED, CC.bal_acc.fused, "CONCH + MUSK fused"]].forEach(([col, v, lab], i) => {
    const y = 1.95 + i * 1.05;
    dot(s, LM, y + 0.17, 0.24, col, lab + " marker");
    T(s, pct2(v), LM + 0.36, y, 3.3, 0.58, { fontSize: 36, bold: true, valign: "middle" });
    T(s, lab, LM + 0.36, y + 0.58, 3.3, 0.3, { fontSize: 15, color: MUTED });
  });
  card(s, LM, 5.15, 3.75, 1.05);
  T(s, `+${FR.per_class_recall_delta_fused_minus_musk_pts.TUM.toFixed(1)} points of tumour recall: most of the net gain`, LM + 0.2, 5.2, 3.35, 0.95, { fontSize: 16, bold: true, valign: "middle" });

  // forest plot: native scatter with custom error bars (added in patchCharts)
  const cx = 4.65, cy = 1.9, cw = 8.08, ch = 3.85, f = { x: 0.325, y: 0.02, w: 0.445, h: 0.83 };
  const rows = [
    { lab: "CONCH + MUSK", sub: "Colorectal", d: CC, hi: true },
    { lab: "CONCH + MUSK", sub: "Camelyon17", d: CA },
    { lab: "Open control pair", sub: "Colorectal", d: OC },
    { lab: "Open control pair", sub: "Camelyon17", d: OA },
  ];
  T(s, "Fused minus best single reader, 95% CI", cx + cw * f.x - 0.6, 1.5, 5, 0.35, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  const blank = (n, at, v) => Array.from({ length: n }, (_, i) => (at.includes(i) ? v : null));
  const nRow = 12, yOf = (i) => 4 - i; // rows 0-7: CI ends; rows 8-11: point estimates
  s.addChart(pres.charts.SCATTER, [
    { name: "x", values: [...rows.flatMap((r) => r.d.delta_ci_pts), ...rows.map((r) => r.d.delta_vs_best_single_pts)] },
    ...rows.map((r, i) => ({ name: `95% CI: ${r.lab}, ${r.sub}`, values: blank(nRow, [2 * i, 2 * i + 1], yOf(i)) })),
    { name: "forest-primary", values: blank(nRow, [8], yOf(0)) },
    { name: "forest-controls", values: Array.from({ length: nRow }, (_, j) => (j > 8 ? yOf(j - 8) : null)) },
  ], {
    x: cx, y: cy, w: cw, h: ch, layout: f, chartColors: [HEX.accent3, HEX.accent5], lineSize: 0, lineDataSymbolSize: 12,
    catAxisMinVal: -4, catAxisMaxVal: 2, catAxisMajorUnit: 2, valAxisMinVal: 0.5, valAxisMaxVal: 4.5, valAxisMajorUnit: 1,
    catAxisLabelFontSize: 14, valAxisLabelFontSize: 14, ...noGrid, ...CHART_TEXT, catAxisLabelColor: HEX.accent5,
    showCatAxisTitle: true, catAxisTitle: "Points of balanced accuracy", catAxisTitleFontSize: 14, catAxisTitleColor: HEX.accent5, catAxisTitleFontFace: "+mn-lt",
    showLegend: false, showTitle: false, catAxisLabelFormatCode: '+0;\u22120;0',
    altText: "Forest plot of fused minus best single reader for four conditions. Only CONCH + MUSK on colorectal tiles has a confidence interval above zero.",
  });
  rows.forEach((r, i) => {
    const yc = cy + ch * (f.y + f.h * (i + 0.5) / 4);
    T(s, [{ text: r.lab, options: { bold: !!r.hi, fontSize: 15, breakLine: true } }, { text: r.sub, options: { fontSize: 14, color: MUTED } }], cx, yc - 0.28, cw * f.x - 0.1, 0.56, { valign: "middle" });
    T(s, [{ text: pts(r.d.delta_vs_best_single_pts), options: { bold: true, fontSize: 16, breakLine: true } }, { text: ciTxt(r.d.delta_ci_pts), options: { fontSize: 13, color: MUTED } }], cx + cw * (f.x + f.w) + 0.2, yc - 0.28, 1.65, 0.56, { valign: "middle" });
  });
  source(s, `CIs: tile-level on colorectal (anti-conservative, no slide IDs), slide-clustered on Camelyon17. Test sets: ${int(TC.n_test)} and 1,500 tiles.`);
  const dt = FR.per_class_recall_delta_fused_minus_musk_pts;
  script(s, {
    say: [
      `Overall, on colorectal tiles: ${pct2(CC.bal_acc.fused)} against ${pct2(CC.bal_acc.B)} for MUSK alone, a gain of ${CC.delta_vs_best_single_pts.toFixed(2)} points, mostly from tumour tiles.`,
      "In this forest plot, each line is the gain over the best single model, with a 95 percent confidence interval. Crossing zero means no clear gain.",
      "Only the top row clears zero. On Camelyon17, from an unseen hospital, there is no gain, and our open control pair gains nothing on either dataset.",
      "So the gain is real, but small and dataset-dependent.",
    ],
    asked: [
      "We ran the open pair through the identical protocol, so a gain that came from the method alone would have shown up there too. It did not. Camelyon17 is lymph node tissue.",
      `The colorectal CI is tile-level and therefore optimistic, because this dataset has no slide IDs. Muscle and stroma recall swing by about 13 points in opposite directions and cancel. We first misread that swing as the source of the gain, and corrected it once we broke the gain down by class. See the backup slide.`,
    ],
    sources: "results/cm_local/CM/*/fusion.json, results/open/OPEN/*/fusion.json, results/headline.json.",
  });
}

// ---- triage -------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  const R = CC.review_caught_at_10pct;
  title(s, "Reviewing the least certain 10% catches 80% of errors");
  T(s, "Errors caught at a 10% review budget", LM, 1.5, 4.1, 0.35, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  dot(s, LM, 2.2, 0.24, FUSED, "fused marker");
  T(s, pct1(R.s3_fused_uncertainty), LM + 0.36, 1.86, 3.7, 0.92, { fontSize: 54, bold: true, valign: "middle" });
  T(s, "Fused reader’s uncertainty", LM + 0.36, 2.8, 3.7, 0.32, { fontSize: 16, bold: true });
  [[MUSK, R.B_own_confidence_own_errors, "MUSK alone"], [CONCH, R.A_own_confidence_own_errors, "CONCH alone"], [DISAGREE, R.s1_disagree, "Readers disagree"]].forEach(([col, v, lab], i) => {
    const y = 3.3 + i * 0.5;
    dot(s, LM, y + 0.1, 0.2, col, lab + " marker");
    T(s, pct1(v), LM + 0.36, y, 1.25, 0.42, { fontSize: 22, bold: true, valign: "middle" });
    T(s, lab, LM + 1.65, y, 2.5, 0.42, { fontSize: 16, valign: "middle" });
  });
  card(s, LM, 5.0, 4.15, 1.2);
  T(s, "The signal is the fused reader’s doubt, not disagreement.", LM + 0.2, 5.05, 3.75, 1.1, { fontSize: 18, bold: true, valign: "middle" });

  const rf = TC.review_fraction, keep = rf.map((v, i) => i).filter((i) => rf[i] <= 0.3 + 1e-9);
  const curve = (k) => keep.map((i) => +(100 * TC.curves[k].errors_caught[i]).toFixed(2));
  const xs = keep.map((i) => +(100 * rf[i]).toFixed(1));
  s.addChart(pres.charts.SCATTER, [
    { name: "x", values: xs },
    { name: "Readers disagree", values: curve("s1_disagree") },
    { name: "CONCH alone", values: curve("base_A_own_error") },
    { name: "MUSK alone", values: curve("base_B_own_error") },
    { name: "Fused uncertainty", values: curve("s3_fused_unc") },
    { name: "10% budget", values: xs.map((v) => (v === 10 ? 0 : null)) },
  ], {
    x: 5.0, y: 1.45, w: 7.73, h: 4.85, layout: { x: 0.095, y: 0.03, w: 0.70, h: 0.815 }, lineSize: 2, lineDataSymbol: "none",
    chartColors: [HEX.accent4, HEX.accent1, HEX.accent2, HEX.accent3, HEX.accent5],
    catAxisMinVal: 0, catAxisMaxVal: 30, catAxisMajorUnit: 10, valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25,
    catAxisLabelFontSize: 14, valAxisLabelFontSize: 14, ...CHART_TEXT, catAxisLabelColor: HEX.accent5,
    catGridLine: { style: "none" }, valGridLine: { color: "E8E6EE", size: 0.75 },
    showCatAxisTitle: true, catAxisTitle: "Tiles sent to review (%)", catAxisTitleFontSize: 14, catAxisTitleColor: HEX.accent5, catAxisTitleFontFace: "+mn-lt",
    showValAxisTitle: true, valAxisTitle: "Errors caught (%)", valAxisTitleFontSize: 14, valAxisTitleColor: HEX.accent5, valAxisTitleFontFace: "+mn-lt",
    showLegend: false, showTitle: false,
    altText: `Errors caught against tiles reviewed. At 10% review: fused uncertainty ${pct1(R.s3_fused_uncertainty)}, MUSK ${pct1(R.B_own_confidence_own_errors)}, CONCH ${pct1(R.A_own_confidence_own_errors)}, disagreement ${pct1(R.s1_disagree)}.`,
  });
  source(s, `CRC-VAL-HE-7K, ${int(TC.n_test)} tiles. Single readers flag their own errors by their own confidence.`);
  script(s, {
    say: [
      "The result that surprised us. We expected disagreement between the models to be the best reason to call a pathologist. It is not.",
      "We rank tiles by each warning signal, send the top 10 percent to review, and count the errors caught.",
      `The combined model's own uncertainty catches ${pct1(R.s3_fused_uncertainty)}. MUSK checking itself, ${pct1(R.B_own_confidence_own_errors)}. CONCH, ${pct1(R.A_own_confidence_own_errors)}.`,
      `Disagreement stalls at ${pct1(R.s1_disagree)}, because the models disagree on only ${pct1(CC.s1_disagree_rate)} of tiles.`,
      "So two models help by making one reader that knows better when it is unsure.",
    ],
    asked: [
      `We pre-registered a learned flag that adds disagreement to uncertainty. It ranks tiles almost identically to plain uncertainty, Spearman ${FR.s3_s4_spearman.toFixed(4)}, so it adds nothing.`,
      `On Camelyon17 fused uncertainty catches ${pct1(CA.review_caught_at_10pct.s3_fused_uncertainty)}, while the pre-registered flag reaches ${pct1(CA.review_caught_at_10pct.s4_learned)} against CONCH alone at ${pct1(CA.review_caught_at_10pct.A_own_confidence_own_errors)}, and it does not beat CONCH alone on cleared-tile accuracy. See the backup slide.`,
    ],
    sources: "results/cm_local/CM/crc/review.json, triage_curves.json.",
  });
}

// ---- blind spots --------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  const miss = BS.missed, nMiss = miss.length, op = CC.prereg_rule_test_operating_point;
  title(s, `${BS.n} tiles fooled both readers; ${nMiss === 6 ? "six" : nMiss} also escaped the flag`);
  const d = 1.9, g = 0.12, x0_ = LM, groups = [["TUM", "Truth: tumour. Both readers: normal."], ["STR", "Truth: stroma. Both readers: muscle."]];
  groups.forEach(([truth, lab], r) => {
    const y = 1.5 + r * (d + 0.42);
    T(s, lab, x0_, y, 3 * d + 2 * g, 0.34, { fontSize: 15, bold: true, valign: "middle" });
    miss.filter((m) => m.truth === truth).slice(0, 3).forEach((m, i) => s.addImage({
      path: A(`missed_${m.tile}.jpg`), x: x0_ + i * (d + g), y: y + 0.36, w: d, h: d,
      altText: `Test tile ${m.tile}: labelled ${m.truth}, both readers called ${m.call}, and the review flag did not fire.`,
    }));
  });
  const minConf = Math.min(...miss.flatMap((m) => [m.conch, m.musk]));
  const rx = 7.1, rw = 5.63;
  const stat = (y, n, txt) => { T(s, String(n), rx, y, 1.15, 0.85, { fontSize: 44, bold: true, valign: "middle" }); T(s, txt, rx + 1.2, y, rw - 1.2, 0.85, { fontSize: 16, valign: "middle" }); };
  stat(1.55, BS.n, `tiles (${pct2(CC.confident_shared_errors.rate)}): both wrong (>80% confidence)`);
  stat(2.6, BS.flagged, `caught by the pre-registered flag, which reviews ${pct1(op.flagged_fraction)} of tiles`);
  stat(3.65, nMiss, `escaped it (shown): both readers at least ${(Math.floor(minConf * 100) / 100).toFixed(2)} confident`);
  card(s, rx, 4.8, rw, 1.3);
  T(s, "Two models agreeing is not proof. A pathologist stays in the loop.", rx + 0.25, 4.85, rw - 0.5, 1.2, { fontSize: 20, bold: true, valign: "middle" });
  source(s, `CRC-VAL-HE-7K, truth as labelled in the dataset. ${BS.pairs["STR->MUS"] + BS.pairs["MUS->STR"]} of the ${BS.n} are muscle vs stroma; ${BS.pairs["TUM->NORM"]} are tumour called normal.`);
  script(s, {
    say: [
      "The limit of the idea: shared blind spots, where both models make the same wrong call and both are at least 80 percent sure.",
      `There are ${BS.n}. Most are muscle versus stroma, but ${BS.pairs["TUM->NORM"]} are tumour called normal. The flag catches ${BS.flagged}; these ${nMiss} got through.`,
      "Two models agreeing is not proof, so a pathologist stays in the loop.",
    ],
    asked: [
      `We picked these by a fixed rule, not by hand. On Camelyon17 the flag catches only ${CA.confident_shared_errors.flagged_by_prereg_rule} of ${CA.confident_shared_errors.n} such tiles.`,
    ],
    sources: "results/cm_local/CM/crc/review.json, triage_curves.json, tile_preds.npz.",
  });
}

// ---- why quantise, why a small device: the reasons, before the board results
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  const ck = MS.ratios.CM.teacher_checkpoint_bytes_sum / 1e6, fp = MS.students.CM.fp32_param_bytes / 1e6, i8 = MS.students.CM.int8_onnx_bytes / 1e6;
  const row = (prec) => X86.rows.find((r) => r.backbone === "mobilenetv4_conv_small" && r.precision === prec && r.threads === 4);
  const xSpeed = row("fp32").p50_ms / row("int8").p50_ms, xRam = row("fp32").rss_mb / row("int8").rss_mb;
  const fpAcc = KC.cm.fp32_student_bal_acc, bdAcc = KC.cm.bal_acc.board, agree = KC.cm.pred_agreement_with_fp32_student.board;
  const PRICE = 79; // USD, 4 GB board (ours has 3.6 GiB usable), Arduino blog 2026-06-26, effective 2026-07-06
  const MARIA_PCT = 95.6; // FCC Communications Status Report for Areas Impacted by Hurricane Maria, 2017-09-23 (DOC-346860A1)
  const MARS_MIN = 22; // maximum one-way delay, McBrayer, Chai, Judd (NASA Langley), ASCEND 2022, NTRS 20220013418
  title(s, "Why shrink it: AI that works without a network");
  T(s, "Model size (MB)", LM, 1.25, 6, 0.32, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  hBars(s, ["CONCH + MUSK checkpoints", "Student, fp32", "Student, int8"], [ck, fp, i8].map((v) => +v.toFixed(1)), [BASE, BASE, HEX.accent3], {
    x: LM, y: 1.55, w: CW, h: 1.4, max: ck * 1.1, series: "Model size (MB)", fmt: "#,##0.0", font: 15, gap: 30, layout: { x: 0.215, y: 0.02, w: 0.775, h: 0.96 },
    alt: `Model size in MB: CONCH and MUSK checkpoints ${ck.toFixed(0)}, student fp32 ${fp.toFixed(1)}, student int8 ${i8.toFixed(1)}.`,
  });
  const bullets = (x, w, head, items) => {
    T(s, head, x, 3.08, w, 0.42, { fontSize: 20, bold: true, valign: "middle" });
    T(s, items.flatMap(([lead, rest], i) => [
      { text: lead + " ", options: { bold: true, bullet: { indent: 16 } } },
      { text: rest, options: { breakLine: i < items.length - 1 } },
    ]), x, 3.54, w, 2.75, { fontSize: 17, paraSpaceAfter: 9 });
  };
  bullets(LM, 5.8, "Why quantise to int8", [
    ["Smaller:", `${fp.toFixed(1)} MB down to ${i8.toFixed(1)} MB, ×${(fp / i8).toFixed(1)}`],
    ["Faster and lighter:", `×${xSpeed.toFixed(1)} faster and ×${xRam.toFixed(1)} less RAM on x86`],
    ["Small cost:", `${pct2(fpAcc)} to ${pct2(bdAcc)} balanced accuracy; ${pct1(agree)} of calls unchanged`],
  ]);
  bullets(6.93, 5.8, "Why run on a small device", [
    ["Low-access clinics:", "no GPU, server or network"],
    ["Disaster relief:", `Hurricane Maria knocked out ${MARIA_PCT}% of Puerto Rico\u2019s cell sites`],
    ["Mars:", `up to ${MARS_MIN} minutes each way to Earth, so decisions must be local`],
    ["Privacy and cost:", `images stay local; $${PRICE} list price`],
  ]);
  source(s, "Checkpoints include text encoders. Speed and RAM: MobileNetV4 backbone, 4 threads, x86; fp32 not timed on the board. Sources: Wilson et al., Lancet 2018; FCC 2017; McBrayer et al., NASA, 2022; Arduino 2026.");
  script(s, {
    say: [
      "Two big models and a GPU means over two gigabytes of checkpoints, which the places with the least access cannot run.",
      `A small device could serve a clinic with no GPU or second pathologist, or a disaster zone: after Hurricane Maria, ${MARIA_PCT} percent of Puerto Rico's cell sites were down.`,
      `An extreme case, not pathology but the same principle: Mars is up to ${MARS_MIN} minutes from Earth each way, so a rover cannot wait for a server.`,
      `Quantising stores each number in 8 bits rather than 32. Our student goes from ${fp.toFixed(1)} to ${i8.toFixed(1)} megabytes, and ${pct1(agree)} of its calls do not change.`,
    ],
    asked: [
      "These are reasons to build this way, not deployments. TwinRead has not been field-tested in a clinic or disaster zone, and this board is not space-rated.",
      `Wilson et al. name workforce and infrastructure among the barriers to pathology access in low- and middle-income countries. The ${MARIA_PCT}% is the FCC snapshot of 23 September 2017; some providers had not reported. NASA's AEGIS on Perseverance targeted rocks without direction from Earth from May 2022.`,
      `Speed and RAM are for the MobileNetV4 backbone alone on x86 at 4 threads (${row("fp32").p50_ms.toFixed(2)} vs ${row("int8").p50_ms.toFixed(2)} ms, ${row("fp32").rss_mb.toFixed(0)} vs ${row("int8").rss_mb.toFixed(0)} MB). We did not time fp32 on the UNO Q and did not measure power. $${PRICE} is the 4 GB list price since 6 July 2026; the 2 GB board is $59.`,
    ],
    sources: "results/student/model_sizes.json, results/uno_q/backbone_latency_x86_sim.json, results/uno_q/onboard/int8_kernel_check.json, FCC report DOC-346860A1, Wilson et al. Lancet 2018, NASA NTRS 20220013418, NASA Science on AEGIS (Sol 442), Arduino blog 26 June 2026.",
  });
}

// ---- student model: distillation and architecture ---------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  const st = MS.students.CM, wt = SM.loss_weights;
  title(s, "Distillation: one small student learns from both models");
  const head = (txt, x, w) => T(s, txt, x, 1.2, w, 0.32, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  head("Input", LM, 2); head("Shared trunk", 3.05, 2.2); head("Four heads", 5.8, 2.9); head("Learns from the teachers", 9.25, 3.48);
  s.addImage({ path: A("class_TUM.jpg"), x: LM, y: 1.58, w: 1.45, h: 1.45, altText: "An example H&E tile" });
  T(s, "224 px tile, with camera effects simulated by K2", LM, 3.08, 2.0, 0.75, { fontSize: 14, color: MUTED });
  arrow(s, 2.62, 2.3, 0.39, 0);
  box(s, [
    { text: "MobileNetV4 conv-small", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: `${(st.params_trunk / 1e6).toFixed(2)} M parameters`, options: { fontSize: 14 } },
  ], 3.05, 1.58, 2.2, 2.3, FUSED, WHITE);
  [
    ["Head A: 512 numbers", "CONCH embedding", CONCH],
    ["Head B: 1,024 numbers", "MUSK embedding", MUSK],
    ["Tissue call: 9 classes", "Fused probabilities and true labels", FUSED],
    ["Review score", "Teacher’s review flag", MUTED],
  ].forEach(([h, t_, col], i) => {
    const y = 1.58 + i * 0.59, ht = 0.53, c = y + ht / 2;
    arrow(s, 5.28, c, 0.48, 0);
    box(s, h, 5.8, y, 2.9, ht, TINT, INK, { fontSize: 16, bold: true });
    arrow(s, 8.74, c, 0.47, 0, { line: { color: MUTED, width: 1.75, beginArrowType: "triangle" } });
    box(s, t_, 9.25, y, 3.48, ht, col, WHITE, { fontSize: 15 });
  });
  T(s, "How it trains", LM, 4.15, CW, 0.38, { fontSize: 18, bold: true, valign: "middle" });
  T(s, [
    ["Copy the teachers:", "match each teacher’s embedding, the fused soft labels and the review flag"],
    ["Learn the truth:", "true tissue labels as well, so it is not only imitating"],
    ["Survive a camera:", "K2 simulates a USB camera filming a screen during training"],
    ["Run anywhere:", "teachers are needed for training only, never on the board"],
  ].flatMap(([lead, rest], i, all) => [
    { text: lead + " ", options: { bold: true, bullet: { indent: 16 } } },
    { text: rest, options: { breakLine: i < all.length - 1 } },
  ]), LM, 4.53, CW, 1.8, { fontSize: 17, paraSpaceAfter: 5 });
  source(s, `Loss weights: embeddings ${wt.cos}, fused soft labels ${wt.kd} (T = ${SM.T}), true labels ${wt.ce}, review ${wt.rev}. ${SM.epochs} epochs, batch ${SM.batch}. Training targets are out-of-fold. Heads A and B share one trunk, so they are far less independent than CONCH and MUSK.`);
  script(s, {
    say: [
      "This is the part we built and trained ourselves, and it is the heart of the project.",
      "Distillation means a small student model learns to imitate big teacher models. It sees the same tile and is scored on how closely it matches each teacher. The teachers are only needed during training; the board never needs them.",
      `Our student is a MobileNet, designed for small devices: one shared trunk of ${(st.params_trunk / 1e6).toFixed(1)} million parameters feeding four small heads.`,
      "Two heads copy the descriptions from CONCH and MUSK. One learns the combined tissue call. One learns when the teachers would send a tile for review.",
      "So one small network carries both readers, the fusion and the review flag.",
    ],
    built: [
      "We first tried an extra loss term to preserve similarities between tiles. On validation it made the student match its teachers worse, so we turned it off.",
    ],
    asked: [
      `Head A and head B are projection heads, cosine-matched to the text-aligned embeddings. Tissue call: temperature-${SM.T} KL to the fused logits plus cross-entropy. Review score: binary cross-entropy on the teacher flag.`,
      `Agreement with the fused teacher on clean test tiles is ${pct1(SM.test.clean.agree_with_fused_teacher)}. The trained student is fine-tuned with labels, so it is not a like-for-like test against frozen probes.`,
      `Training targets are cross-fitted, 5-fold, so the student never copies a teacher that memorised that tile. Best epoch was ${SM.best_epoch} of ${SM.epochs}, chosen on validation.`,
    ],
    sources: "results/student/cm_crc/metrics.json, results/student/model_sizes.json, results/headline.json, src/twinread/student.py, docs/progress.md (R26).",
  });
}

// ---- what our training achieved -----------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  const c = S.CM.fp32_test_bal_acc, tc = SM.test.clean, step = K.student_train_step_images_per_s.CM;
  const behind = 100 * (CC.bal_acc.fused - D.cm_int8_bal_acc_board), pctSize = S.params_pct_of_teacher_vision_towers;
  title(s, `Our trained student: ${pctSize.toFixed(1)}% of the size, ${behind.toFixed(1)} points behind`);
  T(s, "Balanced accuracy, colorectal test set (%)", LM, 1.2, 7, 0.32, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  hBars(s, ["CONCH + MUSK fused", "MUSK", "CONCH", "Our student", "Our student, int8 on the board"], [CC.bal_acc.fused, CC.bal_acc.B, CC.bal_acc.A, c.clean, D.cm_int8_bal_acc_board].map((v) => +(100 * v).toFixed(2)), [BASE, BASE, BASE, HEX.accent3, HEX.accent3], {
    x: LM, y: 1.5, w: 7.2, h: 2.85, max: 118, series: "Balanced accuracy (%)", fmt: "0.00", font: 15, gap: 30, layout: { x: 0.4, y: 0.02, w: 0.58, h: 0.96 },
    alt: `Balanced accuracy: fused ${pct2(CC.bal_acc.fused)}, MUSK ${pct2(CC.bal_acc.B)}, CONCH ${pct2(CC.bal_acc.A)}, our student ${pct2(c.clean)}, and int8 on the board ${pct2(D.cm_int8_bal_acc_board)}.`,
  });
  T(s, "Our student under simulated camera effects (%)", 8.2, 1.2, 4.53, 0.32, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  hBars(s, ["None", "Medium", "Strongest"], [c.clean, c["sev0.5"], c["sev1.0"]].map((v) => +(100 * v).toFixed(2)), [HEX.accent3, HEX.accent3, HEX.accent3], {
    x: 8.2, y: 1.5, w: 4.53, h: 2.85, max: 125, series: "Balanced accuracy under K2 (%)", fmt: "0.00", font: 15, gap: 45, layout: { x: 0.26, y: 0.02, w: 0.72, h: 0.96 },
    alt: `Student balanced accuracy: ${pct2(c.clean)} with no camera effects, ${pct2(c["sev0.5"])} at severity 0.5, ${pct2(c["sev1.0"])} at severity 1.0.`,
  });
  const stat = (x, w, big, txt) => {
    T(s, big, x, 4.5, w, 0.75, { fontSize: 36, bold: true, valign: "middle" });
    T(s, txt, x, 5.27, w, 0.95, { fontSize: 16 });
  };
  stat(LM, 3.85, `${SM.epochs} epochs`, `on ${int(SM.n.train)} labelled tiles, on one RTX 5060`);
  stat(4.75, 3.85, `${int(step.k2)}/s`, "images through a full training step, with K2 camera effects");
  stat(8.9, 3.83, pct1(tc.agree_with_fused_teacher), "of calls on clean test tiles match the fused teacher");
  source(s, "The student is fine-tuned with labels, so it is not a like-for-like test of features. Camera effects are the K2 simulation, not real captures; the teachers were not run under them.");
  script(s, {
    say: [
      "Here is what our training achieved, on test tiles it never saw.",
      `Before quantising, the student scores ${pct2(c.clean)}. That ties CONCH and sits close to MUSK, with ${pctSize.toFixed(1)} percent of their vision parameters.`,
      `On the board, after quantising, it scores ${pct2(D.cm_int8_bal_acc_board)}: ${behind.toFixed(1)} points behind the combined model. It matches the combined teacher on ${pct1(tc.agree_with_fused_teacher)} of calls.`,
      `We trained it on simulated camera damage, because a cheap camera filming a screen is what this board would see. At the strongest setting it still reaches ${pct2(c["sev1.0"])}.`,
      "The honest cost: it is below MUSK and fusion, and two heads share one trunk, so they are not truly independent readers.",
    ],
    asked: [
      "The student's trunk is fine-tuned end to end with labels, while the teachers are frozen features plus probes. So matching them is expected and does not show better features than the foundation models.",
      `Before quantising, its flag agrees with the teacher flag on ${pct1(tc.device_flag.agreement)} of tiles, kappa ${tc.device_flag.kappa.toFixed(2)}. Its review score has AUROC ${tc.review_auroc.toFixed(2)} against the teacher flag.`,
      `Quantising costs about a point on the board, from ${pct2(c.clean)} to ${pct2(D.cm_int8_bal_acc_board)}. The same int8 model scored ${pct2(D.cm_int8_bal_acc_x86)} on x86 with default kernels, an onnxruntime artefact.`,
    ],
    sources: "results/student/cm_crc/metrics.json, results/student/model_sizes.json, results/student/cm_crc/throughput.json, results/uno_q/onboard/testset_score.json, results/headline.json.",
  });
}

// ---- kernels ------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  const pre = K.k1_e2e_preprocess_tiles_per_s, aug = K.student_train_aug_images_per_s.CM, step = K.student_train_step_images_per_s.CM;
  const augX = K.student_train_aug_speedup_k2_vs_torch.CM, stepX = step.k2 / step.torch;
  title(s, `CUDA kernels: ${x0(K.k1_preprocess_speedup_vs_pil)} preprocessing, ${x1(augX)} augmentation`);
  const col = (x, big, name, sub, labels, vals, colors, whole, wholeTxt, alt) => {
    T(s, big, x, 1.4, 1.95, 0.85, { fontSize: 44, bold: true, valign: "middle" });
    T(s, name, x + 2.0, 1.4, 3.8, 0.4, { fontSize: 18, bold: true, valign: "middle" });
    T(s, sub, x + 2.0, 1.8, 3.8, 0.72, { fontSize: 14, color: MUTED });
    hBars(s, labels, vals, colors, { x, y: 2.72, w: 5.8, h: 1.7, max: Math.max(...vals) * 1.38, series: name, alt, font: 15, gap: 35, layout: { x: 0.27, y: 0.03, w: 0.71, h: 0.94 } });
    card(s, x, 4.6, 5.8, 1.5);
    T(s, whole, x + 0.25, 4.72, 1.75, 1.26, { fontSize: 32, bold: true, valign: "middle" });
    T(s, wholeTxt, x + 2.05, 4.72, 3.55, 1.26, { fontSize: 16, valign: "middle" });
  };
  col(LM, x0(K.k1_preprocess_speedup_vs_pil), "K1: dual-reader preprocessing", "One tile in, CONCH 448 px and MUSK 384 px out, bit-exact to the official transforms",
    ["PIL (official)", "K1 kernel"], [pre.pil_reference_cpu, pre.k1_gpu_incl_h2d], [BASE, HEX.accent3], x2(K.k1_e2e_speedup_vs_reference),
    `whole embedding run: the model forward is ${Math.round(100 * K.model_forward_share_of_reference_e2e)}% of the time`,
    `Tiles per second inside the embedding run: PIL ${int(pre.pil_reference_cpu)}, K1 ${int(pre.k1_gpu_incl_h2d)}.`);
  col(6.93, x1(augX), "K2: training augmentation", "Nine dependent stages plus JPEG, each within one grey level of a float64 reference",
    ["torch GPU", "K2 kernel"], [aug.torch, aug.k2], [BASE, HEX.accent3], x2(stepX),
    `whole student training step: ${int(step.torch)} to ${int(step.k2)} images/s`,
    `Augmentation images per second inside student training: torch ${int(aug.torch)}, K2 ${int(aug.k2)}.`);
  source(s, "RTX 5060; compile-checked for Delta A100 and DeltaAI GH200; compute-sanitizer clean. Tiles/s (K1), images/s (K2). K2 simulates a USB camera filming a screen.");
  script(s, {
    say: [
      "Training like that needs heavy image processing, so we wrote two small programs that run directly on the GPU.",
      `K2, the camera simulator, is ${augX.toFixed(1)} times faster than torch, which makes a full training step ${stepX.toFixed(2)} times faster.`,
      `K1 prepares tiles for both models, pixel-exact and ${Math.round(K.k1_preprocess_speedup_vs_pil)} times faster. The whole run gains only ${K.k1_e2e_speedup_vs_reference.toFixed(2)} times, because the models take most of the time.`,
    ],
    asked: [
      `With input already on the GPU, K1 reaches ${int(K.k1_cuda_tiles_per_s_224)} tiles per second. K2 reaches ${int(K.k2_cuda_images_per_s_sev1)} images per second standalone. Training with no augmentation at all runs at ${int(step.none)} images per second, so K2 is close to free. See the backup slide.`,
      "We built K1 because the organizers saw CPU preprocessing as a bottleneck on a hosted per-image server. Our batched run is a different regime: preprocessing was 9.5 percent of it, which is why we quote both numbers.",
    ],
    sources: "results/kernels/*.json, results/student/*/throughput.json.",
  });
}

// ---- dependent kernels: how the speed is achieved. Graphs from make_kernel_figures.py.
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  title(s, "Fusing dependent stages avoids DRAM round trips");
  card(s, LM, 1.12, CW, 0.82);
  T(s, [
    { text: "A dependent kernel ", options: { bold: true } }, { text: "chains stages where each one needs the previous stage’s output. " },
    { text: "Fusing ", options: { bold: true } }, { text: "runs the chain in one launch, with intermediates in on-chip shared memory." },
  ], LM + 0.22, 1.12, CW - 0.44, 0.82, { fontSize: 17, valign: "middle" });
  const floorPct = Math.round(100 * (KF.k1_over_floor - 1));
  [
    { x: LM, h: "Why fuse", t: `One launch per stage sends every intermediate through DRAM: ×${KF.traffic_ratio.toFixed(1)} the traffic.`, img: "kernel_traffic.png",
      alt: `Cumulative DRAM traffic along K2's ten stages: ${KF.unfused_gb.toFixed(2)} GB with one launch per stage, ${KF.fused_gb.toFixed(2)} GB fused. Calculated from tensor sizes.` },
    { x: 4.72, h: "What it accomplishes", t: `K2 takes ${KF.k2_ms.toFixed(1)} ms per batch, less than the unfused chain’s memory traffic alone.`, img: "kernel_k2_time.png",
      alt: `Measured time per batch: K2 ${KF.k2_ms.toFixed(1)} ms, torch GPU ${KF.torch_ms.toFixed(0)} ms, torch with cheaper noise ${KF.torch_randn_ms.toFixed(0)} ms, numpy over a second. Unfused memory traffic alone would take ${KF.unfused_traffic_ms.toFixed(1)} ms.` },
    { x: 8.84, h: "Why it’s cool", t: `K1 runs within ${floorPct}% of the memory-bandwidth floor and stays bit-exact.`, img: "kernel_k1_time.png",
      alt: `Measured time per batch: K1 ${KF.k1_ms.toFixed(2)} ms against a calculated memory floor of ${KF.k1_floor_ms.toFixed(2)} ms; Triton ${KF.k1_triton_ms.toFixed(2)} ms; torch interpolate ${KF.k1_interpolate_ms.toFixed(0)} ms, not bit-exact.` },
  ].forEach((c) => {
    T(s, c.h, c.x, 2.05, 3.89, 0.32, { fontSize: 17, bold: true, valign: "middle" });
    T(s, c.t, c.x, 2.37, 3.89, 0.52, { fontSize: 14 });
    s.addImage({ path: A(c.img), x: c.x, y: 2.92, w: 3.89, h: 3.45, altText: c.alt });
  });
  source(s, `DRAM traffic and floors are calculated from tensor sizes and the measured memcpy rate (${Math.round(KF.memcpy_gbps)} GB/s), not hardware counters. Times are CUDA events on a shared RTX 5060. No hand-written unfused CUDA version was measured.`);
  script(s, {
    say: [
      "The idea behind the speed: K2 is a chain of ten steps, each needing the result of the step before.",
      `Normally every result goes out to GPU memory and back. We run nine steps in one pass using fast on-chip memory, so memory traffic falls ${KF.traffic_ratio.toFixed(1)} times. The graphs are here for questions.`,
    ],
    asked: [
      `Traffic is calculated from tensor sizes: ${KF.unfused_gb.toFixed(2)} GB per batch unfused, ${KF.fused_gb.toFixed(2)} GB fused. K2 takes ${KF.k2_ms.toFixed(1)} ms, less than the unfused traffic alone would take; torch GPU ops take ${KF.torch_ms.toFixed(0)} ms.`,
      `Fusing alone does not explain the whole ×${Math.round(KF.x_torch)}. We did not measure a hand-written unfused CUDA version. The memory floor shows it would take at least ${KF.unfused_traffic_ms.toFixed(1)} ms plus compute, so fusing is worth at least ×${(KF.unfused_traffic_ms / KF.k2_ms).toFixed(2)}, and the rest is torch overhead.`,
      `K1 runs within ${floorPct} percent of the floor set by GPU memory bandwidth and matches the official transforms bit for bit. torch interpolate is ${Math.round(KF.k1_interpolate_ms / KF.k1_ms)} times slower and up to ${Math.round(KF.interpolate_max_levels)} grey levels off. Our Triton version of the same plan is within ${Math.ceil(100 * (KF.k1_triton_ms / KF.k1_ms - 1))}% of K1, so most of the gain comes from the design, not the language.`,
      `×${Math.round(KF.x_torch)} is the standalone bench at batch 256, severity 1.0. ×${K.student_train_aug_speedup_k2_vs_torch.CM.toFixed(1)} on the previous slide is inside the training loop at batch 128, severity 0.5. Every K2 stage is within one grey level of a float64 numpy reference, and compute-sanitizer memcheck, racecheck and synccheck are clean.`,
    ],
    sources: "results/kernels/k1_bench.json, k1_parity.json, k2_bench.json, k2_sanitizer.json, src/twinread/csrc/k2_camera_aug.cu.",
  });
}

// ---- UNO Q setup: what runs where ---------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  const st = D.stage_p50_ms, ms = (v) => `${v.toFixed(1)} ms`;
  title(s, "UNO Q setup: what runs where");
  card(s, LM, 1.15, CW, 2.45);
  T(s, [{ text: "Linux side   ", options: { bold: true } }, { text: `Qualcomm QRB2210, 4× Cortex-A53 at ${D.cpu_max_ghz.toFixed(1)} GHz, ${(D.mem_total_mib / 1024).toFixed(1)} GiB RAM, Debian 13, no GPU` }], LM + 0.25, 1.22, CW - 0.5, 0.42, { fontSize: 17, valign: "middle" });
  const steps = [
    ["Frame in", "stored tile; USB camera next", st["frame read (PNG decode)"]],
    ["Crop, tissue check", "numpy", st["crop to 224 px"] + st["tissue check"]],
    ["int8 student", "onnxruntime on CPU", D.cm_int8_p50_ms_4_threads, true],
    ["Probes, review rule", "numpy", st["probes + softmax"] + st["review rule"]],
    ["Web page", "call and flag, local only", st["tissue fraction again (UI)"] + st["UI frame (box + JPEG)"]],
  ];
  const w = 2.1, gap = 0.24, x0 = LM + 0.335;
  steps.forEach(([h, sub, t_, hi], i) => {
    const x = x0 + i * (w + gap);
    box(s, [{ text: h, options: { bold: true, fontSize: 16, breakLine: true } }, { text: sub, options: { fontSize: 13, color: hi ? WHITE : MUTED } }], x, 1.72, w, 1.2, hi ? FUSED : WHITE, hi ? WHITE : INK);
    if (i < steps.length - 1) arrow(s, x + w + 0.02, 2.32, gap - 0.04, 0);
    T(s, ms(t_), x, 2.98, w, 0.42, { fontSize: 18, bold: true, align: "center", valign: "middle" });
  });
  arrow(s, 5.3, 3.6, 0, 0.3);
  card(s, LM, 3.9, CW, 1.25);
  T(s, [{ text: "Microcontroller side", options: { bold: true, breakLine: true } }, { text: "STM32U585, Zephyr" }], LM + 0.25, 3.95, 3.3, 1.15, { fontSize: 17, valign: "middle" });
  [["Bridge call", `${ms(D.led_bridge_rpc_ms_per_call)} per message`], ["LED sketch", "flashed over USB-C"], ["8×13 LED matrix", "shows the status"]].forEach(([h, sub], i) => {
    const x = 4.1 + i * 2.95;
    box(s, [{ text: h, options: { bold: true, fontSize: 16, breakLine: true } }, { text: sub, options: { fontSize: 13, color: MUTED } }], x, 4.05, 2.55, 0.95, WHITE, INK);
    if (i < 2) arrow(s, x + 2.57, 4.52, 0.36, 0);
  });
  [["USB-C", "power; draw not measured"], ["Offline", "the app binds to the board and makes no network calls"], [`About ${Math.round(D.app_frame_ms_p50_sum_of_stages)} ms`, "per app frame; the tissue check costs more than the model"]].forEach(([h, sub], i) => {
    const x = LM + i * 4.1;
    T(s, [{ text: h, options: { bold: true, fontSize: 24, breakLine: true } }, { text: sub, options: { fontSize: 15 } }], x, 5.27, 3.85, 1.05);
  });
  source(s, `Median stage times on the board at 4 threads. Web page time: second tissue pass plus JPEG. The model alone uses ${D.cm_int8_rss_mb.toFixed(0)} MB of RAM.`);
  script(s, {
    cue: "Hold up the board now, if you brought it.",
    say: [
      "The Arduino UNO Q is one small board with two computers on it.",
      "The Linux side has four low-power Arm cores, 3.6 gigabytes of memory and no GPU.",
      `We crop each frame and check it shows tissue. Our ${S.int8_onnx_mb.toFixed(0)} megabyte student runs in ${D.cm_int8_p50_ms_4_threads.toFixed(1)} milliseconds, then the probes and review rule make the call.`,
      "The microcontroller runs the LED matrix, which shows the status. Power comes over USB-C, and the app makes no network calls.",
    ],
    built: [
      "We timed every stage on the board. The surprise: our simple tissue check costs more time than the neural network.",
    ],
    asked: [
      "No tissue has been filmed through a real camera yet. Today's numbers use stored test tiles; camera robustness so far is the K2 simulation.",
      `Bridge calls: two per frame, ${ms(st["LED busy (Bridge RPC)"] + st["LED result (Bridge RPC)"])} in total. The sketch compiles for arduino:zephyr:unoq and an unknown method is refused. Bundles are checked against SHA256SUMS before loading.`,
    ],
    sources: "results/uno_q/onboard/board_info.json, stage_timing_cm.json, latency.json, results/uno_q/mcu_compile.json, device/uno_q_app.py, device/mcu/twinread_led/twinread_led.ino.",
  });
}

// ---- board results --------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  title(s, "Implementation on Arduino UNO Q");
  s.addImage({ path: A("uno_q.jpg"), x: LM, y: 1.5, w: 6.05, h: 4.8, altText: "The Arduino UNO Q running TwinRead, powered over USB-C, with its LED matrix lit" });
  const rx = 7.15, rw = 5.58;
  const stat = (y, n, txt) => { T(s, n, rx, y, rw, 0.82, { fontSize: 46, bold: true, valign: "middle" }); T(s, txt, rx, y + 0.84, rw, 0.62, { fontSize: 16, color: INK }); };
  stat(1.45, `${D.cm_int8_p50_ms_4_threads.toFixed(1)} ms`, "per tile, model inference: int8, 4 threads, offline");
  stat(3.05, pct2(D.cm_int8_bal_acc_board), `balanced accuracy on all ${int(TC.n_test)} test tiles, below every teacher (${(100 * CC.bal_acc.A).toFixed(2)}–${pct2(CC.bal_acc.fused)})`);
  stat(4.65, `${S.int8_onnx_mb.toFixed(1)} MB`, `${S.params_millions.toFixed(2)} M parameters: ${S.params_pct_of_teacher_vision_towers.toFixed(1)}% of the two teachers’ vision towers`);
  source(s, `All measured on the Arduino UNO Q: Qualcomm QRB2210, 4× Cortex-A53, 3.6 GiB RAM, onnxruntime on CPU. Stored test tiles; no camera captures yet.`);
  script(s, {
    say: [
      `Measured on the board: ${D.cm_int8_p50_ms_4_threads.toFixed(1)} milliseconds per tile, and ${pct2(D.cm_int8_bal_acc_board)} across all ${int(TC.n_test)} test tiles, in ${Math.round(D.cm_full_loop_seconds)} seconds.`,
    ],
    built: [
      `Testing on real hardware paid off. On our x86 machine the quantised model scored lower, ${pct2(D.cm_int8_bal_acc_x86)}. We traced that to one onnxruntime kernel. With it off, only ${KC.cm.board_vs_x86_noopt.pred_mismatches} of ${int(TC.n_test)} calls differ.`,
    ],
    asked: [
      `A full app frame takes about ${Math.round(D.app_frame_ms_p50_sum_of_stages)} ms and the full loop runs at ${D.cm_full_loop_tiles_per_s.toFixed(1)} tiles per second. The model uses ${D.cm_int8_rss_mb.toFixed(0)} MB of RAM and takes ${D.cm_int8_p50_ms_1_thread.toFixed(1)} ms on one thread. See the backup slide.`,
    ],
    sources: "results/uno_q/onboard/*.json, results/student/model_sizes.json.",
  });
}

// ---- how we built it ----------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT_PLAIN", sectionTitle: "Talk" });
  const collapse = DRIFT.candidates.minmax.label_agreement, kept = DRIFT.val_drift.label_agreement;
  title(s, "How we built it");
  const steps = [
    ["Plan", "Design, plan and data splits written down before the code"],
    ["Read", "CONCH and MUSK on one RTX 5060: 20k of 100k training tiles"],
    ["Freeze", "Rules chosen on validation and pre-registered; test scored once"],
    ["Train", "Our student distilled from both, with K2 camera effects, then int8"],
    ["Deploy", "Flashed to the UNO Q; everything re-measured on the board"],
  ];
  const w = 2.25, gap = 0.22;
  steps.forEach(([h, sub], i) => {
    const x = LM + i * (w + gap);
    box(s, String(i + 1), x, 1.22, 0.46, 0.46, FUSED, WHITE, { shape: pres.shapes.OVAL, fontSize: 16, bold: true, margin: 0, objectName: "step number" });
    if (i < steps.length - 1) arrow(s, x + 0.52, 1.45, w + gap - 0.58, 0);
    T(s, h, x, 1.76, w, 0.38, { fontSize: 20, bold: true, valign: "middle" });
    T(s, sub, x, 2.15, w - 0.05, 1.2, { fontSize: 15 });
  });
  T(s, "What broke", 0.85, 3.4, 5.6, 0.36, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  T(s, "What we did", 7.45, 3.4, 5, 0.36, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  [
    [`Two int8 calibration methods collapsed: only ${Math.round(100 * collapse)}% of calls survived`, `Percentile calibration: ${pct1(kept)} survive on validation`],
    [`int8 scored lower on x86 (${pct2(KC.cm.bal_acc.x86)}) than on the board (${pct2(KC.cm.bal_acc.board)})`, `Traced to one onnxruntime kernel, by running on real hardware`],
    ["CONCH and MUSK licences bar sharing their student", "Open control pair, with a shareable bundle"],
  ].forEach(([gap_, next], i) => {
    const y = 3.8 + i * 0.86;
    card(s, LM, y, CW, 0.76);
    T(s, gap_, 0.85, y, 5.75, 0.76, { fontSize: 16, valign: "middle" });
    arrow(s, 6.78, y + 0.38, 0.45, 0);
    T(s, next, 7.45, y, 5.05, 0.76, { fontSize: 16, bold: true, valign: "middle" });
  });
  script(s, {
    say: [
      "How it came together: we wrote the design and data splits down first, then ran both foundation models on one desktop GPU, on 20 thousand of 100 thousand training tiles.",
      "We froze our rules on validation and scored test once. Then we trained the student, quantised it, and re-measured everything on the board.",
      `Things broke. Our first quantisation attempts collapsed: only ${Math.round(100 * collapse)} percent of calls survived. A more careful calibration kept ${pct1(kept)}.`,
      "Licences on CONCH and MUSK stop us sharing the student, so we also built everything on an open model pair.",
      "We also misread our own result: we called the fusion gain a muscle and stroma trade. A per-class breakdown showed those cancel and the gain is mostly tumour, so we corrected every write-up.",
    ],
    asked: [
      "Calibration is how quantisation decides the range each 8-bit number covers. One extreme value can stretch that range and waste the precision. Percentile calibration ignores the extremes.",
      "The local subset keeps this reproducible on one GPU, and its limits are stated. The full-data kit for NCSA Delta is written and tested locally, and has not run there.",
    ],
    sources: "docs/design.md, docs/plan.md, docs/rulings.md (R28, R33), docs/progress.md, results/student/cm_crc/int8_drift.json, results/uno_q/onboard/int8_kernel_check.json.",
  });
}

// ---- future directions --------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT_PLAIN", sectionTitle: "Talk" });
  title(s, "Future Directions");
  const f = D.cm_board_device_flag, op = CC.prereg_rule_test_operating_point;
  T(s, "What remains", 0.85, 1.42, 5.6, 0.36, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  T(s, "Next step", 7.45, 1.42, 5, 0.36, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  [
    ["The fusion gain is fragile: validation chose the rule by 1–2 tiles", "Rerun on all 100k training tiles on NCSA Delta (kit ready)"],
    [`The pre-registered threshold flagged ${pct1(op.flagged_fraction)} of tiles, not 10%`, "Calibrate per site; add slide IDs for clustered CIs"],
    [`The board’s flag tracks the teachers weakly (κ ${f.kappa.toFixed(2)}), on stored tiles only`, "Real camera captures on the board, then clinical validation"],
  ].forEach(([gap, next], i) => {
    const y = 1.82 + i * 1.08;
    card(s, LM, y, CW, 0.96);
    T(s, gap, 0.85, y, 5.75, 0.96, { fontSize: 18, valign: "middle" });
    arrow(s, 6.78, y + 0.48, 0.45, 0);
    T(s, next, 7.45, y, 5.05, 0.96, { fontSize: 18, bold: true, valign: "middle" });
  });
  script(s, {
    say: [
      "The fusion gain is fragile, so next is a full run on NCSA Delta with all 100 thousand training tiles. Our job scripts are tested locally but have not run there yet.",
      `The pre-registered threshold flagged ${pct1(op.flagged_fraction)} of tiles for a 10 percent target, so thresholds need calibrating per site.`,
      `The board's flag tracks the big models only weakly, kappa ${f.kappa.toFixed(2)}, and it has only seen stored tiles, so real camera captures come next.`,
    ],
    asked: [
      "The board's two readers are two heads on one trunk, which is why its flag tracks the teachers weakly.",
      `The board flag covers ${pct1(f.fraction)} of tiles and catches ${pct1(f.errors_caught)} of its own errors. Kappa is agreement corrected for chance: 0 is chance, 1 is perfect.`,
    ],
    sources: "results/headline.json, ncsa/README.md.",
  });
}

// ---- why it matters ----------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT_PLAIN", sectionTitle: "Talk" });
  title(s, "Why it matters to pathologists and patients");
  [
    [FUSED, "For pathologists", "A short list of tiles worth re-checking, with each reader’s call shown."],
    [MUSK, "For patients", "A low-cost offline second look could help where no second pathologist is available."],
    [CONCH, "For the field", "Two readers help by making a better-calibrated reader, not through disagreement."],
  ].forEach(([col, h, txt], i) => {
    const y = 1.6 + i * 1.6;
    dot(s, LM, y + 0.07, 0.24, col, h + " marker");
    T(s, h, LM + 0.38, y, 5.5, 0.4, { fontSize: 22, bold: true, valign: "middle" });
    T(s, txt, LM + 0.38, y + 0.44, 5.5, 0.95, { fontSize: 18 });
  });
  s.addImage({ path: A("app_ui.png"), x: 7.0, y: 1.6, w: 5.73, h: 3.56, altText: "The TwinRead app on the UNO Q: a stored tile, both student heads, the fused call and an AGREE verdict" });
  T(s, "The app on the UNO Q, replaying a stored test tile. Its two heads share one student network.", 7.0, 5.24, 5.73, 0.55, { fontSize: 13, color: MUTED });
  script(s, {
    say: [
      "For pathologists: a short list of tiles worth a second look, with each model's call shown.",
      "For patients: a low-cost, offline second look could help where no second pathologist is available, pending validation.",
      "For the field: a pre-registered finding about why two readers help.",
      "To be clear, this is a research prototype, not a diagnostic device, and it is not clinically validated.",
    ],
    asked: [
      `The 53 ms in the screenshot covers the whole step: frame read, crop, tissue check and network, about ${Math.round(["frame read (PNG decode)", "crop to 224 px", "uint8 -> float NCHW", "int8 student (onnxruntime)", "probes + softmax", "tissue check", "review rule"].reduce((a, k) => a + D.stage_p50_ms[k], 0))} ms by stage medians. ${D.cm_int8_p50_ms_4_threads.toFixed(1)} ms is the network alone at 4 threads.`,
    ],
  });
}

// ---- close -------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CLOSE", sectionTitle: "Talk" });
  title(s, "TwinRead in three numbers");
  [
    [`${pts(CC.delta_vs_best_single_pts)} pts`, "balanced accuracy from fusion on colorectal tiles"],
    [pct1(CC.review_caught_at_10pct.s3_fused_uncertainty), "of errors caught by reviewing the least certain 10% of tiles"],
    [`${D.cm_int8_p50_ms_4_threads.toFixed(1)} ms`, `per tile on an offline board: a ${S.int8_onnx_mb.toFixed(1)} MB model, ${pct2(D.cm_int8_bal_acc_board)} accurate`],
  ].forEach(([n, txt], i) => {
    const x = LM + i * 4.16;
    T(s, n, x, 1.75, 3.9, 1.05, { fontSize: 54, bold: true, color: WHITE, valign: "middle" });
    T(s, txt, x, 2.9, 3.65, 1.1, { fontSize: 18, color: WHITE });
  });
  T(s, "Thank you. Questions?", LM, 4.55, CW, 0.6, { fontSize: 28, bold: true, color: WHITE, valign: "middle" });
  T(s, "Yubo Wu, Noah Zhang, William Liu, Franklin Liu, Andy Chen", LM, 5.25, CW, 0.32, { fontSize: 16, color: WHITE });
  T(s, "github.com/iujab/twinread-conch-musk-edge", LM, 5.6, CW, 0.32, { fontSize: 14, color: RULE });
  script(s, {
    say: [
      `Three numbers. Plus ${CC.delta_vs_best_single_pts.toFixed(2)} points from fusion on colorectal tiles, with no gain on Camelyon17. ${pct1(CC.review_caught_at_10pct.s3_fused_uncertainty)} of errors caught by reviewing 10 percent of tiles. And ${D.cm_int8_p50_ms_4_threads.toFixed(1)} milliseconds per tile, from a ${S.int8_onnx_mb.toFixed(0)} megabyte model we trained ourselves, on an offline board.`,
      "Thank you. We would welcome your questions.",
    ],
    after: (total) => `Leave this slide up during questions. Backup slides follow the references. Spoken total is about ${mmss(total)} at ${WPM} words a minute. To save time, skip the "How we built it" bullets outside the process slide.`,
  });
}

// ---- references and backup ------------------------------------------------------------------------------------------
pres.addSection({ title: "References and backup" });
{
  const s = pres.addSlide({ masterName: "CONTENT_PLAIN", sectionTitle: "References and backup" });
  title(s, "References");
  const refs = [
    "Elmore JG, et al. Diagnostic concordance among pathologists interpreting breast biopsy specimens. JAMA 2015. doi:10.1001/jama.2015.1405",
    "Fleming KA, et al. The Lancet Commission on diagnostics: transforming access to diagnostics. Lancet 2021. doi:10.1016/S0140-6736(21)00673-5",
    "Wilson ML, et al. Access to pathology and laboratory medicine services: a crucial gap. Lancet 2018. doi:10.1016/S0140-6736(18)30459-8",
    "Lu MY, et al. A visual-language foundation model for computational pathology (CONCH). Nat Med 2024. doi:10.1038/s41591-024-02856-4",
    "Xiang J, et al. A vision–language foundation model for precision oncology (MUSK). Nature 2025. doi:10.1038/s41586-024-08378-w",
    "Ikezogwo WO, et al. Quilt-1M: one million image-text pairs for histopathology (QuiltNet). NeurIPS 2023. arXiv:2306.11207",
    "Li et al. Multi-modal foundation models for computational pathology: a survey. TMLR 2025. PMC13081565",
    "Muneer A, et al. Review on multimodal data integration for cancer research. Artif Intell Rev 2026. doi:10.1007/s10462-026-11522-9",
    "Kather JN, et al. NCT-CRC-HE-100K and CRC-VAL-HE-7K (CC-BY-4.0). Bandi P, et al. Camelyon17; Koh PW, et al. WILDS (Camelyon17-WILDS, CC0).",
    "Open models: PathGen-CLIP (CC-BY-4.0), QuiltNet-B-16 (MIT), timm mobilenetv4_conv_small.",
    "FCC. Communications Status Report for Areas Impacted by Hurricane Maria. 23 September 2017.",
    "McBrayer KT, Chai PR, Judd EL. Communication Delays, Disruptions, and Blackouts for Crewed Mars Missions. NASA, ASCEND 2022. NTRS 20220013418.",
    "NASA Science. Perseverance\u2019s SuperCam uses AEGIS for the first time. 2022.",
    "Arduino. A heads-up on the Arduino UNO Q board pricing. Arduino blog, 26 June 2026.",
  ];
  T(s, refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 4 } })), LM, 1.2, CW, 5.05, { fontSize: 13 });
  T(s, "Code, results JSON and this deck’s generator: github.com/iujab/twinread-conch-musk-edge. Every TwinRead number here is read from results/*.json at build time.", LM, 6.3, CW, 0.5, { fontSize: 12, color: MUTED });
  s.addNotes("Not presented. Leave out of the talk unless asked.");
}
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "References and backup" });
  title(s, "Backup: data, splits and statistics");
  const rows = [
    [th("Split"), th("Data"), th("Tiles"), th("Used for")],
    ["Train", "NCT-CRC-HE-100K subset", int(SM.n.train), "Probes and student"],
    ["Validation", "Same source, group-disjoint", int(SM.n.val), "Fusion rule, thresholds"],
    ["Test", "CRC-VAL-HE-7K, 9 classes", int(TC.n_test), "Evaluated once, after pre-registration"],
    ["Test", "Camelyon17-WILDS, hospital 2", "1,500", "Unseen hospital; trained on hospitals 0, 3, 4"],
  ];
  s.addTable(rows, tableOpts({ x: LM, y: 1.45, w: CW, colW: [1.55, 4.2, 1.2, 5.18], rowH: 0.46 }));
  T(s, [
    { text: "Statistics. ", options: { bold: true } }, { text: "Percentile bootstrap, 1,000 resamples. Colorectal CIs and McNemar tests are tile-level and anti-conservative because the dataset has no slide IDs; Camelyon17 CIs are clustered by slide.", options: { breakLine: true } },
    { text: "Licences. ", options: { bold: true } }, { text: "CONCH and MUSK are CC-BY-NC-ND, so their student is never distributed. PathGen-CLIP + QuiltNet-B-16 is the control pair and the shareable bundle.", options: { breakLine: true } },
    { text: "Scale. ", options: { bold: true } }, { text: "Teacher numbers are from a local 20k-tile subset on one RTX 5060. The Delta kit is written and tested locally and has not run on Delta." },
  ], LM, 3.95, CW, 2.3, { fontSize: 16, paraSpaceAfter: 8 });
  source(s, "Sources: results/cm_local/CM/*/preregistration.json, results/student/cm_crc/metrics.json, ncsa/README.md.");
  s.addNotes("Backup for questions on data, leakage, statistics or licensing.");
}
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "References and backup" });
  title(s, "Backup: why the colorectal gain is fragile");
  const dd = FR.per_class_recall_delta_fused_minus_musk_pts;
  const names = { STR: "Stroma", TUM: "Tumour", ADI: "Adipose", DEB: "Debris", LYM: "Lymphocytes", NORM: "Normal mucosa", BACK: "Background", MUC: "Mucus", MUS: "Smooth muscle" };
  const ord = Object.keys(dd).sort((a, b) => dd[b] - dd[a]);
  s.addChart(pres.charts.BAR, [{ name: "Fused minus MUSK recall (points)", labels: ord.map((k) => names[k]), values: ord.map((k) => +dd[k].toFixed(2)) }], {
    x: LM, y: 1.5, w: 7.3, h: 4.75, barDir: "bar", chartColors: ord.map((k) => (k === "TUM" ? HEX.accent3 : BASE)), invertedColors: ord.map((k) => (k === "TUM" ? HEX.accent3 : BASE)), barGapWidthPct: 35,
    catAxisOrientation: "maxMin", valAxisMinVal: -16, valAxisMaxVal: 16, valAxisMajorUnit: 8, ...noGrid, ...CHART_TEXT, catAxisLabelFontSize: 14, valAxisLabelFontSize: 13, catAxisLabelPos: "low",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelFormatCode: "+0.0;−0.0;0.0", dataLabelColor: HEX.dk1, showLegend: false, showTitle: false,
    showValAxisTitle: true, valAxisTitle: "Fused minus MUSK recall (points)", valAxisTitleFontSize: 13, valAxisTitleColor: HEX.accent5, valAxisTitleFontFace: "+mn-lt",
    altText: "Per-class recall change, fused minus MUSK. Stroma and smooth muscle swing by about 13 points in opposite directions and cancel; tumour gains 3.6 points.",
  });
  const rx = 8.25, rw = 4.48;
  T(s, `${pts(FR.per_class_contribution_to_bal_acc_delta_pts.TUM)} of the ${pts(CC.delta_vs_best_single_pts)}`, rx, 1.55, rw, 0.6, { fontSize: 30, bold: true, valign: "middle" });
  T(s, "points of balanced accuracy come from tumour recall. Muscle and stroma cancel.", rx, 2.17, rw, 0.8, { fontSize: 16 });
  T(s, `${int(FR.val_correct_tiles.average)} / ${int(FR.val_correct_tiles.gated_mlp)} / ${int(FR.val_correct_tiles.concat_lr)}`, rx, 3.2, rw, 0.6, { fontSize: 30, bold: true, valign: "middle" });
  T(s, `of ${int(FR.n_val)} validation tiles correct for the average, gated MLP and concat rules. Validation is near ceiling.`, rx, 3.82, rw, 0.8, { fontSize: 16 });
  T(s, pts(100 * FR.test_delta_vs_musk_by_candidate.gated_mlp), rx, 4.85, rw, 0.6, { fontSize: 30, bold: true, valign: "middle" });
  T(s, "test points for the runner-up rule, the gated MLP, against MUSK.", rx, 5.47, rw, 0.6, { fontSize: 16 });
  source(s, "Post hoc and descriptive. Balanced accuracy is the mean of 9 class recalls, so each class contributes its recall change divided by 9. Source: results/headline.json.");
  s.addNotes("Backup for 'how robust is the fusion gain?'. Answer: not very. This is why the next step is the full-data run on Delta.");
}
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "References and backup" });
  title(s, "Backup: where the review flag falls short");
  const row = (n, c) => [n, pct1(c.review_caught_at_10pct.s3_fused_uncertainty), pct1(c.review_caught_at_10pct.s4_learned), pct1(Math.max(c.review_caught_at_10pct.A_own_confidence_own_errors, c.review_caught_at_10pct.B_own_confidence_own_errors)), pct2(c.acc_cleared_at_10pct.s4), pct2(c.acc_cleared_at_10pct.best_single_own)];
  s.addTable([
    [th("At a 10% budget"), th("Fused uncertainty"), th("Learned flag"), th("Best single reader"), th("Flag: cleared-tile accuracy"), th("Best single: cleared-tile accuracy")],
    row("Colorectal", CC), row("Camelyon17", CA),
  ], tableOpts({ x: LM, y: 1.45, w: CW, colW: [2.13, 1.9, 1.7, 1.9, 2.25, 2.25], rowH: [0.72, 0.46, 0.46] }));
  const op = (c) => c.prereg_rule_test_operating_point;
  s.addTable([
    [th("Pre-registered threshold on test"), th("Target flagged"), th("Actually flagged"), th("Errors caught")],
    ["Colorectal", "10%", pct1(op(CC).flagged_fraction), pct1(op(CC).errors_caught)],
    ["Camelyon17", "10%", pct1(op(CA).flagged_fraction), pct1(op(CA).errors_caught)],
  ], tableOpts({ x: LM, y: 3.45, w: CW, colW: [5.03, 2.3, 2.5, 2.3], rowH: 0.46 }));
  T(s, `Validation sits near ceiling (${pct2(FR.val_fused_bal_acc)}), and validation to test is a distribution shift, so a threshold calibrated on it does not transfer. Recalibrate on the target site.`, LM, 5.1, CW, 0.9, { fontSize: 18 });
  source(s, "Single readers flag their own errors by their own confidence. Source: results/cm_local/CM/*/review.json, results/headline.json.");
  s.addNotes("Backup for 'does the flag generalise?'. On Camelyon17 the pre-registered flag clears tiles no better than CONCH alone, and the threshold over-flags colorectal.");
}
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "References and backup" });
  title(s, "Backup: kernel throughput");
  s.addTable([
    [th("K1: tiles/s, both readers, 224 px"), th("Rate", { align: "right" })],
    ["PIL, 1 CPU process", { text: int(K.pil_single_process_tiles_per_s_224), options: { align: "right" } }],
    ["PIL DataLoader, 8 workers", { text: int(K.dataloader_8w_tiles_per_s_224), options: { align: "right" } }],
    ["torch GPU ops (exact)", { text: int(K.k1_torch_exact_tiles_per_s_224), options: { align: "right" } }],
    ["Triton (our baseline)", { text: int(K.k1_triton_tiles_per_s_224), options: { align: "right" } }],
    ["K1 CUDA, including host copy", { text: int(K.k1_cuda_incl_h2d_tiles_per_s_224), options: { align: "right" } }],
    [{ text: "K1 CUDA, input on GPU", options: { bold: true } }, { text: int(K.k1_cuda_tiles_per_s_224), options: { bold: true, align: "right" } }],
  ], tableOpts({ x: LM, y: 1.45, w: 5.85, colW: [4.45, 1.4], rowH: 0.44 }));
  s.addTable([
    [th("K2: images/s, severity 1.0"), th("Rate", { align: "right" })],
    ["numpy, 1 process", { text: int(K.k2_numpy_1proc_images_per_s), options: { align: "right" } }],
    ["numpy, 4 processes", { text: int(K.k2_numpy_4proc_images_per_s), options: { align: "right" } }],
    ["torch GPU ops", { text: int(K.k2_torch_gpu_images_per_s_sev1), options: { align: "right" } }],
    [{ text: "K2 CUDA", options: { bold: true } }, { text: int(K.k2_cuda_images_per_s_sev1), options: { bold: true, align: "right" } }],
  ], tableOpts({ x: 6.88, y: 1.45, w: 5.85, colW: [4.45, 1.4], rowH: 0.44 }));
  T(s, `K1 cosine to the official transforms: at least ${K.k1_vs_reference_embedding_cosine_min.conch.toFixed(6)}. The ${x2(K.k1_e2e_speedup_vs_reference)} end-to-end figure is for a batched, GPU-forward-bound run, where preprocessing is ${pct1(K.preprocess_share_of_reference_e2e)} of the time. Per-image serving and faster GPUs are a different regime, not measured here.`, LM, 4.8, CW, 1.3, { fontSize: 17 });
  source(s, "CUDA-event timing (warmup 3, median of 10) on a shared GPU, so throughput is an estimate. Source: results/kernels/*.json.");
  s.addNotes("Backup for compute questions. Kernel bench uses batch 256 with both readers' outputs.");
}
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "References and backup" });
  title(s, `Backup: a ${Math.round(D.app_frame_ms_p50_sum_of_stages)} ms app frame and int8 fidelity`);
  const st = D.stage_p50_ms, tot = (k) => +st[k].toFixed(1);
  const stages = [["Tissue check", tot("tissue check")], ["Tissue fraction for UI", tot("tissue fraction again (UI)")], ["int8 student", tot("int8 student (onnxruntime)")], ["LED result (RPC)", tot("LED result (Bridge RPC)")], ["LED busy (RPC)", tot("LED busy (Bridge RPC)")], ["Frame read", tot("frame read (PNG decode)")], ["UI frame", tot("UI frame (box + JPEG)")]];
  const cols = stages.map(([n]) => (n === "int8 student" ? HEX.accent3 : BASE));
  T(s, "Median time per stage (ms)", LM, 1.45, 6, 0.35, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  hBars(s, stages.map((r) => r[0]), stages.map((r) => r[1]), cols, { x: LM, y: 1.8, w: 6.2, h: 4.45, max: 30, series: "Median ms per stage", fmt: "0.0", gap: 35, alt: "Median milliseconds per app stage on the UNO Q. The numpy tissue check is the largest cost, not the network." });
  const f = D.cm_board_device_flag;
  s.addTable([
    [th("CONCH + MUSK student, int8"), th("Balanced accuracy", { align: "right" })],
    ["On the board", { text: pct2(KC.cm.bal_acc.board), options: { bold: true, align: "right" } }],
    ["x86, fused int8 kernels off", { text: pct2(KC.cm.bal_acc.x86_noopt), options: { align: "right" } }],
    ["x86, onnxruntime defaults", { text: pct2(KC.cm.bal_acc.x86), options: { align: "right" } }],
    ["fp32 student", { text: pct2(KC.cm.fp32_student_bal_acc), options: { align: "right" } }],
  ], tableOpts({ x: 7.1, y: 1.5, w: 5.63, colW: [3.5, 2.13], rowH: 0.44 }));
  T(s, [
    { text: `Board vs x86 kernels off: ${KC.cm.board_vs_x86_noopt.pred_mismatches} of ${int(TC.n_test)} calls differ.`, options: { breakLine: true } },
    { text: `Board flag: ${pct1(f.fraction)} of tiles, catches ${pct1(f.errors_caught)} of its own errors, agreement with the teacher flag ${pct1(f.agreement)} (κ ${f.kappa.toFixed(2)}).`, options: { breakLine: true } },
    { text: "The device’s two readers are two heads on one shared backbone." },
  ], 7.1, 3.95, 5.63, 2.3, { fontSize: 16, paraSpaceAfter: 8 });
  source(s, `Full app loop ${D.cm_full_loop_tiles_per_s.toFixed(1)} tiles/s. Each Bridge RPC call to the MCU costs about ${Math.round(D.led_bridge_rpc_ms_per_call)} ms. Source: results/uno_q/onboard/*.json.`);
  s.addNotes("Backup for edge-device questions. The largest cost in a frame is the numpy tissue check, not the network.");
}

// ---- native chart details pptxgenjs does not expose ------------------------------------------------------------------
const sc = (name) => `<a:solidFill><a:schemeClr val="${name}"/></a:solidFill>`;
const ln = (name, widthPt, dash) => `<a:ln w="${Math.round(widthPt * 12700)}" cap="rnd">${sc(name)}${dash ? `<a:prstDash val="${dash}"/>` : ""}<a:round/></a:ln>`;
const noMarker = `<c:marker><c:symbol val="none"/></c:marker>`;
const circle = (name, size) => `<c:marker><c:symbol val="circle"/><c:size val="${size}"/><c:spPr>${sc(name)}<a:ln w="19050">${sc("bg1")}</a:ln></c:spPr></c:marker>`;
const flags = (o = {}) => `<c:showLegendKey val="0"/><c:showVal val="${o.val ? 1 : 0}"/><c:showCatName val="0"/><c:showSerName val="${o.ser ? 1 : 0}"/><c:showPercent val="0"/><c:showBubbleSize val="0"/>`;
const textProps = (color, sz, bold) => `<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="${sz * 100}" b="${bold ? 1 : 0}">${sc(color)}<a:latin typeface="+mn-lt"/></a:defRPr></a:pPr><a:endParaRPr lang="en-US"/></a:p></c:txPr>`;
const serLabel = (idx, pos, color = "tx1", bold = true, dy = 0) => `<c:dLbls><c:dLbl><c:idx val="${idx}"/>${dy ? `<c:layout><c:manualLayout><c:x val="0"/><c:y val="${dy}"/></c:manualLayout></c:layout>` : ""}<c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>${textProps(color, 14, bold)}<c:dLblPos val="${pos}"/>${flags({ ser: true })}</c:dLbl><c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>${flags()}<c:extLst><c:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}" xmlns:c15="http://schemas.microsoft.com/office/drawing/2012/chart"><c15:showLeaderLines val="0"/></c:ext></c:extLst></c:dLbls>`;
const numLit = (vals) => `<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="${vals.length}"/>${vals.map((v, i) => `<c:pt idx="${i}"><c:v>${v}</c:v></c:pt>`).join("")}</c:numLit>`;
const errBars = (dir, type, plus, minus, line) => `<c:errBars><c:errDir val="${dir}"/><c:errBarType val="${type}"/><c:errValType val="cust"/><c:noEndCap val="1"/><c:plus>${numLit(plus)}</c:plus><c:minus>${numLit(minus)}</c:minus><c:spPr>${line}</c:spPr></c:errBars>`;

function rebuildScatter(xml, specs) {
  let i = -1;
  return xml.replace(/<c:ser>[\s\S]*?<\/c:ser>/g, (m) => {
    i += 1;
    const sp = specs[i];
    const grab = (tag) => m.match(new RegExp(`<c:${tag}>[\\s\\S]*?</c:${tag}>`))[0].replace(/<c:pt idx="\d+"><c:v>\s*<\/c:v><\/c:pt>/g, "");
    return `<c:ser><c:idx val="${i}"/><c:order val="${i}"/>${grab("tx")}<c:spPr>${sp.line}</c:spPr>${sp.marker}${sp.dPt || ""}${sp.dLbls || ""}${sp.err || ""}${grab("xVal")}${grab("yVal")}<c:smooth val="0"/></c:ser>`;
  });
}

async function patchCharts(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const names = Object.keys(zip.files).filter((n) => /^ppt\/charts\/chart\d+\.xml$/.test(n));
  const noLine = `<a:ln w="19050"><a:noFill/></a:ln>`;
  for (const n of names) {
    let xml = await zip.file(n).async("string");
    if (xml.includes("forest-primary")) {
      xml = rebuildScatter(xml, [
        { line: ln("accent3", 3.5), marker: noMarker },
        { line: ln("accent5", 2.5), marker: noMarker },
        { line: ln("accent5", 2.5), marker: noMarker },
        { line: ln("accent5", 2.5), marker: noMarker },
        { line: noLine, marker: circle("accent3", 14) },
        { line: noLine, marker: circle("accent5", 12) },
      ]);
      // y axis: no labels, drawn at x = 0 as the reference line
      xml = xml.replace(/(<c:valAx>(?:(?!<\/c:valAx>)[\s\S])*?<c:axPos val="l"\/>(?:(?!<\/c:valAx>)[\s\S])*?)<c:tickLblPos val="[^"]+"\/>/, '$1<c:tickLblPos val="none"/>');
    } else if (xml.includes("Fused uncertainty")) {
      const rf = TC.review_fraction.filter((v) => v <= 0.3 + 1e-9), at = (pct) => rf.findIndex((v) => Math.abs(100 * v - pct) < 1e-6);
      const i10 = at(10);
      const dPt = (color) => `<c:dPt><c:idx val="${i10}"/>${circle(color, 11)}<c:bubble3D val="0"/></c:dPt>`;
      const lastIdx = at(30);
      xml = rebuildScatter(xml, [
        { line: ln("accent4", 2.25), marker: noMarker, dPt: dPt("accent4"), dLbls: serLabel(lastIdx, "r", "tx1", false) },
        { line: ln("accent1", 2.25), marker: noMarker, dPt: dPt("accent1"), dLbls: serLabel(lastIdx, "r", "tx1", false, 0.05) },
        { line: ln("accent2", 2.25), marker: noMarker, dPt: dPt("accent2"), dLbls: serLabel(lastIdx, "r", "tx1", false) },
        { line: ln("accent3", 4), marker: noMarker, dPt: dPt("accent3"), dLbls: serLabel(lastIdx, "r", "tx1", true, -0.05) },
        { line: `<a:ln w="9525"><a:noFill/></a:ln>`, marker: noMarker, dLbls: "", err: errBars("y", "plus", xVals100(rf, i10), rf.map(() => 0), ln("accent5", 1.25, "dash")) },
      ]);
    }
    xml = xml.replace(/<c:dLbls>[\s\S]*?<\/c:dLbls>/g, (m) => m.replace(/<a:bodyPr\/>/g, '<a:bodyPr wrap="none"/>'));
    xml = xml.replace(/<c:dispBlanksAs val="[a-zA-Z]+"\/>/, '<c:dispBlanksAs val="gap"/>');
    zip.file(n, xml);
  }
  // Empty cells in the embedded workbooks: pptxgenjs writes <v></v>, which should simply be absent.
  const books = Object.keys(zip.files).filter((n) => /^ppt\/embeddings\/.*\.xlsx$/.test(n));
  for (const b of books) {
    const inner = await JSZip.loadAsync(await zip.file(b).async("nodebuffer"));
    const sheet = inner.file("xl/worksheets/sheet1.xml");
    if (!sheet) continue;
    const txt = await sheet.async("string");
    const fixed = txt.replace(/<c r="[A-Z]+\d+"(?: [a-z]+="[^"]*")*>\s*<v>\s*<\/v>\s*<\/c>/g, "");
    if (fixed !== txt) {
      inner.file("xl/worksheets/sheet1.xml", fixed);
      zip.file(b, await inner.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
    }
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}
function xVals100(rf, i10) { return rf.map((_, i) => (i === i10 ? 100 : 0)); }

// pptxgenjs writes a second <a:pPr> before a later run when a paragraph mixes run options. The schema allows one, first,
// and it is the first that carries bullets and indents. Applies to slides, layouts and notes.
async function fixParagraphProps(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const names = Object.keys(zip.files).filter((n) => /^ppt\/(slides\/slide|slideLayouts\/slideLayout|notesSlides\/notesSlide)\d+\.xml$/.test(n));
  for (const n of names) {
    const xml = await zip.file(n).async("string");
    const fixed = xml.replace(/<a:p>[\s\S]*?<\/a:p>/g, (p) => {
      let seen = false;
      return p.replace(/<a:pPr(?:\s[^>]*)?(?:\/>|>[\s\S]*?<\/a:pPr>)/g, (m) => {
        if (seen) return "";
        seen = true;
        return m;
      });
    });
    if (fixed !== xml) zip.file(n, fixed);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

// pptxgenjs writes all notes into one <a:t>, and PowerPoint does not treat a newline there as a paragraph. One paragraph per line.
async function formatNotes(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const names = Object.keys(zip.files).filter((n) => /^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(n));
  const para = (line) => {
    const bullet = line.startsWith("\u2022 "), head = /^(\[\d+:\d\d\]|How we built it:|If asked:)$/.test(line), spaced = head || /^(Sources:|Leave this)/.test(line);
    const pPr = bullet ? `<a:pPr marL="171450" indent="-171450"><a:buFont typeface="Arial"/><a:buChar char="\u2022"/></a:pPr>` : `<a:pPr marL="0" indent="0">${spaced ? `<a:spcBef><a:spcPts val="600"/></a:spcBef>` : ""}<a:buNone/></a:pPr>`;
    return `<a:p>${pPr}<a:r><a:rPr lang="en-US" dirty="0"${head ? ' b="1"' : ""}/><a:t>${bullet ? line.slice(2) : line}</a:t></a:r></a:p>`;
  };
  for (const n of names) {
    const xml = await zip.file(n).async("string");
    zip.file(n, xml.replace(/<a:p><a:r><a:rPr[^>]*\/><a:t>([^<]*\n[^<]*)<\/a:t><\/a:r>(?:<a:endParaRPr[^>]*\/>)?<\/a:p>/, (_, body) => body.split("\n").filter((l) => l.trim()).map(para).join("")));
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

async function applyTheme(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const names = Object.keys(zip.files).filter((n) => /^ppt\/theme\/theme\d*\.xml$/.test(n));
  for (const n of names) {
    let xml = await zip.file(n).async("string");
    const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
    const scheme = `<a:clrScheme name="${THEME.name}">${slots.map((k) => `<a:${k}><a:srgbClr val="${HEX[k]}"/></a:${k}>`).join("")}</a:clrScheme>`;
    xml = xml.replace(/<a:clrScheme[\s\S]*?<\/a:clrScheme>/, scheme).replace(/(<a:theme [^>]*name=")[^"]*"/, `$1${THEME.name}"`);
    zip.file(n, xml);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await patchCharts(OUT);
  await fixParagraphProps(OUT);
  await formatNotes(OUT);
  await applyTheme(OUT);
  console.log("wrote", OUT);
})();
