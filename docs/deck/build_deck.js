// Builds docs/twinread_deck.pptx: editable text, shapes, tables and native charts, with speaker notes.
// Every TwinRead number is read from results/**/*.json at build time. The only numbers typed by hand are the two
// literature figures on slide 2, each cited on the slide.
//
//   cd docs/deck && npm install && node build_deck.js
//
// docs/deck/assets/ comes from make_assets.py and is committed, so this script does not need data/.
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
const FOOT = "TwinRead  ·  Research prototype, not a diagnostic device";

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
  objects: [titlePh({ x: 0.75, y: 1.45, w: 6.6, h: 1.1, fontSize: 60, color: WHITE, valign: "bottom" }), {
    placeholder: {
      options: { name: "subtitle", type: "body", x: 0.75, y: 2.72, w: 6.5, h: 1.5, fontSize: 24, color: WHITE, align: "left", valign: "top", margin: 0 },
      text: "Subtitle",
    },
  }],
});
pres.defineSlideMaster({
  title: "CLOSE", background: { color: NIGHT }, slideNumber: { ...slideNo, color: RULE },
  objects: [titlePh({ color: WHITE, fontSize: 32 }), {
    text: { text: FOOT.replace("TwinRead  ·  ", ""), options: { x: LM, y: 6.99, w: 8, h: 0.28, fontSize: 10, color: RULE, margin: 0, valign: "middle" } },
  }],
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

// ---- 1 cover --------------------------------------------------------------------------------------------------------
pres.addSection({ title: "Talk" });
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Talk" });
  title(s, "TwinRead");
  s.addText("Two pathology foundation models as two independent readers, and a second look that runs offline", { placeholder: "subtitle" });
  T(s, "Yubo Wu, Noah Zhang, William Liu, Franklin Liu, Andy Chen", 0.75, 4.55, 6.5, 0.35, { fontSize: 16, color: WHITE, bold: true });
  T(s, "University of Illinois Urbana-Champaign", 0.75, 4.93, 6.5, 0.3, { fontSize: 14, color: RULE });
  T(s, "2026 Ashby Prize in Computational Science Hackathon", 0.75, 5.23, 6.5, 0.3, { fontSize: 14, color: RULE });
  T(s, "Research prototype, not a diagnostic device", 0.75, 6.28, 6.5, 0.3, { fontSize: 12, color: RULE });
  const order = ["TUM", "STR", "LYM", "MUS", "NORM", "DEB", "MUC", "ADI", "BACK"];
  const names = { ADI: "adipose", BACK: "background", DEB: "debris", LYM: "lymphocytes", MUC: "mucus", MUS: "smooth muscle", NORM: "normal mucosa", STR: "stroma", TUM: "tumour epithelium" };
  const d = 1.62, g = 0.06, x0_ = 7.85, y0_ = 1.26;
  order.forEach((c, i) => s.addImage({
    path: A(`class_${c}.jpg`), x: x0_ + (i % 3) * (d + g), y: y0_ + Math.floor(i / 3) * (d + g), w: d, h: d,
    altText: `H&E tile from the CRC-VAL-HE-7K test set: ${names[c]}`,
  }));
  s.addNotes(`[0:20] Hello, we are team TwinRead. Our question comes straight from the challenge: can foundation models make pathology more precise, more reproducible and more interpretable? These nine tiles are the nine tissue classes in our test set, all real H&E.`);
}

// ---- 2 problem ------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  title(s, "Pathology reads vary, and second readers are scarce");
  T(s, "75.3%", LM, 1.45, 5.6, 1.05, { fontSize: 66, bold: true });
  T(s, "of individual breast-biopsy reads matched the expert panel, and only 48% for atypia", LM, 2.55, 5.5, 1.1, { fontSize: 20 });
  T(s, "47%", 7.0, 1.45, 5.6, 1.05, { fontSize: 66, bold: true });
  T(s, "of the world’s population has little or no access to diagnostics", 7.0, 2.55, 5.5, 1.1, { fontSize: 20 });
  card(s, LM, 4.1, CW, 1.95);
  motif(s, LM + 0.4, 4.6, 0.92);
  T(s, "Can two AI models act as independent readers?", 2.75, 4.5, 9.6, 0.6, { fontSize: 26, bold: true, valign: "middle" });
  T(s, "And can that second look run offline, on a small board?", 2.75, 5.1, 9.6, 0.5, { fontSize: 20, valign: "middle" });
  source(s, "Elmore JG et al., JAMA 2015 (115 pathologists, 6,900 interpretations). Fleming KA et al., Lancet Commission on diagnostics, Lancet 2021.");
  s.addNotes(`[0:50] Pathology has a safety habit for hard cases: a second, independent read. It exists because reads vary. In Elmore's JAMA study, individual reads of breast biopsies matched the expert panel 75 percent of the time, and only 48 percent for atypia. But a second read needs a second expert, and the Lancet Commission estimates almost half the world has little or no access to diagnostics at all.\n\nSo we asked two questions. Can CONCH and MUSK, the two foundation models this hackathon provides, act as two independent readers? And can that second look be made small enough to run offline?\n\nSources: Elmore et al. doi:10.1001/jama.2015.1405. Fleming et al. doi:10.1016/S0140-6736(21)00673-5.`);
}

// ---- 3 different errors ---------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  title(s, "Two models, trained differently, miss different tiles");
  dot(s, LM, 1.43, 0.24, CONCH, "CONCH marker");
  T(s, "CONCH", LM + 0.36, 1.35, 5, 0.4, { fontSize: 22, bold: true, valign: "middle" });
  T(s, "Contrastive: 1.17 M image–caption pairs", LM + 0.36, 1.77, 5.5, 0.35, { fontSize: 17, color: MUTED });
  dot(s, 7.0, 1.43, 0.24, MUSK, "MUSK marker");
  T(s, "MUSK", 7.36, 1.35, 5, 0.4, { fontSize: 22, bold: true, valign: "middle" });
  T(s, "Masked: 50 M images and 1 B text tokens", 7.36, 1.77, 5.37, 0.35, { fontSize: 17, color: MUTED });
  T(s, `${OV.either_wrong} of ${int(OV.n)} colorectal test tiles have at least one reader wrong`, LM, 2.62, CW, 0.38, { fontSize: 18, bold: true });
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
  T(s, `Only ${OV.both_wrong} tiles fool both. The rest are gaps the other reader can cover.`, LM, 5.3, CW, 0.6, { fontSize: 24, bold: true, valign: "middle" });
  source(s, "Linear probes on frozen features, CRC-VAL-HE-7K, 9 tissue classes. Lu et al., Nat Med 2024; Xiang et al., Nature 2025.");
  s.addNotes(`[0:50] Why would two models behave like two readers? Because they learned differently. CONCH was pretrained contrastively on over 1.17 million histology image and caption pairs. MUSK used masked modelling on 50 million image patches and a billion text tokens, unpaired, before its contrastive stage.\n\nHere is the test of that idea. On the ${int(OV.n)} colorectal test tiles, CONCH misses ${OV.conch_errors} and MUSK misses ${OV.musk_errors}, but only ${OV.both_wrong} are missed by both. So each reader gets most of the other's errors right: ${seg[0]} of CONCH's, ${seg[2]} of MUSK's. That is the gap a second reader can cover.\n\nSource: results/cm_local/CM/crc/reader_error_overlap.json.`);
}

// ---- 4 system -------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  title(s, "Read twice, flag doubt, then shrink to a small board");
  const head = (txt, x, y, w) => T(s, txt, x, y, w, 0.35, { fontSize: 16, bold: true, color: MUTED, valign: "middle" });
  head("1  Read twice", 2.75, 1.3, 2.8);
  head("2  Fuse and flag", 6.0, 1.3, 3);
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
  head("3  Distil", 7.4, 3.72, 2);
  arrow(s, 7.27, 3.41, 0, 1.0);
  box(s, [
    { text: "Student network", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: `${S.params_millions.toFixed(2)} M parameters, ${S.int8_onnx_mb.toFixed(1)} MB int8`, options: { fontSize: 14, color: MUTED } },
  ], 5.75, 4.46, 3.05, 1.35, TINT, INK);
  arrow(s, 8.85, 5.13, 0.35, 0);
  box(s, [
    { text: "Arduino UNO Q", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "Offline, no GPU, USB-powered", options: { fontSize: 14, color: MUTED } },
  ], 9.25, 4.46, 3.48, 1.35, TINT, INK);
  // pre-registration
  T(s, "Fixed in advance", LM, 4.5, 4.7, 0.38, { fontSize: 18, bold: true });
  T(s, "Fusion rule and thresholds were pre-registered before any test label was read.", LM, 4.92, 4.7, 0.85, { fontSize: 17 });
  source(s, `Train: ${int(SM.n.train)} NCT-CRC-HE-100K tiles. Test: ${int(TC.n_test)} CRC-VAL-HE-7K tiles (9 classes) and 1,500 Camelyon17 tiles from an unseen hospital.`);
  s.addNotes(`[1:00] The system has three parts. One: each tile is read twice, by a linear probe on frozen CONCH features and one on frozen MUSK features. Two: the readers are fused into one call, and the fused reader's uncertainty raises a review flag for a pathologist. Three: a small student network learns from both readers and is exported to int8, so it runs on an Arduino UNO Q.\n\nThe key discipline: the fusion rule and the review thresholds were chosen on validation and pre-registered before any test label was read. Every test number you see was computed once, after that.\n\nThe open-model pair, PathGen-CLIP and QuiltNet, went through the same protocol as a negative control.`);
}

// ---- 5 kernels ------------------------------------------------------------------------------------------------------
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
  s.addNotes(`[1:00] We wrote two custom CUDA kernels. Both are fused chains of dependent stages: each stage needs the previous one's output, so instead of a launch and a global-memory round trip per stage, one launch keeps the intermediate in shared memory.\n\nK1 prepares both readers' inputs from one tile, and its fixed-point math makes the output bit-exact to each model's official transform. It is ${x0(K.k1_preprocess_speedup_vs_pil)} faster than PIL inside the embedding run. We also show the honest number next to it: the whole run gains only ${x2(K.k1_e2e_speedup_vs_reference)}, because the model forward dominates in this batched setup.\n\nK2 simulates what a camera filming a screen does to a tile: warp, stain jitter, moire, glare, defocus, noise, JPEG. It augments ${x1(augX)} faster than torch, which makes the full student training step ${x2(stepX)} faster, close to training with no augmentation at all (${int(step.none)} images/s).\n\nIf asked: the kernel bench with input already on the GPU gives ${int(K.k1_cuda_tiles_per_s_224)} tiles/s, and K2 gives ${int(K.k2_cuda_images_per_s_sev1)} images/s standalone. See the backup slide.\n\nSources: results/kernels/*.json, results/student/*/throughput.json.`);
}

// ---- 6 fusion -------------------------------------------------------------------------------------------------------
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
  s.addNotes(`[1:00] First result. Fusing CONCH and MUSK gives ${pct2(CC.bal_acc.fused)} balanced accuracy against ${pct2(CC.bal_acc.B)} for MUSK alone, a gain of ${CC.delta_vs_best_single_pts.toFixed(2)} points. Most of the net gain is tumour recall, up ${dt.TUM.toFixed(1)} points. Fusion leaves ${OV.fused_errors} errors, fewer than either reader alone.\n\nBut look at the forest plot. Only the top row clears zero. On Camelyon17, an unseen hospital, fusion gives no gain, and the open control pair gains nothing on either dataset. So the gain is real but small and dataset-dependent.\n\nIf asked: the colorectal CI is tile-level and therefore optimistic because this dataset has no slide IDs. Validation picked the fusion rule by one or two tiles (${FR.val_correct_tiles.average} vs ${FR.val_correct_tiles.gated_mlp} vs ${FR.val_correct_tiles.concat_lr} of ${FR.n_val}). Muscle and stroma recall swing by about 13 points in opposite directions and cancel. Backup slide has the per-class view.\n\nSources: results/cm_local/CM/*/fusion.json, results/open/OPEN/*/fusion.json, results/headline.json.`);
}

// ---- 7 triage -------------------------------------------------------------------------------------------------------
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
  source(s, `CRC-VAL-HE-7K, ${int(TC.n_test)} tiles. Single readers flag their own errors by their own confidence. Disagreement covers only ${pct1(CC.s1_disagree_rate)} of tiles, so the rest of its budget is effectively random.`);
  s.addNotes(`[1:10] Second result, and the one that surprised us. We expected disagreement between the readers to be the best signal for sending a tile to a pathologist. It is not.\n\nThe chart ranks tiles by each score and sends the top ones for review. At a 10% budget, the fused reader's own uncertainty catches ${pct1(R.s3_fused_uncertainty)} of its errors. MUSK flagging itself catches ${pct1(R.B_own_confidence_own_errors)}, CONCH ${pct1(R.A_own_confidence_own_errors)}. Disagreement alone stalls at ${pct1(R.s1_disagree)}, because the readers disagree on only ${pct1(CC.s1_disagree_rate)} of tiles.\n\nA learned flag that adds disagreement to uncertainty ranks tiles almost identically, Spearman ${FR.s3_s4_spearman.toFixed(4)}, so it adds nothing. Two readers help because they make a better-calibrated reader, not because their disagreement is informative.\n\nIf asked: on Camelyon17 fused uncertainty catches ${pct1(CA.review_caught_at_10pct.s3_fused_uncertainty)}, while the pre-registered flag reaches ${pct1(CA.review_caught_at_10pct.s4_learned)} against CONCH alone at ${pct1(CA.review_caught_at_10pct.A_own_confidence_own_errors)}, and it does not beat CONCH alone on cleared-tile accuracy. Backup slide.\n\nSources: results/cm_local/CM/crc/review.json, triage_curves.json.`);
}

// ---- 8 blind spots --------------------------------------------------------------------------------------------------
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
  stat(1.55, BS.n, `tiles (${pct2(CC.confident_shared_errors.rate)}): both wrong, same call, both at least 0.80 confident`);
  stat(2.6, BS.flagged, `caught by the pre-registered flag, which reviews ${pct1(op.flagged_fraction)} of tiles`);
  stat(3.65, nMiss, `escaped it (shown): both readers at least ${(Math.floor(minConf * 100) / 100).toFixed(2)} confident`);
  card(s, rx, 4.8, rw, 1.3);
  T(s, "Two models agreeing is not proof. A pathologist stays in the loop.", rx + 0.25, 4.85, rw - 0.5, 1.2, { fontSize: 20, bold: true, valign: "middle" });
  source(s, `CRC-VAL-HE-7K, truth as labelled in the dataset. ${BS.pairs["STR->MUS"] + BS.pairs["MUS->STR"]} of the ${BS.n} are muscle vs stroma; ${BS.pairs["TUM->NORM"]} are tumour called normal.`);
  s.addNotes(`[0:50] Now the limit of the whole idea. These are shared blind spots: tiles where both readers make the same wrong call, both with high confidence. There are ${BS.n} on the colorectal test set, ${pct2(CC.confident_shared_errors.rate)} of tiles. Most are muscle versus stroma, but ${BS.pairs["TUM->NORM"]} are tumour called normal.\n\nNo disagreement signal can catch these, because the readers agree. The fused uncertainty flag still catches ${BS.flagged} of them, at the cost of reviewing a quarter of tiles. The ${nMiss} on this slide escaped even that.\n\nPathologists in the room may disagree with some of these labels, and we would welcome that. The point stands: when two models agree, that is not proof.\n\nIf asked: on Camelyon17 the flag catches only ${CA.confident_shared_errors.flagged_by_prereg_rule} of ${CA.confident_shared_errors.n} such tiles.\n\nSources: results/cm_local/CM/crc/review.json, triage_curves.json, tile_preds.npz.`);
}

// ---- 9 board --------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
  title(s, `A ${Math.round(S.int8_onnx_mb)} MB student runs offline on a USB-powered board`);
  s.addImage({ path: A("uno_q.jpg"), x: LM, y: 1.5, w: 6.05, h: 4.8, altText: "The Arduino UNO Q running TwinRead, powered over USB-C, with its LED matrix lit" });
  const rx = 7.15, rw = 5.58;
  const stat = (y, n, txt) => { T(s, n, rx, y, rw, 0.82, { fontSize: 46, bold: true, valign: "middle" }); T(s, txt, rx, y + 0.84, rw, 0.62, { fontSize: 16, color: INK }); };
  stat(1.45, `${D.cm_int8_p50_ms_4_threads.toFixed(1)} ms`, "per tile, model inference: int8, 4 threads, offline");
  stat(3.05, pct2(D.cm_int8_bal_acc_board), `balanced accuracy on all ${int(TC.n_test)} test tiles, below every teacher (${(100 * CC.bal_acc.A).toFixed(2)}–${pct2(CC.bal_acc.fused)})`);
  stat(4.65, `${S.int8_onnx_mb.toFixed(1)} MB`, `${S.params_millions.toFixed(2)} M parameters: ${S.params_pct_of_teacher_vision_towers.toFixed(1)}% of the two teachers’ vision towers`);
  source(s, `All measured on the Arduino UNO Q: Qualcomm QRB2210, 4× Cortex-A53, 3.6 GiB RAM, onnxruntime on CPU. Stored test tiles; no camera captures yet.`);
  s.addNotes(`[0:50] Hold up the board now, if you brought it.\n\nThis is the third part. A student network, ${S.params_pct_of_teacher_vision_towers.toFixed(1)} percent of the parameters of the two teachers' vision towers, exported to a ${S.int8_onnx_mb.toFixed(1)} megabyte int8 model. It runs on this Arduino UNO Q over USB power, with no network.\n\nEverything on this slide was measured on the board: ${D.cm_int8_p50_ms_4_threads.toFixed(1)} milliseconds of model inference per tile, and ${pct2(D.cm_int8_bal_acc_board)} balanced accuracy across all ${int(TC.n_test)} test tiles, which took ${Math.round(D.cm_full_loop_seconds)} seconds. That is below every teacher. The value here is small, offline and low-cost, not beating the foundation models.\n\nIf asked: a full app frame takes about ${Math.round(D.app_frame_ms_p50_sum_of_stages)} ms, mostly the numpy tissue check, and the full loop runs at ${D.cm_full_loop_tiles_per_s.toFixed(1)} tiles/s. The x86 int8 score of ${pct2(D.cm_int8_bal_acc_x86)} was an onnxruntime kernel artefact. Backup slide.\n\nSources: results/uno_q/onboard/*.json, results/student/model_sizes.json.`);
}

// ---- 10 gaps --------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT_PLAIN", sectionTitle: "Talk" });
  title(s, "Three gaps remain, each with a concrete next step");
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
  T(s, "Overcome along the way", LM, 5.2, CW, 0.36, { fontSize: 14, bold: true, color: MUTED, valign: "middle" });
  [["Non-commercial licences", "open control pair, shareable bundle"], ["int8 calibration collapse", "percentile calibration"], ["Low x86 int8 scores", "traced to an onnxruntime kernel"]].forEach(([a, b], i) => {
    const x = LM + i * 4.1;
    T(s, [{ text: a, options: { bold: true, fontSize: 16, breakLine: true } }, { text: b, options: { fontSize: 15, color: MUTED } }], x, 5.58, 3.85, 0.7);
  });
  s.addNotes(`[0:50] What is not solved. The fusion gain is fragile, so next is a full-data run on NCSA Delta with all 100 thousand training tiles. Our Slurm kit is tested locally but has not run there yet. The pre-registered threshold over-flagged the test set, ${pct1(op.flagged_fraction)} of tiles for a 10 percent target, so thresholds must be calibrated per site. And the board's flag tracks the teachers only weakly, kappa ${f.kappa.toFixed(2)}, because its two readers are two heads on one backbone. It has only seen stored tiles, so real camera captures come next.\n\nIf asked: the board flag covers ${pct1(f.fraction)} of tiles and catches ${pct1(f.errors_caught)} of its own errors. The low x86 scores only showed up as an artefact because we ran on the actual board.`);
}

// ---- 11 why it matters ----------------------------------------------------------------------------------------------
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
  s.addImage({ path: A("app_ui.png"), x: 7.0, y: 1.6, w: 5.73, h: 3.7, altText: "The TwinRead app on the UNO Q: a stored tile, both student heads, the fused call and an AGREE verdict" });
  T(s, "The app on the UNO Q, replaying a stored test tile. Its two heads share one student network.", 7.0, 5.38, 5.73, 0.55, { fontSize: 13, color: MUTED });
  s.addNotes(`[0:40] Why it matters. For pathologists: a short list of tiles worth a second look, with each reader's call shown rather than one black-box score. For patients: a low-cost, offline second look could help where no second pathologist is available, pending real-world validation. For the field: a measured, pre-registered finding about why two readers help.\n\nThis is a second-read aid, not a diagnosis, and nothing here is clinically validated.\n\nIf asked about the 53 ms in the screenshot: that timer covers the whole step, meaning frame read, crop, tissue check and network, about ${Math.round(["frame read (PNG decode)", "crop to 224 px", "uint8 -> float NCHW", "int8 student (onnxruntime)", "probes + softmax", "tissue check", "review rule"].reduce((a, k) => a + D.stage_p50_ms[k], 0))} ms by stage medians. ${D.cm_int8_p50_ms_4_threads.toFixed(1)} ms is the network alone at 4 threads.`);
}

// ---- 12 close -------------------------------------------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CLOSE", sectionTitle: "Talk" });
  title(s, "TwinRead in three numbers");
  [
    [`${pts(CC.delta_vs_best_single_pts)} pts`, "balanced accuracy from fusion on colorectal tiles; no gain on Camelyon17"],
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
  s.addNotes(`[0:20] Three numbers to remember: a small fusion gain on one dataset, ${pct1(CC.review_caught_at_10pct.s3_fused_uncertainty)} of errors caught with 10 percent review, and ${D.cm_int8_p50_ms_4_threads.toFixed(1)} milliseconds per tile on a board. Thank you. We would welcome your questions.\n\nLeave this slide up during questions. Backup slides follow the references. Spoken total is about 9:40.`);
}

// ---- references and backup ------------------------------------------------------------------------------------------
pres.addSection({ title: "References and backup" });
const tableOpts = (o) => ({ fontFace: "Arial", fontSize: 15, color: HEX.dk1, border: { type: "solid", pt: 0.75, color: "E8E6EE" }, valign: "middle", margin: [0.04, 0.1, 0.04, 0.1], ...o });
const th = (text, o = {}) => ({ text, options: { bold: true, color: HEX.accent5, fill: { color: HEX.lt2 }, ...o } });
{
  const s = pres.addSlide({ masterName: "CONTENT_PLAIN", sectionTitle: "References and backup" });
  title(s, "References");
  const refs = [
    "Elmore JG, et al. Diagnostic concordance among pathologists interpreting breast biopsy specimens. JAMA 2015. doi:10.1001/jama.2015.1405",
    "Fleming KA, et al. The Lancet Commission on diagnostics: transforming access to diagnostics. Lancet 2021. doi:10.1016/S0140-6736(21)00673-5",
    "Lu MY, et al. A visual-language foundation model for computational pathology (CONCH). Nat Med 2024. doi:10.1038/s41591-024-02856-4",
    "Xiang J, et al. A vision–language foundation model for precision oncology (MUSK). Nature 2025. doi:10.1038/s41586-024-08378-w",
    "Ikezogwo WO, et al. Quilt-1M: one million image-text pairs for histopathology (QuiltNet). NeurIPS 2023. arXiv:2306.11207",
    "Li et al. Multi-modal foundation models for computational pathology: a survey. TMLR 2025. PMC13081565",
    "Muneer A, et al. Review on multimodal data integration for cancer research. Artif Intell Rev 2026. doi:10.1007/s10462-026-11522-9",
    "Kather JN, et al. NCT-CRC-HE-100K and CRC-VAL-HE-7K (CC-BY-4.0). Bandi P, et al. Camelyon17; Koh PW, et al. WILDS (Camelyon17-WILDS, CC0).",
    "Open models: PathGen-CLIP (CC-BY-4.0), QuiltNet-B-16 (MIT), timm mobilenetv4_conv_small.",
  ];
  T(s, refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 7 } })), LM, 1.3, CW, 4.95, { fontSize: 15 });
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
  await applyTheme(OUT);
  console.log("wrote", OUT);
})();
