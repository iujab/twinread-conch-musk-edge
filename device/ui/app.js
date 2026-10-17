const $ = id => document.getElementById(id);
const pct = x => (100 * x).toFixed(0) + "%";
function bar(cls, name, label, p) {
  return `<div class="row ${cls}"><span>${name}</span><div class="track"><div class="fill" style="width:${pct(p)}"></div></div><span class="call">${label} ${pct(p)}</span></div>`;
}
async function tick() {
  try {
    const s = await (await fetch("/state", {cache: "no-store"})).json();
    $("banner").textContent = s.banner;
    const bits = [s.mode_banner, s.host_label, s.bundle_note, s.mode_reason].filter(Boolean);
    $("framelabel").textContent = s.frame_label || "Frame";
    $("modebar").textContent = bits.join("  |  ");
    const st = s.status || "idle";
    $("badge").className = "badge " + st;
    $("badge").textContent = {agree: "AGREE", review: "REVIEW", no_tissue: "NO TISSUE", busy: "READING", idle: "IDLE", error: "ERROR"}[st] || st;
    $("reasons").textContent = s.error ? s.error : (s.reasons && s.reasons.length ? "Why: " + s.reasons.join(", ") : (st === "agree" ? "Both student heads agree" : ""));
    const c = s.call;
    if (c) {
      const top = p => { const k = p.indexOf(Math.max(...p)); return [s.classes[k], p[k]]; };
      const [la, pa] = top(c.probsA), [lb, pb] = top(c.probsB), [lf, pf] = top(c.probsF);
      $("bars").innerHTML = bar("a", s.readers[0], la, pa) + bar("b", s.readers[1], lb, pb) + bar("f", "Fused head", lf, pf);
      $("stats").innerHTML =
        `<div class="stat">Tissue fraction<b>${pct(s.tissue_fraction)}</b></div>` +
        `<div class="stat">Latency (${s.host && s.host.is_uno_q ? "UNO Q" : (s.host ? s.host.machine + " host" : "host")})<b>${s.latency_ms.toFixed(0)} ms</b></div>` +
        `<div class="stat">Fused call<b>${c.label}</b></div>` +
        `<div class="stat">Review score<b>${c.review_score.toFixed(2)}</b></div>`;
      $("meta").textContent = `tile #${s.n} | LED matrix: ${s.led ? "updated" : "not connected"}`;
    }
    $("headsnote").textContent = s.heads_note || "";
    $("foot").textContent = `Thresholds: JS > ${s.thresholds.js}, review head > ${s.thresholds.review}. Pair: ${s.pair}.`;
    $("frame").src = "/frame.jpg?" + Date.now();
  } catch (e) { $("reasons").textContent = "UI cannot reach the device app: " + e; }
}
setInterval(tick, 800); tick();
