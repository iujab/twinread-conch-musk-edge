// Everything lives in this function: a top-level `const top` or `const name` would collide with browser globals and kill the
// whole script, which no unit test can see.
(() => {
  "use strict";

  const $ = id => document.getElementById(id);
  // The board's own screen has no pointer, so it shows no controls. Everyone viewing over the tailnet keeps both.
  const KIOSK = location.hostname === "127.0.0.1";
  if (KIOSK) document.body.classList.add("kiosk");

  // The same plain names the explanation uses: device/explain_prompt.py CLASS_NAMES.
  const NAMES = {ADI: "adipose tissue", BACK: "background", DEB: "debris", LYM: "lymphocytes", MUC: "mucus", MUS: "smooth muscle",
                 NORM: "normal colon mucosa", STR: "cancer-associated stroma", TUM: "colorectal adenocarcinoma epithelium"};
  const WHY = {disagree: "the heads chose different tissue", js: "the heads' confidence differs",
               review_head: "the review head is unsure", no_tissue: "too little tissue in the yellow square"};
  // What the page calls each status. /state, the LED and the prompt keep their own codes.
  const STATUS_WORD = {agree: "Confident", review: "Needs review", no_tissue: "No tissue"};
  const LED_NAMES = {idle: "Idle", busy: "Busy", error: "Error", ...STATUS_WORD};
  const plain = code => NAMES[code] || code;
  const named = code => `${plain(code)} (${code})`;
  const pct = x => Math.round(100 * x) + "%";
  const setText = (el, text) => { if (el.textContent !== text) el.textContent = text; };

  // -- the three bars -------------------------------------------------------------------------------------------------
  function setBar(id, label, p) {
    $(`fill-${id}`).style.width = pct(p);
    setText($(`call-${id}`), `${label} ${pct(p)}`);
  }

  function setBars(call) {
    $("bars").hidden = !call;
    if (!call) return;
    setBar("a", call.labelA, Math.max(...call.probsA));
    setBar("b", call.labelB, Math.max(...call.probsB));
    setBar("f", call.label, call.prob);
  }

  // -- the live picture -----------------------------------------------------------------------------------------------
  // The stream survives an app restart only if it is reopened, so any failure ends it and the next good state reopens it.
  const frame = $("frame");
  let streaming = false;
  function openStream() { streaming = true; frame.src = "/stream.mjpg?" + Date.now(); }
  function closeStream() { streaming = false; frame.removeAttribute("src"); }
  frame.onerror = () => { streaming = false; };

  function setPicture(s) {
    if (!streaming && s.has_picture) openStream();
    document.body.dataset.picture = s.has_picture ? "on" : "off";
    const c = s.controls, st = s.status;
    let note = "";
    if (!s.has_picture) {
      note = st === "camera_starting" ? "Opening the camera" : st === "no_camera" ? "No camera" :
             c.source === "demo" ? "No slide" : "Camera off";
    }
    setText($("fieldnote"), note);
    frame.alt = c.source === "demo" ? "Demo slide. The model reads the whole slide." :
                "Camera view with a yellow square marking the region the model reads.";
    let label = !s.has_picture ? "Frame" : c.source === "demo" && s.demo ? `${s.frame_label} ${s.demo.index + 1} of ${s.demo.total}` : s.frame_label;
    if (s.demo && c.source === "demo") label += s.demo.held_for_explain ? ", waiting for the answer" : c.feed_paused ? ", held" : "";
    setText($("framelabel"), label);
  }

  // -- the result card ------------------------------------------------------------------------------------------------
  // [the badge, the line under it, the sentence under that]. Failure copy says what is wrong and what to do.
  function verdictCopy(s) {
    const c = s.controls, st = s.status, demo = c.source === "demo", call = s.call;
    const where = KIOSK ? ` Start them from the dashboard at ${s.host.hostname}:8080.` : "";
    // A disagreement already implies that the confidence differs.
    const reasons = (s.reasons || []).filter(r => r !== "js" || !s.reasons.includes("disagree"));
    const sentence = text => text.charAt(0).toUpperCase() + text.slice(1);
    // The suggestion is the fused call, and it can differ from what both heads name on their bars. Say so, or the page seems to
    // contradict itself.
    const fusedDiffers = call && call.labelA === call.labelB && call.labelA !== call.label;
    const fusedNote = fusedDiffers ? ` Both heads chose ${named(call.labelA)}; the fused call differs.` : "";
    const last = !s.last ? "" : s.last.status === "no_tissue" ? " Last read: no tissue." : ` Last call: ${named(s.last.label)}, ${pct(s.last.prob)}.`;
    switch (st) {
      case "agree":
        return ["CONFIDENT", `Suggests ${named(call.label)}`, fusedDiffers ? `Nothing flagged it.${fusedNote}` :
                "Both heads chose the same tissue and nothing flagged it."];
      case "review":
        return ["NEEDS REVIEW", `Suggests ${named(call.label)}`,
                "Send for a second look: " + reasons.map(r => WHY[r] || r).join(", ") + "." + fusedNote];
      case "no_tissue":
        return ["NO TISSUE", "No tissue class suggested", demo ? "Too little tissue in the yellow square on this slide." :
                "Too little tissue in the yellow square. Fill the square with the tile."];
      case "standby":
        return ["STANDBY", demo ? "Slides and model are paused" : "Camera and model are off",
                (s.has_picture ? "The picture stays on screen, and nothing is running." : "Nothing is running.") + last + where];
      case "model_paused":
        return ["MODEL OFF", "Not reading", "The model is unloaded, so this picture has no call." + last];
      case "model_loading":
        return ["LOADING", "Loading the model", "The first call appears in a few seconds."];
      case "model_error":
        return ["MODEL FAILED", "The model did not load", sentence(s.model_error.replace("The model did not load: ", ""))];
      case "reading":
        return ["READING", "Reading this picture", ""];
      case "camera_starting":
        return ["OPENING", "Opening the camera", "The picture appears in a moment."];
      case "camera_off":
        return ["CAMERA OFF", "Nothing to read", "The model is on, but it has no picture." + (KIOSK ? where : " Start the camera or run the demo slides.")];
      case "no_camera":
        return ["NO CAMERA", "Camera not found",
                `${sentence(s.camera.reason || "no frames are arriving")}. Plug a USB camera into the dock and it starts by itself` +
                (KIOSK ? "." : ", or press Pause camera to stop looking.")];
      case "error":
        return ["ERROR", "This picture could not be read", `${s.error || "Unknown error"}. The next picture is read as usual.`];
      default:
        return ["STARTING", "", ""];
    }
  }

  const BADGE_CLASS = {agree: "agree", review: "review", no_tissue: "no_tissue", error: "error", model_error: "error"};

  function setTruth(s) {
    const el = $("truth"), demo = s.demo && s.controls.source === "demo";
    el.hidden = !demo;
    if (!demo) return;
    const decided = s.truth_match === true || s.truth_match === false;
    setText($("truth-label"), `True label: ${named(s.demo.truth)}.`);
    setText($("truth-match"), !decided ? "" : s.truth_match ? "Matches the call." : `The call was ${named(s.call.label)}.`);
    el.dataset.match = !decided ? "none" : s.truth_match ? "yes" : "no";
  }

  function setVerdict(s) {
    const st = s.status, c = s.controls, [badge, line, why] = verdictCopy(s);
    document.body.dataset.status = st;
    $("badge").className = `badge ${BADGE_CLASS[st] || "idle"}`;
    setText($("badge"), badge);
    setText($("call"), line);
    setText($("reason"), why);
    setTruth(s);

    const read = s.call && (st === "agree" || st === "review");
    const measured = read || (s.call && st === "no_tissue");
    setBars(read ? s.call : null);
    $("readout").hidden = !measured;
    if (measured) {
      setText($("st-tissue"), pct(s.tissue_fraction));
      setText($("st-time-label"), `Latency (${s.host.is_uno_q ? "UNO Q" : s.host.machine + " host"})`);
      setText($("st-time"), s.latency_ms.toFixed(0) + " ms");
      $("tile-call").hidden = !read;
      $("tile-score").hidden = !read;
      setText($("st-call"), s.call.label);
      setText($("st-score"), s.call.review_score.toFixed(2));
    }
    const led = `LED matrix: ${s.led ? (LED_NAMES[s.led_state] || s.led_state) : "not connected"}`;
    setText($("meta"), measured ? `tile #${s.n} | ${led}` : led);
    const thr = s.thresholds;
    setText($("foot"), (thr.js === undefined ? "" : `Thresholds: JS > ${thr.js.toFixed(3)}, review head > ${thr.review.toFixed(3)}. `) + `Pair: ${s.pair}.`);

    const offNow = st === "standby" || st === "camera_off" || st === "model_paused" || st === "model_error";
    $("starts").hidden = !offNow || KIOSK;
    if (!$("start-reading").dataset.busy) setText($("start-reading"), "Start reading");
    $("start-demo").hidden = c.source === "demo" || !c.feed_paused;
  }

  // -- the controls ---------------------------------------------------------------------------------------------------
  const feedLabel = c => c.source === "demo" ? (c.feed_paused ? "Resume slides" : "Pause slides") :
    !c.feed_paused ? "Pause camera" : c.camera_started ? "Resume camera" : "Start camera";
  const modelLabel = c => !c.model_paused ? "Pause model" : c.model_started ? "Resume model" : "Start model";
  const pressed = (el, on) => el.setAttribute("aria-pressed", on ? "true" : "false");

  function setControls(s) {
    const c = s.controls, demo = c.source === "demo";
    pressed($("src-camera"), c.source === "camera");
    pressed($("src-demo"), demo);
    pressed($("feed"), c.feed_paused);
    pressed($("model"), c.model_paused);
    if (!$("feed").dataset.busy) setText($("feed"), feedLabel(c));
    if (!$("model").dataset.busy) setText($("model"), modelLabel(c));
  }

  const CONTROL_TIMEOUT_MS = 20000;  // a stalled command must not leave every button dead
  let busy = false;
  function say(text) {
    $("notice").hidden = !text;
    setText($("notice"), text);
  }

  async function send(cmd, busyEl, busyText) {
    const urgent = cmd.explain_stop === true;  // Stop must work while any other command is on its way, such as a model loading
    if (busy && !urgent) return;
    if (!urgent) busy = true;
    if (busyEl) { busyEl.dataset.busy = "yes"; setText(busyEl, busyText); }
    try {
      const r = await fetch("/control", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(cmd),
                                         signal: AbortSignal.timeout(CONTROL_TIMEOUT_MS)});
      const j = await r.json();
      if (busyEl) delete busyEl.dataset.busy;
      if (r.ok) { say(""); render(j); } else say(j.error || "The board refused that command.");
    } catch (e) {
      if (busyEl) delete busyEl.dataset.busy;
      say("The board did not answer. Check that it is on and on the tailnet, then try again.");
    }
    if (!urgent) busy = false;
    schedule(150);  // the read a command started takes a moment; show it without waiting for the slow poll
  }

  let last = null;
  const toggleFeed = () => send({feed_paused: !last.controls.feed_paused});
  const toggleModel = () => {
    const off = last.controls.model_paused;
    send({model_paused: !off}, $("model"), off ? "Loading model" : "Unloading model");
  };
  const on = (id, fn) => $(id).addEventListener("click", () => { if (last) fn(); });
  on("src-camera", () => { if (last.controls.source !== "camera") send({source: "camera"}); });
  on("src-demo", () => { if (last.controls.source !== "demo") send({source: "demo"}); });
  on("feed", toggleFeed);
  on("model", toggleModel);
  on("start-reading", () => send({feed_paused: false, model_paused: false}, $("start-reading"), "Starting"));
  on("start-demo", () => send({source: "demo", feed_paused: false}));
  on("prev", () => send({demo_step: -1}));
  on("next", () => send({demo_step: 1}));
  $("pace").addEventListener("change", e => send({demo_interval: Number(e.target.value)}));

  // -- the demo tray --------------------------------------------------------------------------------------------------
  let trayCount = 0, trayCurrent = -1;
  function buildTray(d) {
    const box = $("thumbs");
    box.textContent = "";
    d.truths.forEach((truth, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "thumb";
      b.setAttribute("aria-label", `Slide ${i + 1}, true label ${plain(truth)}`);
      const img = new Image();
      img.alt = "";
      img.src = `/slide/${i}.jpg`;
      b.append(img);
      b.addEventListener("click", () => send({demo_index: i}));
      box.append(b);
    });
    trayCount = d.total;
    trayCurrent = -1;
  }

  function setTray(s) {
    const demo = s.controls.source === "demo" && s.demo;
    $("tray").hidden = !demo;
    document.body.dataset.source = s.controls.source;
    if (!demo) return;
    if (trayCount !== s.demo.total) buildTray(s.demo);
    if (trayCurrent !== s.demo.index) {
      const kids = $("thumbs").children;
      if (kids[trayCurrent]) kids[trayCurrent].removeAttribute("aria-current");
      kids[s.demo.index].setAttribute("aria-current", "true");
      kids[s.demo.index].scrollIntoView({block: "nearest", inline: "center"});
      trayCurrent = s.demo.index;
    }
    const pace = $("pace"), v = String(s.demo.interval);
    if (pace.value !== v) {
      if (![...pace.options].some(o => o.value === v)) pace.add(new Option(`${v} seconds`, v));
      pace.value = v;
    }
  }

  // -- the board vitals -----------------------------------------------------------------------------------------------
  function uptime(sec) {
    const m = Math.floor(sec / 60), h = Math.floor(m / 60), d = Math.floor(h / 24);
    return d ? `${d} d ${h % 24} h` : h ? `${h} h ${m % 60} min` : `${m} min`;
  }

  function setVitals(s) {
    const v = s.system || {}, c = s.controls, cam = s.camera;
    setText($("v-host"), s.host.hostname || s.host.machine);
    setText($("v-up"), v.uptime_s === undefined ? "Unknown" : uptime(v.uptime_s));
    setText($("v-temp"), v.temp_c === undefined ? "Unknown" : `${v.temp_c.toFixed(1)} °C`);
    $("v-temp").dataset.heat = v.temp_c >= 70 ? "hot" : v.temp_c >= 60 ? "warm" : "ok";
    setText($("v-load"), v.load === undefined ? "Unknown" : v.load.toFixed(2));
    setText($("v-mem"), v.mem_available_mb === undefined ? "Unknown" : `${v.mem_available_mb} MB`);
    setText($("v-fps"), !c.camera_open ? "Off" : cam.fps ? cam.fps.toFixed(0) : "Measuring");
    setText($("v-reads"), String(s.reads));
    setText($("v-model"), c.model_loaded ? "Loaded" : "Unloaded");
    setText($("v-camera"), c.camera_open ? "Open" : "Closed");
    setText($("v-explain"), explainerVital(s.explain));
    const host = s.host_label || `host: ${s.host.hostname || s.host.machine}`;
    const source = c.source === "demo" ? "Demo slides" : s.frame_label;
    setText($("modetext"), [source, host, s.source_note, s.mode_reason].filter(Boolean).join(" | "));
  }

  // -- the explainer --------------------------------------------------------------------------------------------------
  // One text at a time, always labelled by where it came from. The server decides what is shown; every viewer sees the same job.
  const AUDIENCE = {doctor: "For a doctor", user: "In plain words"};
  const EX_PROGRESS = {starting: "Starting the language model", preparing: "Preparing the language model",
                       thinking: "Giving the language model the facts", writing: "Writing", checking: "Checking the answer"};
  const EX_SERVER = {off: "Off", starting: "Starting", preparing: "Preparing", stopping: "Stopping"};
  const exActive = job => !!job && job.phase in EX_PROGRESS;
  const ago = sec => sec < 60 ? `${sec} s ago` : `${Math.floor(sec / 60)} min ago`;
  let audience = "doctor", exPicture = null, exRefocus = false;

  function exSource(job) {
    if (job.source !== "model") return "from the reading";
    return `${AUDIENCE[job.audience].toLowerCase()}, ${job.checked ? "written by the on-board language model; numbers and class checked against the reading" :
            "being written by the on-board language model, not checked yet"}`;
  }

  function setExplain(s) {
    const x = s.explain, job = x.job, active = exActive(job), go = $("ex-go");
    const stop = $("ex-stop"), close = $("ex-close");
    const stopHadFocus = document.activeElement === stop, goHadFocus = document.activeElement === go;
    document.body.dataset.explain = job ? "open" : "closed";
    pressed($("ex-doctor"), audience === "doctor");
    pressed($("ex-plain"), audience === "user");
    go.setAttribute("aria-disabled", x.available ? "false" : "true");
    setText(go, x.target === "last" ? "Explain last call" : "Explain this result");
    const why = active ? "" : x.available ? (x.hint || "") : x.why_not;
    setText($("ex-why"), why || "");
    $("ex-why").hidden = !why;
    $("ex-audience").hidden = active;
    go.hidden = active;
    stop.hidden = !active;
    close.hidden = !job || active;
    setText($("ex-progress"), active ? EX_PROGRESS[job.phase] : "");
    setText($("ex-count"), !active ? "" : job.phase === "writing" ? `${job.words} words` : `${job.elapsed_s} s`);
    if (goHadFocus && active) stop.focus();
    if (stopHadFocus && !active) close.focus();

    const panel = $("ex-panel");
    panel.hidden = !job;
    if (!job) {
      exPicture = null;
      if (exRefocus) { exRefocus = false; go.focus(); }
      return;
    }
    panel.dataset.status = job.about.status;
    setText($("ex-status"), `${STATUS_WORD[job.about.status] || job.about.status}.`);
    setText($("ex-class"), job.about.status === "no_tissue" ? "No tissue class suggested." : `Suggests ${named(job.about.label)}, ${pct(job.about.prob)}.`);
    setText($("ex-where"), `${job.about.where}, ${ago(job.about.age_s)}`);
    if (exPicture !== job.id) {
      exPicture = job.id;
      $("ex-pic").src = `/explain/picture.jpg?job=${job.id}`;
      $("ex-pic").alt = `The tile this explanation is about: ${job.about.where}.`;
    }
    $("ex-moved").hidden = job.current;
    setText($("ex-source"), exSource(job));
    $("ex-text").setAttribute("aria-busy", active ? "true" : "false");
    setText($("ex-text"), job.text);
    $("ex-note").hidden = !job.note;
    setText($("ex-note"), job.note || "");
  }

  function explainerVital(x) {
    if (exActive(x.job) && x.server === "ready") return "Writing";
    if (x.server !== "ready") return EX_SERVER[x.server] || x.server;
    return x.unload_in_s === undefined ? "Loaded" : `Loaded, off in ${x.unload_in_s} s`;
  }

  on("ex-doctor", () => { audience = "doctor"; setExplain(last); });
  on("ex-plain", () => { audience = "user"; setExplain(last); });
  on("ex-go", () => { if (last.explain.available) send({explain: audience, explain_decision: last.explain.decision}); });
  on("ex-stop", () => send({explain_stop: true}));
  on("ex-close", () => { exRefocus = true; send({explain_close: true}); });

  // -- state ----------------------------------------------------------------------------------------------------------
  function render(s) {
    last = s;
    setText($("head-a"), s.heads[0]);
    setText($("head-b"), s.heads[1]);
    setPicture(s);
    setVerdict(s);
    setControls(s);
    setTray(s);
    setVitals(s);
    setExplain(s);
  }

  function offline() {
    streaming = false;
    document.body.dataset.status = "offline";
    document.body.dataset.picture = "off";
    document.body.dataset.explain = "closed";  // nothing about the board can be vouched for now
    $("badge").className = "badge idle";
    setText($("badge"), "DISCONNECTED");
    setText($("call"), "The board is not answering");
    setText($("reason"), "Check that it is on and on the tailnet. This page reconnects by itself.");
    $("truth").hidden = true;
    $("readout").hidden = true;
    $("starts").hidden = true;
    setText($("meta"), "");
    setText($("framelabel"), "Frame");
    setText($("fieldnote"), "No picture");
    document.querySelectorAll(".vitals dd").forEach(d => { if (d.id !== "v-host") setText(d, "Unknown"); });
    $("v-temp").dataset.heat = "ok";
    $("ex-panel").hidden = true;
    exPicture = null;
    $("ex-go").setAttribute("aria-disabled", "true");
    $("ex-audience").hidden = false;
    $("ex-go").hidden = false;
    $("ex-stop").hidden = true;
    $("ex-close").hidden = true;
    setText($("ex-progress"), "");
    setText($("ex-count"), "");
    setText($("ex-why"), "The board is not answering.");
    $("ex-why").hidden = false;
  }

  // Nothing changes by itself while a picture is held, so a paused board is asked rarely.
  function pollMs(s) {
    const st = s.status, c = s.controls;
    if (st === "reading" || st === "model_loading" || st === "camera_starting") return 300;
    const base = c.feed_paused ? 2000 : c.model_paused ? 1000 : 400;
    // An answer arrives at about 3.5 tokens a second, so this is enough to read along with it.
    return exActive(s.explain.job) ? Math.min(base, EX_POLL_MS) : base;
  }

  const STATE_TIMEOUT_MS = 4000;
  const EX_POLL_MS = 700;
  let timer = 0;
  function schedule(ms) { clearTimeout(timer); timer = setTimeout(tick, ms); }

  async function tick() {
    let s = null;
    try {
      // A stalled network must not leave an old decision on screen looking current, so a slow answer counts as no answer.
      s = await (await fetch("/state", {cache: "no-store", signal: AbortSignal.timeout(STATE_TIMEOUT_MS)})).json();
    } catch (e) {
      offline();
    }
    if (s) render(s);
    if (!document.hidden) schedule(s ? pollMs(s) : 2000);
  }

  // A hidden tab costs the board nothing: no stream, no polling.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { clearTimeout(timer); closeStream(); } else tick();
  });
  tick();
})();
