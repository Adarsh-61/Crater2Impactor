import { fmtSI, median, fmtEnergyTNT, fmtSciHTML } from "./lib/format.js?v=v41";
import { histogram, scatter, profileChart } from "./lib/charts.js?v=v41";

const $ = (id) => document.getElementById(id);

// Display order and labels follow the original table; keys are the inference quantities.
const PARAMS = [
  ["log10_diameter_m", "logDp", "Impactor diameter"],
  ["log10_velocity_m_s", "logU", "Impact velocity"],
  ["log10_density_kg_m3", "logDelta", "Impactor density"],
  ["log10_strength_pa", "logY", "Target strength"],
  ["log10_target_density_kg_m3", "logRhoT", "Target density"],
  ["angle_from_vertical_deg", null, "Impact angle from vertical"],
];
const DERIVED = [
  ["log10_mass_kg", "logM", "Impactor mass"],
  ["log10_energy_j", "logE", "Kinetic energy"],
  ["log10_momentum_kg_m_s", null, "Linear momentum"],
];
const AXIS = {
  log10_diameter_m: "log₁₀ impactor diameter (m)",
  log10_velocity_m_s: "log₁₀ impact velocity (m/s)",
  log10_density_kg_m3: "log₁₀ impactor density (kg/m³)",
  log10_strength_pa: "log₁₀ target strength (Pa)",
  log10_target_density_kg_m3: "log₁₀ target density (kg/m³)",
  log10_mass_kg: "log₁₀ impactor mass (kg)",
  log10_energy_j: "log₁₀ kinetic energy (J)",
};
const SHORT = {
  log10_diameter_m: "size", log10_velocity_m_s: "speed", log10_density_kg_m3: "impactor density",
  log10_strength_pa: "target strength", log10_target_density_kg_m3: "target density",
};
const STATUS = {
  well_constrained: ["Well constrained", "data-informed"],
  constrained: ["Constrained by the crater", "data-informed"],
  assumption_dependent: ["Depends on assumptions", "assumed"],
  weakly_constrained: ["Weakly constrained", "assumed"],
  prior_dominated: ["Not informed by the crater", "prior-only"],
  unreliable: ["Not reliable: observations incompatible or too few effective samples", "prior-only"],
};
const CLASS_NAME = { regolith: "Regolith", fractured_rock: "Fractured rock (highland crust)", competent_rock: "Competent rock (mare basalt)" };
const PKG_NAME = { lpi2007: "LPI calculator (2007)", holsapple2003: "Holsapple theory (2003)", impacts2022: "Holsapple IMPACTS (2022)" };
const POP_NAME = { asteroidal: "Asteroidal", meteoroid: "Sporadic meteoroid", secondary: "Secondary" };
const BUDGET_NAME = {
  velocity: "Impact speed", impactor_density: "Impactor density", impact_angle: "Impact angle",
  target_density: "Target density", target_strength: "Target strength", angle_law_q: "Angle dependence of crater size",
  target_class: "Target material", scaling_package: "Scaling constants", impactor_population: "Impactor population",
  "crater_size (measurement + model form)": "Diameter measurement and scaling-law scatter",
  "non-linear / interaction": "Interactions between the above",
};

function setErr(id, msg) {
  const box = $(id);
  if (!box) return;
  if (!msg) { box.hidden = true; box.textContent = ""; return; }
  box.hidden = false; box.textContent = msg;
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      if (Array.isArray(data.detail)) detail = data.detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
      else if (data.detail) detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json();
}

function value(id) {
  const el = $(id);
  if (!el || typeof el.value !== "string") return null;
  const raw = el.value.trim();
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function sig3(x) {
  return x >= 0.01 && x < 1e4 ? String(Number(x.toPrecision(3))) : fmtSciHTML(x, 2);
}

function fmtQ(key, v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "n/a";
  if (key === "angle_from_vertical_deg") return `${v.toFixed(0)}°`;
  const x = 10 ** v;
  if (key === "log10_velocity_m_s") return `${(x / 1000).toPrecision(3)} km/s`;
  if (key === "log10_density_kg_m3" || key === "log10_target_density_kg_m3") return `${(x / 1000).toFixed(2)} g/cm³`;
  if (key === "log10_strength_pa") return fmtSI(x, "Pa");
  if (key === "log10_diameter_m") return fmtSI(x, "m");
  if (key === "log10_mass_kg") return `${sig3(x)} kg`;
  if (key === "log10_energy_j") return `${sig3(x)} J`;
  if (key === "log10_momentum_kg_m_s") return `${sig3(x)} kg&middot;m/s`;
  return String(x);
}

// ---------------------------------------------------------------- live input preview (original)
function measurementGeometry() {
  const rim = value("rim"); const depth = value("depth"); const height = value("rh");
  const peakH = value("peak-h"); const floorD = value("floor-d");
  const geo = $("geo-setting") ? $("geo-setting").value : "unconstrained";
  const lat = value("lat"); const lon = value("lon");
  const slot = $("input-profile"); const note = $("input-profile-note");
  if (!slot || !note) return;
  if (!(rim > 0)) {
    slot.innerHTML = `<svg viewBox="0 0 760 275" style="width: 100%; height: auto;" role="img" aria-label="Empty crater geometry guide"><rect width="760" height="275" fill="transparent"/><line x1="20" y1="110" x2="740" y2="110" stroke="#c3d0d2" stroke-width="1" stroke-dasharray="4 4"/><path d="M170 100 C205 125 300 200 380 200 C460 200 555 125 590 100" fill="none" stroke="#8ba1a7" stroke-width="2" stroke-dasharray="5 6"/><text x="380" y="150" fill="#5a686e" text-anchor="middle" font-family="Instrument Sans, sans-serif" font-size="13">enter rim diameter to preview the crater profile</text></svg>`;
    note.textContent = "Enter rim diameter to construct the profile preview.";
    $("live-ratio").textContent = "-";
    $("live-regime").textContent = "Awaiting input dimensions";
    $("live-geo").textContent = "Unconstrained";
    $("live-status").textContent = "Ready";
    return;
  }
  const isComplex = rim >= 10000 || (peakH !== null && peakH > 0) || (floorD !== null && floorD > 0);
  const dNom = isComplex ? (0.2 * 16000 * Math.pow(rim / 16000, 0.301)) : (rim * 0.209);
  const dVal = depth !== null ? depth : dNom;
  const hNom = rim * (isComplex ? 0.025 : 0.036);
  const hVal = height !== null ? height : hNom;
  const ratio = dVal / Math.max(rim, 1);

  const w = 760; const hSvg = 275; const yDatum = 110;
  const xRimL = 170; const xRimR = 590; const xMid = 380; const cavityW = 420;
  const xEjectaL = 85; const xEjectaR = 675;
  const hRel = hVal / Math.max(dVal, 1);
  const rimPx = Math.max(12, Math.min(26, Math.round(hRel * 80 + 10)));
  const yCrest = yDatum - rimPx;
  const depthPx = Math.max(65, Math.min(100, Math.round(ratio * 240 + 45)));
  const yFloor = yDatum + depthPx;

  let pathD = "";
  let peakMarkup = "";
  let xDimD = xMid - 80;
  if (isComplex) {
    const fRatio = Math.min(0.55, Math.max(0.22, (floorD || (rim * 0.40)) / rim));
    const floorHalfW = (cavityW / 2) * fRatio;
    const xFL = xMid - floorHalfW;
    const xFR = xMid + floorHalfW;
    const hasPeak = (peakH !== null && peakH > 0);
    const pkVal = hasPeak ? peakH : 0;
    const pkRatio = Math.min(0.68, Math.max(0.20, pkVal / Math.max(dVal, 1)));
    const pkPx = Math.round(depthPx * pkRatio);
    const yPeak = yFloor - pkPx;
    const pkHalfW = Math.min(32, Math.max(16, floorHalfW * 0.42));
    if (hasPeak && pkVal > 0) {
      pathD = `M ${xEjectaL} ${yDatum} C ${xEjectaL + 40} ${yDatum}, ${xRimL - 25} ${yCrest}, ${xRimL} ${yCrest} C ${xRimL + 25} ${yCrest + 10}, ${xFL - 28} ${yFloor - 5}, ${xFL} ${yFloor} L ${xMid - pkHalfW} ${yFloor} C ${xMid - pkHalfW * 0.5} ${yFloor}, ${xMid - 8} ${yPeak}, ${xMid} ${yPeak} C ${xMid + 8} ${yPeak}, ${xMid + pkHalfW * 0.5} ${yFloor}, ${xMid + pkHalfW} ${yFloor} L ${xFR} ${yFloor} C ${xFR + 28} ${yFloor - 5}, ${xRimR - 25} ${yCrest + 10}, ${xRimR} ${yCrest} C ${xRimR + 25} ${yCrest}, ${xEjectaR - 40} ${yDatum}, ${xEjectaR} ${yDatum}`;
      peakMarkup = `<line x1="${xMid}" y1="${yFloor}" x2="${xMid}" y2="${yPeak}" stroke="#0e4b68" stroke-width="1.1" stroke-dasharray="2 2"/><circle cx="${xMid}" cy="${yPeak}" r="2.5" fill="#0e4b68"/><rect x="${xMid - 45}" y="${yPeak - 20}" width="90" height="16" rx="3" fill="#ffffff" opacity="0.9" stroke="#c9d8dc" stroke-width="0.8"/><text x="${xMid}" y="${yPeak - 8}" fill="#0e4b68" text-anchor="middle" font-family="Instrument Sans, sans-serif" font-size="11" font-weight="600">peak ${fmtSI(pkVal, "m")}</text>`;
    } else {
      pathD = `M ${xEjectaL} ${yDatum} C ${xEjectaL + 40} ${yDatum}, ${xRimL - 25} ${yCrest}, ${xRimL} ${yCrest} C ${xRimL + 25} ${yCrest + 10}, ${xFL - 28} ${yFloor - 5}, ${xFL} ${yFloor} L ${xFR} ${yFloor} C ${xFR + 28} ${yFloor - 5}, ${xRimR - 25} ${yCrest + 10}, ${xRimR} ${yCrest} C ${xRimR + 25} ${yCrest}, ${xEjectaR - 40} ${yDatum}, ${xEjectaR} ${yDatum}`;
    }
    xDimD = Math.max(xRimL + 50, (xRimL + xFL) / 2);
  } else {
    pathD = `M ${xEjectaL} ${yDatum} C ${xEjectaL + 40} ${yDatum}, ${xRimL - 25} ${yCrest}, ${xRimL} ${yCrest} C ${xRimL + 35} ${yCrest + 20}, ${xMid - 80} ${yFloor}, ${xMid} ${yFloor} C ${xMid + 80} ${yFloor}, ${xRimR - 35} ${yCrest + 20}, ${xRimR} ${yCrest} C ${xRimR + 25} ${yCrest}, ${xEjectaR - 40} ${yDatum}, ${xEjectaR} ${yDatum}`;
  }
  const bedrockPoly = `${pathD} L ${xEjectaR} 242 L ${xEjectaL} 242 Z`;
  const depthStr = depth !== null ? fmtSI(depth, "m") : fmtSI(dVal, "m") + " (nominal)";
  const heightStr = height !== null ? fmtSI(height, "m") : fmtSI(hVal, "m") + " (nominal)";
  const rimStr = fmtSI(rim, "m");
  const regimeTitle = isComplex ? (rim >= 50000 ? "Complex Basin" : "Complex Crater") : "Simple Bowl";
  const dPillW = Math.max(120, Math.round(depthStr.length * 6.8 + 48));
  const hPillW = Math.max(130, Math.round(heightStr.length * 6.8 + 66));
  const rPillW = Math.max(150, Math.round(rimStr.length * 6.8 + 82));
  slot.innerHTML = `<svg viewBox="0 0 ${w} ${hSvg}" style="width: 100%; height: auto;" role="img" aria-label="Crater cross-section profile"><defs><marker id="arr-dim" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M 0 1.5 L 3.5 3 L 0 4.5 Z" fill="#1d5e7b"/></marker><marker id="arr-dim-rev" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M 3.5 1.5 L 0 3 L 3.5 4.5 Z" fill="#1d5e7b"/></marker><linearGradient id="crust-grad" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#eaf1f3" stop-opacity="0.6"/><stop offset="100%" stop-color="#f7fafb" stop-opacity="0.1"/></linearGradient></defs><path d="${bedrockPoly}" fill="url(#crust-grad)" stroke="none"/><line x1="20" y1="${yDatum}" x2="${w - 20}" y2="${yDatum}" stroke="#788c92" stroke-width="1" stroke-dasharray="4 4" opacity="0.85"/><text x="24" y="${yDatum - 6}" fill="#485a5f" font-family="Instrument Sans, sans-serif" font-size="10.5" font-weight="600">pre-impact surface</text><path d="${pathD}" fill="none" stroke="#1d5e7b" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/><circle cx="${xRimL}" cy="${yCrest}" r="2.8" fill="#1d5e7b"/><circle cx="${xRimR}" cy="${yCrest}" r="2.8" fill="#1d5e7b"/>${peakMarkup}<line x1="${xRimL}" y1="242" x2="${xRimR}" y2="242" stroke="#1d5e7b" stroke-width="1.1" marker-start="url(#arr-dim-rev)" marker-end="url(#arr-dim)"/><line x1="${xRimL}" y1="${yCrest + 6}" x2="${xRimL}" y2="249" stroke="#1d5e7b" stroke-width="0.8" stroke-dasharray="2 2" opacity="0.4"/><line x1="${xRimR}" y1="${yCrest + 6}" x2="${xRimR}" y2="249" stroke="#1d5e7b" stroke-width="0.8" stroke-dasharray="2 2" opacity="0.4"/><rect x="${xMid - rPillW / 2}" y="233" width="${rPillW}" height="18" rx="3" fill="#ffffff" opacity="0.95" stroke="#c9d8dc" stroke-width="0.8"/><text x="${xMid}" y="246" fill="#1d5e7b" text-anchor="middle" font-family="Instrument Sans, sans-serif" font-size="11.5" font-weight="600">rim diameter: ${rimStr}</text><line x1="${xDimD}" y1="${yCrest}" x2="${xDimD}" y2="${yFloor}" stroke="#1d5e7b" stroke-width="1.1" marker-start="url(#arr-dim-rev)" marker-end="url(#arr-dim)"/><line x1="${xDimD - 6}" y1="${yCrest}" x2="${xRimL}" y2="${yCrest}" stroke="#1d5e7b" stroke-width="0.8" stroke-dasharray="2 2" opacity="0.4"/><line x1="${xDimD - 6}" y1="${yFloor}" x2="${xMid}" y2="${yFloor}" stroke="#1d5e7b" stroke-width="0.8" stroke-dasharray="2 2" opacity="0.4"/><rect x="${xDimD - dPillW / 2}" y="${(yCrest + yFloor) / 2 - 9}" width="${dPillW}" height="18" rx="3" fill="#ffffff" opacity="0.95" stroke="#c9d8dc" stroke-width="0.8"/><text x="${xDimD}" y="${(yCrest + yFloor) / 2 + 4}" fill="#1d5e7b" text-anchor="middle" font-family="Instrument Sans, sans-serif" font-size="11" font-weight="600">depth: ${depthStr}</text><line x1="${xRimR}" y1="${yCrest}" x2="${xRimR}" y2="${yDatum}" stroke="#1d5e7b" stroke-width="1.1" stroke-dasharray="2 2"/><rect x="${xRimR - hPillW / 2}" y="${yCrest - 24}" width="${hPillW}" height="18" rx="3" fill="#ffffff" opacity="0.95" stroke="#c9d8dc" stroke-width="0.8"/><text x="${xRimR}" y="${yCrest - 11}" fill="#1d5e7b" text-anchor="middle" font-family="Instrument Sans, sans-serif" font-size="11" font-weight="600">rim height: ${heightStr}</text><text x="${xMid}" y="26" fill="#1d5e7b" text-anchor="middle" font-family="Instrument Sans, sans-serif" font-size="16" font-weight="600">${regimeTitle} (d/D = ${ratio.toFixed(3)})</text></svg>`;

  let geoNote = "";
  if (geo && geo !== "unconstrained" && geo !== "auto") geoNote = ` &middot; Setting: ${$("geo-setting").selectedOptions[0].textContent}`;
  else if (lat !== null && lon !== null) geoNote = ` &middot; Coordinates: ${lat.toFixed(2)}°, ${lon.toFixed(2)}°`;
  note.innerHTML = `Entered profile: rim ${fmtSI(rim, "m")}, depth ${depth ? fmtSI(depth, "m") : "not entered"}, rim height ${height ? fmtSI(height, "m") : "not entered"}${isComplex ? (peakH !== null ? `, peak ${fmtSI(peakH, "m")}` : "") + (floorD ? `, floor ${fmtSI(floorD, "m")}` : "") : ""}${geoNote}.`;

  $("live-ratio").textContent = ((depth || dNom) / rim).toFixed(3);
  $("live-regime").textContent = isComplex
    ? (rim >= 50000 ? "Complex basin (outside model domain)" : "Complex crater (outside model domain)")
    : "Simple bowl (model domain up to ~10 km)";
  let gText = $("geo-setting").selectedOptions[0].textContent;
  if ((geo === "auto") && lat !== null && lon !== null) gText = `From coordinates (${lat.toFixed(1)}°, ${lon.toFixed(1)}°)`;
  else if (geo === "auto") gText = "Auto: no coordinates entered, global prior";
  $("live-geo").textContent = gText;
  if (depth !== null && depth >= rim) $("live-status").textContent = "Physically impossible: depth ≥ rim diameter";
  else if (height !== null && height >= rim) $("live-status").textContent = "Physically impossible: rim height ≥ rim diameter";
  else if (floorD !== null && floorD >= rim) $("live-status").textContent = "Physically impossible: floor diameter ≥ rim diameter";
  else if (peakH !== null && depth !== null && peakH >= depth) $("live-status").textContent = "Anomalous: peak height ≥ crater depth";
  else if (depth !== null) {
    const dd = depth / rim;
    $("live-status").textContent = dd < 0.11 ? "Shallower than fresh craters (degraded or infilled)" : (dd <= 0.3 ? "Depth consistent with a fresh crater" : "Deeper than fresh craters; check depth");
  } else $("live-status").textContent = "Depth not entered";
}

// ---------------------------------------------------------------- results
function decision(j) {
  const flags = j.flags || [];
  const outDomain = j.input.rim_diameter_m > 10000 || flags.some((f) => /not smaller than|Secondary-impact/.test(f));
  const uInformed = j.quantities.log10_velocity_m_s.status !== "prior_dominated";
  if (outDomain) return ["Outside the model domain", "status-warn", "The crater or the inputs fall outside the range the scaling model covers; see the notes below."];
  if (uInformed) return ["Constrained with independent data", "status-ok", "Independent measurements narrow the speed, and with it the mass."];
  const ev = j.input.event || {};
  const free = ["speed"];
  if ((ev.elev_deg === null || ev.elev_deg === undefined) && !ev.ejecta_pattern) free.push("angle");
  if (ev.delta_kg_m3 === null || ev.delta_kg_m3 === undefined) free.push("impactor density");
  const list = free.length > 1 ? `${free.slice(0, -1).join(", ")} and ${free[free.length - 1]}` : free[0];
  return ["Size and speed trade off", "status-neutral", `The diameter fixes a combination of mass and speed. ${list[0].toUpperCase() + list.slice(1)} follow${free.length > 1 ? "" : "s"} the stated assumptions.`];
}

function insightText(j) {
  const Q = j.quantities;
  const m = Q.log10_mass_kg; const dp = Q.log10_diameter_m; const u = Q.log10_velocity_m_s; const e = Q.log10_energy_j;
  const fmtR = (k, q) => `${fmtQ(k, q.q05)} to ${fmtQ(k, q.q95)}`;
  let t = `For this crater, the model estimates an impactor mass of ${fmtQ("log10_mass_kg", m.median)} (90% range: ${fmtR("log10_mass_kg", m)}), an impactor diameter of ${fmtQ("log10_diameter_m", dp.median)} and a kinetic energy of ${fmtQ("log10_energy_j", e.median)}.`;
  if (u.status === "prior_dominated") t += ` These assume an impact speed of ${fmtR("log10_velocity_m_s", u)}, which the crater does not constrain.`;
  else t += ` The impact speed is ${fmtQ("log10_velocity_m_s", u.median)} (90% range: ${fmtR("log10_velocity_m_s", u)}).`;
  t += ` Across the selected assumptions, the 90% mass range is ${fmtQ("log10_mass_kg", m.envelope90[0])} to ${fmtQ("log10_mass_kg", m.envelope90[1])}.`;
  return t;
}

function renderVerdict(j, input) {
  const [label, cls, sub] = decision(j);
  const state = document.createElement("span");
  state.className = `reading-status ${cls}`;
  state.textContent = label;
  $("decision-tag").replaceChildren(state);
  $("verdict-notes").textContent = sub;
  const warn = $("warn");
  warn.replaceChildren();
  if (j.flags && j.flags.length) {
    warn.hidden = false;
    warn.className = "verdict-warning";
    j.flags.forEach((w) => { const p = document.createElement("p"); p.textContent = w; warn.appendChild(p); });
  } else { warn.hidden = true; }
  $("verdict-sentence").innerHTML = insightText(j);

  const site = j.input.site;
  $("measurements-line").innerHTML = `Input: rim ${fmtSI(input.rim, "m")}, depth ${input.depth === null ? "not entered" : fmtSI(input.depth, "m")}, rim height ${input.rh === null ? "not entered" : fmtSI(input.rh, "m")} &middot; Target context: ${site.label}`;

  const mc = j.morphometry_checks;
  const dd = mc.depth_m.observed_dD;
  $("diag-morph").innerHTML = `
    <div class="diag-item"><span class="diag-label">Rim diameter (D<sub>rim</sub>)</span><span class="diag-val">${fmtSI(input.rim, "m")}</span></div>
    <div class="diag-item"><span class="diag-label">Crater depth (d)</span><span class="diag-val">${mc.depth_m.observed !== null ? fmtSI(mc.depth_m.observed, "m") : "Not entered"}</span></div>
    <div class="diag-item"><span class="diag-label">Depth/diameter ratio</span><span class="diag-val">${dd !== null ? dd.toFixed(3) : "n/a"}</span></div>
    <div class="diag-item"><span class="diag-label">Fresh lunar craters</span><span class="diag-val">d/D ${mc.depth_m.fresh_crater_dD_range[0]} to ${mc.depth_m.fresh_crater_dD_range[1]} (model ${mc.depth_m.model_dD.toFixed(2)})</span></div>
    <div class="diag-item"><span class="diag-label">Depth assessment</span><span class="diag-val">${mc.depth_m.assessment || "n/a"}</span></div>
    <div class="diag-item"><span class="diag-label">Rim height (h<sub>rim</sub>)</span><span class="diag-val">${mc.rim_height_m.observed !== null ? fmtSI(mc.rim_height_m.observed, "m") : "Not entered"} (model ${fmtSI(mc.rim_height_m.model_expected, "m")})</span></div>`;

  const topCls = Object.entries(j.class_probabilities).sort((a, b) => b[1] - a[1])[0][0];
  $("diag-geo").innerHTML = `
    <div class="diag-item"><span class="diag-label">Setting</span><span class="diag-val">${site.label}</span></div>
    <div class="diag-item"><span class="diag-label">Latitude (&phi;)</span><span class="diag-val">${input.lat !== null ? input.lat.toFixed(2) + "&deg;" : "Not entered"}</span></div>
    <div class="diag-item"><span class="diag-label">Longitude (&lambda;)</span><span class="diag-val">${input.lon !== null ? input.lon.toFixed(2) + "&deg;" : "Not entered"}</span></div>
    <div class="diag-item"><span class="diag-label">Mare substrate probability</span><span class="diag-val">${(site.p_mare * 100).toFixed(0)}%</span></div>
    <div class="diag-item"><span class="diag-label">Regolith thickness (prior)</span><span class="diag-val">about 4.5 m mare, 5.3 m highland; not mapped locally</span></div>
    <div class="diag-item"><span class="diag-label">Most probable target</span><span class="diag-val">${CLASS_NAME[topCls]}</span></div>`;

  const ang = j.input.event && j.input.event.elev_deg !== null && j.input.event.elev_deg !== undefined ? 90 - j.input.event.elev_deg : null;
  const melt = mc.melt_volume_m3;
  $("diag-research").innerHTML = `
    <div class="diag-item"><span class="diag-label">Crater regime</span><span class="diag-val">${mc.is_complex_regime ? "Complex (outside model domain)" : "Simple bowl"}</span></div>
    <div class="diag-item"><span class="diag-label">Central peak height</span><span class="diag-val">${mc.central_peak_height_m.observed ? fmtSI(mc.central_peak_height_m.observed, "m") : "Not entered"}</span></div>
    <div class="diag-item"><span class="diag-label">Floor diameter</span><span class="diag-val">${mc.floor_diameter_m.observed ? fmtSI(mc.floor_diameter_m.observed, "m") : "Not entered"}</span></div>
    <div class="diag-item"><span class="diag-label">Impact angle (&theta;<sub>v</sub>)</span><span class="diag-val">${ang !== null ? ang.toFixed(1) + "&deg; from vertical (entered)" : "Unknown (isotropic prior)"}</span></div>
    <div class="diag-item"><span class="diag-label">Continuous ejecta</span><span class="diag-val">${mc.ejecta_radius_m.observed ? fmtSI(mc.ejecta_radius_m.observed, "m") + ` (${mc.ejecta_radius_m.ratio_to_rim_radius.toFixed(1)} rim radii)` : "Not entered"}</span></div>
    <div class="diag-item"><span class="diag-label">Shock melt</span><span class="diag-val">${melt.observed !== null ? fmtSciHTML(melt.observed, 2) + " m&sup3; entered; " : ""}model ${fmtSciHTML(melt.model_q05_q50_q95[1], 2)} m&sup3;</span></div>`;

  const rel = j.mass_velocity_relation;
  $("invariant-card").hidden = false;
  $("invariant-formula").innerHTML = `<i>m</i> &asymp; ${fmtSciHTML(10 ** rel.log10_mass_at_20km_s, 2)} kg &middot; (<i>U</i> / 20 km/s)<sup>${rel.exponent.toFixed(2)}</sup>`;
  $("degradation-note").innerHTML = `Fitted to the posterior; scatter about the curve ${rel.residual_sd_dex.toFixed(2)} dex from target properties, angle and scaling constants. Point-source theory gives an exponent of &minus;3&mu;, about &minus;1.2 in regolith and &minus;1.65 in rock.`;
}

function bar(key, q, bounds) {
  const b = bounds[key];
  if (!b) return "";
  const [lo, hi] = b;
  const left = Math.max(0, ((q.q05 - lo) / (hi - lo)) * 100);
  const width = Math.max(((q.q95 - q.q05) / (hi - lo)) * 100, 0.4);
  const medPos = ((q.median - lo) / (hi - lo)) * 100;
  return `<div class="cibar" aria-label="90 percent range within the prior range"><div class="ci" style="left:${left}%;width:${Math.min(width, 100 - left)}%"></div><div class="med" style="left:${medPos}%"></div></div>`;
}

function renderTable(j) {
  const rows = $("rows"); rows.replaceChildren();
  const ev = j.input.event || {};
  const entered = {
    angle_from_vertical_deg: ev.elev_deg !== null && ev.elev_deg !== undefined,
    log10_velocity_m_s: ev.velocity_m_s !== null && ev.velocity_m_s !== undefined,
    log10_density_kg_m3: ev.delta_kg_m3 !== null && ev.delta_kg_m3 !== undefined,
    log10_mass_kg: ev.mass_kg !== null && ev.mass_kg !== undefined,
  };
  const hasPattern = ev.ejecta_pattern !== null && ev.ejecta_pattern !== undefined;
  const anyEntered = Object.values(entered).some(Boolean) || hasPattern || (ev.flash_energy_j !== null && ev.flash_energy_j !== undefined);
  const add = (key, label, derived) => {
    const q = j.quantities[key];
    let [stext, scls] = STATUS[q.status] || [q.status, "assumed"];
    if (entered[key]) [stext, scls] = ["Set by the entered measurement", "data-informed"];
    else if (key === "angle_from_vertical_deg" && hasPattern && q.status !== "prior_dominated") stext = q.status === "weakly_constrained" ? "Weakly constrained by the ejecta pattern" : "Constrained by the ejecta pattern";
    else if (anyEntered && scls === "data-informed") stext = "Constrained by the crater and entered measurements";
    const tr = document.createElement("tr");
    if (derived) tr.className = "derived-row";
    let mid = fmtQ(key, q.median);
    if (key === "log10_energy_j") mid += `<br><span class="sub-note">${fmtEnergyTNT(10 ** q.median / 4.184e15)}</span>`;
    tr.innerHTML = `
      <td>${label}</td>
      <td><span class="status ${scls}" title="${q.explanation}">${stext}</span></td>
      <td class="numeric">${mid}</td>
      <td class="numeric">${fmtQ(key, q.q05)} to ${fmtQ(key, q.q95)}<br><span class="sub-note">all assumptions: ${fmtQ(key, q.envelope90[0])} to ${fmtQ(key, q.envelope90[1])}</span></td>
      <td>${derived ? "" : bar(key, q, j.prior_bounds)}</td>`;
    rows.appendChild(tr);
  };
  PARAMS.forEach(([k, , label]) => add(k, label, false));
  DERIVED.forEach(([k, , label]) => add(k, label, true));
}

function renderClasses(j) {
  const wrap = $("classes"); wrap.replaceChildren();
  const entries = Object.entries(j.class_probabilities).sort((a, b) => b[1] - a[1]);
  entries.forEach(([name, p]) => {
    const pct = Math.round(p * 100);
    const row = document.createElement("div");
    row.className = "clsbar";
    row.innerHTML = `<span>${CLASS_NAME[name]}</span><div><div class="bar" style="width:${pct}%"></div></div><span class="pct">${pct}%</span>`;
    wrap.appendChild(row);
  });
  $("classes-note").textContent = "Set by the crater size relative to the regolith thickness and by the target setting. The diameter itself does not distinguish target materials.";
}

function buildRadii(sampleCraters) {
  const rMax = Math.max(...sampleCraters.map((c) => c.rim_diameter_m)) * 0.75;
  return Array.from({ length: 161 }, (_, i) => rMax * i / 160);
}

function profileCurve(c, radii) {
  const R = c.rim_diameter_m / 2; const sig = 0.09 * R;
  const c0 = Math.exp(-(R * R) / (2 * sig * sig));
  const effectiveDepth = c.depth_m + c.rim_height_m * c0;
  return {
    values: radii.map((r) => {
      const hRim = c.rim_height_m * Math.exp(-((r - R) ** 2) / (2 * sig * sig));
      const u = Math.max(0, 1 - (r / (0.82 * R)) ** 2);
      return hRim - effectiveDepth * (u ** 1.25);
    }),
  };
}

function renderEvidence(j, input) {
  const H = { logDp: "log10_diameter_m", logU: "log10_velocity_m_s", logDelta: "log10_density_kg_m3", logY: "log10_strength_pa", logRhoT: "log10_target_density_kg_m3", logM: "log10_mass_kg", logE: "log10_energy_j" };
  for (const [slotId, key] of Object.entries(H)) {
    const slot = $(`hist-${slotId}`);
    if (!slot) continue;
    const m = j.marginals[key]; const q = j.quantities[key];
    histogram(slot, { edges: m.edges, counts: m.counts, prior: m.prior_counts || null, xLabel: AXIS[key], median: q.median, ci: [q.q05, q.q95], width: 560, height: 220 });
  }
  const xs = j.scatter.map((r) => r[0]); const ys = j.scatter.map((r) => r[1]);
  scatter($("scatter"), { xs, ys, xLabel: AXIS.log10_diameter_m, yLabel: AXIS.log10_velocity_m_s, xMedian: median(xs), width: 560, height: 280 });
  const C = j.correlations; const pairs = [];
  C.names.forEach((a, i) => C.names.slice(i + 1).forEach((b, n) => pairs.push([a, b, C.matrix[i][i + n + 1]])));
  pairs.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
  $("corr-note").textContent = `Strongest posterior correlations: ${pairs.slice(0, 3).map(([a, b, v]) => `${SHORT[a]} and ${SHORT[b]} (${v >= 0 ? "+" : ""}${v.toFixed(2)})`).join(", ")}. Smaller impactors must be faster to make the same crater; the spread along the band comes from the assumed speed distribution.`;
  const radii = buildRadii(j.sample_craters);
  let observed = null;
  if (input.depth !== null) {
    const obs = { rim_diameter_m: input.rim, depth_m: input.depth, rim_height_m: input.rh !== null ? input.rh : 0.036 * input.rim };
    observed = { rim_radius: input.rim / 2, curve: profileCurve(obs, radii).values };
  } else {
    observed = { rim_radius: input.rim / 2 };
  }
  profileChart($("profiles"), { radii, curves: j.sample_craters.map((c) => profileCurve(c, radii)), observed, width: 560, height: 280 });
  $("profiles-note").textContent = input.depth !== null
    ? "Blue: craters predicted from sampled posterior impactors with the model's fresh-crater shape (d/D 0.21). Dark dashed: the entered profile."
    : "Craters predicted from sampled posterior impactors with the model's fresh-crater shape (d/D 0.21).";
}

function renderSensitivity(j) {
  const rows = $("scenario-rows"); rows.replaceChildren();
  for (const s of j.scenarios) {
    if (s.posterior_weight < 0.001) continue;
    const q = s.quantiles.log10_mass_kg;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${POP_NAME[s.population] || s.population}</td><td>${PKG_NAME[s.package] || s.package}</td><td class="numeric">${Math.round(100 * s.posterior_weight)}%<br><span class="sub-note">assumed ${Math.round(100 * s.prior_weight)}%</span></td><td class="numeric">${fmtQ("log10_mass_kg", q[2])}</td><td class="numeric">${fmtQ("log10_mass_kg", q[0])} to ${fmtQ("log10_mass_kg", q[4])}</td>`;
    rows.appendChild(tr);
  }
  const b = $("budget-rows"); b.replaceChildren();
  Object.entries(j.uncertainty_budget_log10_mass.shares).sort((x, y) => y[1] - x[1]).forEach(([k, v]) => {
    if (v < 0.005) return;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${BUDGET_NAME[k] || k}</td><td class="numeric">${Math.round(v * 100)}%</td>`;
    b.appendChild(tr);
  });
}

function renderAll(j, input) {
  renderVerdict(j, input); renderTable(j); renderClasses(j); renderEvidence(j, input); renderSensitivity(j);
  $("meta-line").textContent = `${j.n_posterior_samples.toLocaleString("en-US")} posterior samples across ${j.diagnostics.n_scenarios} combinations of impactor population and scaling constants.`;
  $("out").hidden = false;
}

function buildPackages(opts) {
  const box = $("pkgs");
  if (!box) return;
  box.replaceChildren();
  for (const [k, p] of Object.entries(opts.packages)) {
    const lab = document.createElement("label");
    lab.className = "pkg-option";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = k; cb.checked = opts.default_packages.includes(k);
    lab.append(cb, document.createTextNode(` ${PKG_NAME[k] || p.label}`));
    box.appendChild(lab);
  }
}

function payload(input) {
  const popSel = $("pop").value;
  const pop = popSel === "artificial_calibrated" ? "artificial" : popSel;
  const psec = Math.min(Math.max(value("psec") ?? 0, 0), 100) / 100;
  const prim = pop === "both" ? { asteroidal: 0.5, meteoroid: 0.5 } : { [pop]: 1 };
  const pops = {};
  for (const [k, v] of Object.entries(prim)) pops[k] = v * (1 - psec);
  if (psec > 0) pops.secondary = psec;
  const pk = [...document.querySelectorAll("#pkgs input:checked")].map((c) => c.value);
  if (!pk.length) throw new Error("Select at least one set of scaling constants.");
  const ev = {};
  if (input.angle !== null) ev.angle_from_vertical_deg = input.angle;
  if (value("ev-u") !== null) { ev.velocity_km_s = value("ev-u"); ev.velocity_sigma_rel = (value("ev-us") ?? 10) / 100; }
  if (value("ev-m") !== null) { ev.mass_kg = value("ev-m"); ev.mass_sigma_dex = value("ev-ms") ?? 0.1; }
  if (value("ev-rho") !== null) ev.impactor_density_kg_m3 = value("ev-rho");
  if (value("ev-f") !== null) {
    ev.flash_energy_j = value("ev-f");
    ev.flash_definition = $("ev-fdef").value;
    ev.flash_sigma_dex = value("ev-fs") ?? 0.15;
  }
  const body = {
    rim_diameter_m: input.rim,
    sigma_meas_dex: Math.log10(1 + (value("sd") ?? 7) / 100),
    geological_setting: input.geo,
    latitude_deg: input.lat, longitude_deg: input.lon,
    depth_m: input.depth, rim_height_m: input.rh,
    central_peak_height_m: input.peakH, floor_diameter_m: input.floorD,
    ejecta_radius_m: input.ejectaR, melt_volume_m3: input.meltVol,
    populations: pops, packages: pk,
    ejecta_pattern: $("ejecta-pattern").value || null,
    sigma_mf_dex: value("smf") ?? 0.1,
    mass_prior_b: $("mb").value === "" ? null : parseFloat($("mb").value),
  };
  if (popSel === "artificial_calibrated") body.spacecraft_calibration = true;
  if ($("rocky").value) body.rocky_ejecta = $("rocky").value === "true";
  if (Object.keys(ev).length) body.event = ev;
  return body;
}

async function run(event) {
  event.preventDefault();
  setErr("err", "");
  $("out").hidden = true;
  const input = {
    rim: value("rim"), depth: value("depth"), rh: value("rh"),
    geo: $("geo-setting").value, lat: value("lat"), lon: value("lon"),
    peakH: value("peak-h"), floorD: value("floor-d"), ejectaR: value("ejecta-r"),
    angle: value("angle"), meltVol: value("melt-vol"), pattern: $("ejecta-pattern").value,
  };
  if (!(input.rim > 0)) { setErr("err", "Rim diameter must be a positive value."); return; }
  if (input.angle !== null && (input.angle < 0 || input.angle >= 90)) { setErr("err", "Impact angle from vertical must be between 0 and 90 degrees."); return; }
  $("run").disabled = true;
  $("run").querySelector("span").textContent = "Estimating…";
  try {
    const j = await fetchJSON("/api/v2/infer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload(input)) });
    renderAll(j, input);
    $("out").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setErr("err", `Inference failed: ${error.message}`);
  } finally {
    $("run").disabled = false;
    $("run").querySelector("span").textContent = "Estimate impact parameters";
  }
}

const formEl = $("f");
formEl.addEventListener("submit", run);

document.querySelectorAll(".preset-chip").forEach((button) => button.addEventListener("click", () => {
  $("rim").value = button.dataset.rim || "";
  $("depth").value = button.dataset.depth || "";
  $("rh").value = button.dataset.rh || "";
  $("geo-setting").value = button.dataset.geo || "unconstrained";
  $("lat").value = button.dataset.lat || "";
  $("lon").value = button.dataset.lon || "";
  $("peak-h").value = button.dataset.peakh || "";
  $("floor-d").value = button.dataset.floord || "";
  $("ejecta-r").value = button.dataset.ejectar || "";
  $("angle").value = button.dataset.angle || "";
  $("melt-vol").value = button.dataset.meltvol || "";
  $("ejecta-pattern").value = button.dataset.pattern || "";
  ["ev-u", "ev-us", "ev-rho", "ev-m", "ev-ms", "ev-f"].forEach((id) => { $(id).value = ""; });
  $("ev-u").value = button.dataset.u || "";
  if (button.dataset.u) { $("ev-us").value = button.dataset.us || "2"; $("event-panel").open = true; }
  $("ev-f").value = button.dataset.flash || "";
  $("ev-fdef").value = button.dataset.fdef || "luminous_range_integrated";
  $("ev-fs").value = button.dataset.fs || "0.15";
  if (button.dataset.flash) $("event-panel").open = true;
  $("pop").value = button.dataset.pop || "asteroidal";
  if (button.dataset.angle || button.dataset.ejectar || button.dataset.meltvol || button.dataset.pattern) $("research-panel").open = true;
  measurementGeometry();
  formEl.requestSubmit();
}));

["rim", "depth", "rh", "lat", "lon", "peak-h", "floor-d", "ejecta-r", "angle", "melt-vol"].forEach((id) => {
  const el = $(id);
  if (el) el.addEventListener("input", measurementGeometry);
});
$("geo-setting").addEventListener("change", measurementGeometry);

fetchJSON("/api/v2/options").then(buildPackages).catch((e) => setErr("err", `Could not load model options: ${e.message}`));
measurementGeometry();
const refreshRestoredGeometry = () => setTimeout(measurementGeometry, 250);
window.addEventListener("load", refreshRestoredGeometry, { once: true });
window.addEventListener("pageshow", refreshRestoredGeometry);
