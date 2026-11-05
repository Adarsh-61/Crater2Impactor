import { lineChart, stripPlot } from "./lib/charts.js?v=v42";
import { fmtSI } from "./lib/format.js?v=v42";

const $ = (id) => document.getElementById(id);
const pct = (x) => `${Math.round(100 * x)}%`;
const dex = (x) => `${x.toFixed(2)} dex`;
const kg2 = (x) => Number(x.toPrecision(2)).toLocaleString("en-US");
const sgn = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)}`;
const QNAME = { log10_mass_kg: "Impactor mass", log10_energy_j: "Kinetic energy", log10_velocity_m_s: "Impact speed",
  log10_momentum_kg_m_s: "Momentum", log10_diameter_m: "Impactor diameter" };
const PKG = { lpi2007: "LPI calculator (2007)", holsapple2003: "Holsapple theory (2003)", impacts2022: "Holsapple IMPACTS (2022)" };

function kv(el, rows) {
  el.innerHTML = rows.map(([k, v]) => `<div class="diag-item"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
}
function addRow(tbody, cells) {
  const tr = document.createElement("tr");
  tr.innerHTML = cells.map((c, i) => `<td${i ? ' class="numeric"' : ""}>${c}</td>`).join("");
  tbody.appendChild(tr);
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

function pooledCoverage(C) {
  const keys = ["log10_mass_kg", "log10_energy_j", "log10_velocity_m_s"];
  const out = {};
  for (const k of keys) {
    let n = 0, c90 = 0, c68 = 0; const w = [];
    for (const r of C.results) {
      const q = r.per_quantity[k];
      n += r.n; c90 += q.coverage90 * r.n; c68 += q.coverage68 * r.n; w.push(q.median_width90_dex);
    }
    out[k] = { n, c90: c90 / n, c68: c68 / n, w: w.reduce((a, b) => a + b, 0) / w.length };
  }
  return out;
}

function render(bm, cmp) {
  const B = bm.blocks;
  const pc = pooledCoverage(B.C);
  const A1 = cmp.A_synthetic && cmp.A_synthetic.A1_original_world;
  const RT = cmp.runtime;
  const perCrater = RT ? RT.v2_median_s : null;
  const perCraterOld = RT ? RT.original_median_s : null;
  kv($("sum-method"), [
    ["Approach", "Posterior of the stated model, computed directly"],
    ["Neural network", "Not used (previous version kept for comparison)"],
    ["Scaling constants", "Three published sets, compared"],
    ["Median time per crater", perCrater ? `${perCrater.toFixed(2)} s (previous: ${perCraterOld.toFixed(2)} s)` : "n/a"],
  ]);
  kv($("sum-cal"), [
    ["Mass, 90% coverage", pct(pc.log10_mass_kg.c90)],
    ["Energy, 90% coverage", pct(pc.log10_energy_j.c90)],
    ["Speed, 90% coverage", pct(pc.log10_velocity_m_s.c90)],
    ["Test craters", pc.log10_mass_kg.n.toLocaleString("en-US")],
  ]);
  const P = B.P.summary; const W = B.W.packages.lpi2007; const L = B.L.recovered;
  kv($("sum-real"), [
    ["Spacecraft crater diameters inside predicted 90% range", P.forward_inside_90_AB],
    ["Diameter error, spacecraft", `${sgn(P.forward_median_log10_residual_AB)} dex (median)`],
    ["New craters 7 m and larger", `${W.predicted_total_ge7m_at_delta0_F1.toFixed(0)} predicted, ${W.observed_total_ge7m} observed`],
    ["Scaling exponent from laboratory sand", `${L.mu.toFixed(3)} (published 0.41)`],
  ]);

  // calibration table
  const cr = $("cal-rows"); cr.replaceChildren();
  for (const [k, v] of Object.entries(pc)) addRow(cr, [QNAME[k], pct(v.c68), pct(v.c90), dex(v.w), v.n.toLocaleString("en-US")]);
  $("cal-summary").textContent = `Pooled over synthetic craters drawn from each of the three sets of scaling constants (5 m to 5 km). The sampling uncertainty of each coverage value is about ${Math.round(100 * Math.sqrt(0.09 / (pc.log10_mass_kg.n / 3)))} percentage points per set.`;

  // measured impacts
  const C = cmp.C_real_impacts;
  const rr = $("real-rows"); rr.replaceChildren();
  const byId = Object.fromEntries(B.P.events.map((e) => [e.id, e]));
  for (const e of C.events) {
    const p = byId[e.id];
    const label = p ? p.label : e.id;
    const fm = (a) => a ? `${kg2(a[1])} kg<br><span class="sub-note">${kg2(a[0])} to ${kg2(a[2])} kg</span>` : "n/a";
    addRow(rr, [
      `${label}${e.quality === "C" ? '<br><span class="sub-note">mass from a secondary source</span>' : ""}`,
      `${e.observed_d_m.toFixed(1)} m`,
      `${e.v2_forward_median_m.toFixed(1)} m<br><span class="sub-note">${e.v2_forward_90_m[0].toFixed(0)} to ${e.v2_forward_90_m[1].toFixed(0)} m</span>`,
      `${e.known_mass_kg.toLocaleString("en-US")} kg`,
      fm(e.final_inverse_mass_q05_q50_q95_kg || e.v2_inverse_mass_q05_q50_q95_kg),
      fm(e.final_with_speed_and_angle_mass_q05_q50_q95_kg || e.v2_with_speed_and_angle_mass_q05_q50_q95_kg),
    ]);
  }
  const inv = C.inverse_mass_AB;
  const invD = inv.final_diameter_only || inv.v2_diameter_only;
  const invE = inv.final_with_speed_and_angle || inv.v2_with_speed_and_angle;
  $("real-err-d").textContent = invD.log10_mass_kg.mae.toFixed(2);
  $("real-err-e").textContent = invE.log10_mass_kg.mae.toFixed(2);
  $("real-summary").innerHTML = `${C.events.filter((e) => e.quality !== "C").length} well-documented impacts. Predicted diameter: median offset ${sgn(P.forward_median_log10_residual_AB)} dex, all inside the 90% range. Mass from the crater alone assumes natural asteroid speeds (about 18 km/s), so for these slow spacecraft it is too low by ${Math.abs(invD.log10_mass_kg.bias).toFixed(1)} dex on average; given the known speed and angle, the mass error is ${invE.log10_mass_kg.mae.toFixed(2)} dex.`;
  const AB = C.events.filter((e) => e.quality !== "C");
  stripPlot($("real-strip"), {
    groups: [
      { name: "previous physics", values: AB.map((e) => e.observed_d_m / e.original_forward_45deg_m) },
      { name: "current model", values: AB.map((e) => e.observed_d_m / e.v2_forward_median_m) },
    ],
    yTarget: 1, yLabel: "observed / predicted diameter", height: 240,
  });

  // comparison table
  const t = $("cmp-rows"); t.replaceChildren();
  const add = (name, prev, cur, keys) => {
    for (const k of keys) {
      if (!prev || !cur || !prev[k] || !cur[k]) continue;
      addRow(t, [name, QNAME[k], dex(prev[k].mae), dex(cur[k].mae), pct(prev[k].coverage90), pct(cur[k].coverage90),
        prev[k].interval_score90.toFixed(2), cur[k].interval_score90.toFixed(2)]);
    }
  };
  const A = cmp.A_synthetic || {};
  if (A.A1_original_world) add("Synthetic, previous model's own assumptions", A.A1_original_world.original_app_diameter_only, (A.A1_original_world.final_diameter_only || A.A1_original_world.v2_diameter_only), ["log10_mass_kg", "log10_energy_j", "log10_velocity_m_s"]);
  if (A.A1_original_world) add("Same, previous version also given simulated depth and other measurements", A.A1_original_world.original_network_full_14_channel_context, (A.A1_original_world.final_diameter_only || A.A1_original_world.v2_diameter_only), ["log10_mass_kg", "log10_energy_j"]);
  if (A.A2_v2_world) add("Synthetic, current model's assumptions", A.A2_v2_world.original_app_diameter_only, (A.A2_v2_world.final_diameter_only || A.A2_v2_world.v2_diameter_only), ["log10_mass_kg", "log10_energy_j", "log10_velocity_m_s"]);
  if (A.A3_both_misspecified) add("Synthetic, assumptions wrong for both", A.A3_both_misspecified.original_app_diameter_only, (A.A3_both_misspecified.final_diameter_only || A.A3_both_misspecified.v2_diameter_only), ["log10_mass_kg", "log10_energy_j"]);
  add("Spacecraft craters, diameter only", C.inverse_mass_AB.original_app, invD, ["log10_mass_kg"]);
  const n1 = A.A1_original_world ? (A.A1_original_world.final_diameter_only || A.A1_original_world.v2_diameter_only).n : 0;
  $("cmp-summary").textContent = `Error is the mean absolute difference between the posterior median and the true value. Synthetic sets have about ${n1} craters each; the spacecraft set has 5. Each version is favoured on synthetic craters drawn from its own assumptions, so those rows show behaviour under correct and incorrect assumptions rather than accuracy in nature. In the previous version's simulation, depth and the other measurements are generated from the same impactor, so they help by construction; whether real crater depths carry this information has not been tested. On the spacecraft craters both versions assume natural impact speeds; the current version is worse there unless the known speed is entered.`;

  // population
  const bins = B.W.bins_m.map((b) => b[0]);
  const cum = (arr) => arr.map((_, i) => arr.slice(i).reduce((a, b) => a + b, 0));
  const series = [{ name: "observed", xs: bins, ys: cum(B.W.observed_counts) }];
  for (const [k, v] of Object.entries(B.W.packages)) series.push({ name: PKG[k] || k, xs: bins, ys: cum(v.predicted_counts_at_delta0_F1) });
  lineChart($("pop-chart"), { xs: bins, series, xLabel: "crater diameter (m)", yLabel: "new craters at least this large", logX: true, fmtX: (x) => String(x), height: 280 });
  const pr = $("pop-rows"); pr.replaceChildren();
  for (const [k, v] of Object.entries(B.W.packages)) {
    const d = v.delta_log10_D_posterior; const y = v.regolith_strength_log10_pa_posterior;
    addRow(pr, [PKG[k] || k, v.predicted_total_ge7m_at_delta0_F1.toFixed(1), String(v.observed_total_ge7m),
      `${sgn(d.q05)} to ${sgn(d.q95)} dex`, `${fmtSI(10 ** y.q05, "Pa")} to ${fmtSI(10 ** y.q95, "Pa")}`]);
  }
  $("pop-summary").textContent = `Survey exposure ${(B.W.exposure_km2_yr / 1e6).toFixed(2)} million km²·yr. The impact rate itself is uncertain by about a factor of four, which is included; a crater size offset consistent with zero means the scaling law and the impact rate agree. Survey totals were read from the published figure.`;

  // laboratory
  const lab = $("lab-rows"); lab.replaceChildren();
  addRow(lab, ["Scaling exponent recovered from dry-sand impacts", `${L.mu.toFixed(3)} (90%: ${L.mu_90[0].toFixed(3)} to ${L.mu_90[1].toFixed(3)})`, "0.41 in the published constants"]);
  const g = B.L.gravity_trend_log10_D10G_over_D502G;
  addRow(lab, ["Gravity dependence predicted for an independent laboratory", `${g.predicted.toFixed(3)}`, `${g.observed.toFixed(3)} measured`]);
  const off = B.L.held_out.map((x) => x.log10_pred_over_obs);
  addRow(lab, ["Size offset for that laboratory's sand", `${sgn(Math.min(...off))} to ${sgn(Math.max(...off))} dex`, "material-specific coefficient"]);
  const BL = cmp.B_laboratory;
  if (BL) {
    addRow(lab, ["Crater volume error, previous physics", `${BL.original_production_v1.mae.toFixed(3)} dex`, `${BL.n} sand impacts`]);
    for (const k of ["lpi2007", "holsapple2003", "impacts2022"]) addRow(lab, [`Crater volume error, ${PKG[k]}`, `${BL["v2_" + k].mae.toFixed(3)} dex`, `${BL.n} sand impacts`]);
  }

  // robustness
  const rb = $("rob-rows"); rb.replaceChildren();
  const RN = {
    "angle law q=1.3 (steeper than declared 0.5-1.0)": "Crater size depends more strongly on impact angle than assumed",
    "model-form residual 0.2 dex (declared 0.1)": "Real craters scatter twice as much about the scaling law",
    "30% secondaries, engine assumes primaries only": "30% of craters are secondaries, not declared",
    "30% secondaries, engine told P(secondary)=0.3": "30% of craters are secondaries, declared",
    "meteoroid population, engine assumes asteroidal": "Impactors are meteoroids, asteroids assumed",
  };
  for (const r of B.R.results) addRow(rb, [RN[r.label] || r.label, pct(r.per_quantity.log10_mass_kg.coverage90), pct(r.per_quantity.log10_velocity_m_s.coverage90)]);

  // location
  const T = B.T.cross_validation;
  $("loc-text").textContent = `Coordinates determine whether the crater lies on mare basalt or highland crust, and the model uses this. They do not predict the local regolith thickness: across ${T.n_points} published measurements, an interpolated map predicted thickness worse (log error ${T.rmse_ln_idw.toFixed(2)}) than a single average value (${T.rmse_ln_global_mean.toFixed(2)}). The model therefore uses average thicknesses for mare and highland terrain.`;
}

Promise.all([getJSON("/api/v2/benchmarks"), getJSON("/api/v2/comparison").catch(() => ({}))])
  .then(([bm, cmp]) => render(bm, cmp))
  .catch((e) => { $("err").textContent = `Could not load validation results: ${e.message}`; $("err").hidden = false; });

getJSON("/api/v2/findings")
  .then((F) => {
    const tb = $("find-rows"); tb.replaceChildren();
    for (const f of F.findings) {
      const tr = document.createElement("tr");
      for (const c of [f.finding, f.evidence, f.result, f.status, `${f.id}: ${f.source.file.split("/").pop()}`]) {
        const td = document.createElement("td"); td.textContent = c; tr.appendChild(td);
      }
      tb.appendChild(tr);
    }
  })
  .catch(() => { $("find-rows").closest("section").hidden = true; });
