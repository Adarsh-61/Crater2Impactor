import { fmtSI, fmtSciHTML, fmtEnergyTNT } from "./lib/format.js?v=v42";
import { profileChart } from "./lib/charts.js?v=v42";

const $ = (id) => document.getElementById(id);

const REGIME_TEXT = {
  gravity: "Gravity regime: lunar gravity, not target strength, stops crater growth; size depends on a combination of impactor momentum and energy.",
  intermediate: "Intermediate regime: gravity and material strength both shape the resulting crater cavity.",
  strength: "Strength regime: target strength limits excavation; the scaling constants are least certain here.",
};

function profileCurve(radii, rimD, depth, rimH, isComplex = false, peakH = 0) {
  const R = rimD / 2;
  const sig = 0.09 * R;
  const c0 = Math.exp(-(R * R) / (2 * sig * sig));
  const D_eff = depth + rimH * c0;
  const floorR = isComplex ? R * 0.40 : 0;
  const peakSig = 0.12 * R;
  return {
    values: radii.map((r) => {
      const hRim = rimH * Math.exp(-((r - R) ** 2) / (2 * sig * sig));
      if (isComplex && floorR > 0) {
        if (r <= floorR) {
          const pk = peakH > 0 ? peakH * Math.exp(-(r * r) / (2 * peakSig * peakSig)) * Math.max(0, 1 - (r / floorR) ** 2) : 0;
          return -depth + pk + hRim;
        } else if (r < R) {
          const w = (r - floorR) / (R - floorR);
          return -depth * (1 - w ** 1.5) + hRim;
        }
        return hRim;
      }
      const inner = 0.82 * R;
      const u = Math.max(0, 1 - (r / inner) ** 2);
      return hRim - D_eff * (u ** 1.25);
    }),
  };
}

function kv(entries) {
  const grid = document.createDocumentFragment();
  for (const [k, v] of entries) {
    const d = document.createElement("div");
    d.className = "kv-row";
    const kk = document.createElement("span");
    kk.className = "k";
    kk.innerHTML = k;
    const vv = document.createElement("span");
    vv.className = "v";
    vv.innerHTML = v;
    d.append(kk, vv);
    grid.appendChild(d);
  }
  return grid;
}

let lastRadii = null;

function render(res) {
  const rMax = res.rim_diameter_m * 1.6;
  const radii = [];
  for (let i = 0; i <= 200; i++) radii.push((rMax * i) / 200);
  lastRadii = radii;
  const exag = $("exag").checked;
  const curve = profileCurve(radii, res.rim_diameter_m, res.depth_m, res.rim_height_m, false, 0);
  if (exag) {
    curve.values = curve.values.map((h) => h * 6);
  }
  profileChart($("xsec"), {
    radii,
    curves: [curve],
    observed: { rim_radius: res.rim_diameter_m / 2 },
    yLabel: exag ? "elevation (m, 6x vertical exaggeration)" : "elevation (m)",
    width: 560,
    height: 280,
  });

  const dims = $("dims");
  dims.replaceChildren();
  dims.append(kv([
    ["Rim diameter (D<sub>rim</sub>)", fmtSI(res.rim_diameter_m, "m")],
    ["Range across scaling models", `${fmtSI(res.rim_diameter_range_m[0], "m")} to ${fmtSI(res.rim_diameter_range_m[1], "m")}`],
    ["Depth below rim (d)", fmtSI(res.depth_m, "m")],
    ["Rim height above datum (h<sub>rim</sub>)", fmtSI(res.rim_height_m, "m")],
    ["Excavation diameter (D<sub>exc</sub>)", fmtSI(res.excavation_diameter_m, "m")],
  ]));

  const pis = $("pis");
  pis.replaceChildren();
  pis.append(kv([
    ["&pi;<sub>2</sub> (gravity group)", fmtSciHTML(res.pi2, 2)],
    ["&pi;<sub>3</sub> (strength group)", fmtSciHTML(res.pi3, 2)],
    ["&pi;<sub>V</sub> (volume scaling)", fmtSciHTML(res.pi_volume, 2)],
    ["Excavation regime", res.regime_label],
    ["Scaling constants", res.package_label],
  ]));
  $("regime-note").textContent = REGIME_TEXT[res.regime_label] || "";

  const dn = $("domain-note");
  dn.replaceChildren();
  if (!res.is_simple_crater) {
    const box = document.createElement("div");
    box.className = "warnbox";
    box.innerHTML = "<strong>Domain notice</strong><p>The predicted rim diameter exceeds about 10 km. Larger lunar craters collapse into complex forms (terraced walls, flat floors, central peaks), which these simple-crater equations do not describe.</p>";
    dn.appendChild(box);
  }
  if (res.notes.length) {
    const box = document.createElement("div");
    box.className = "warnbox";
    box.innerHTML = "<strong>Validity notice</strong><p>" + res.notes.join(" ") + "</p>";
    dn.appendChild(box);
  }
  const rn = document.createElement("p");
  rn.className = "figure-caption";
  rn.style.textAlign = "left";
  rn.textContent = `Depth is the model's excavation depth (d/D 0.21); fresh lunar craters are shallower (d/D 0.11 to 0.17). Range across scaling models: ${res.range_note}`;
  dn.appendChild(rn);

  const der = $("derived");
  der.replaceChildren();
  const meloshT = res.formation_time_melosh_s || Math.sqrt(res.rim_diameter_m / 1.62);
  const keTNT = fmtEnergyTNT(res.kinetic_energy_j / 4.184e15);
  const p_momentum = res.momentum_kg_m_s;
  const derivedEntries = [
    ["Transient volume (V<sub>t</sub>)", `${fmtSciHTML(res.transient_volume_m3, 2)} m&sup3;`],
    ["Formation time (Holsapple)", fmtSI(res.formation_time_s, "s")],
    ["Excavation time (Melosh)", fmtSI(meloshT, "s")],
    ["Impactor mass (m)", `${fmtSciHTML(res.impactor_mass_kg, 2)} kg`],
    ["Linear momentum (p)", `${fmtSciHTML(p_momentum, 2)} kg&middot;m/s`],
    ["Kinetic energy (E<sub>k</sub>)", `${fmtSciHTML(res.kinetic_energy_j, 2)} J (${keTNT})`],
  ];
  if (res.ejecta_mass_kg) {
    derivedEntries.push(["Ejecta mass", `${fmtSciHTML(res.ejecta_mass_kg, 2)} kg`]);
  }
  if (res.melt_volume_m3 && res.melt_volume_m3 > 0) {
    derivedEntries.push(["Shock melt volume (LPI 2007 relation)", `${fmtSciHTML(res.melt_volume_m3, 2)} m&sup3;`]);
  }
  der.append(kv(derivedEntries));
}

function clearResults() {
  $("dims")?.replaceChildren();
  $("pis")?.replaceChildren();
  $("derived")?.replaceChildren();
  $("domain-note")?.replaceChildren();
  if ($("regime-note")) $("regime-note").textContent = "";
  $("xsec")?.replaceChildren();
}

async function update() {
  const dia = parseFloat($("dia").value);
  const vel = parseFloat($("vel").value);
  const den = parseFloat($("den").value);
  const str = parseFloat($("str").value);
  const tden = parseFloat($("tden").value);
  const ang = parseFloat($("ang").value);
  const mat = parseInt($("mat").value, 10);
  if (!(dia > 0 && vel > 0 && den > 0 && str >= 0 && tden > 0)) {
    setErr("Impactor and target values must be positive numbers (strength may be zero).");
    clearResults();
    return;
  }
  if (!(ang >= 0 && ang < 90)) {
    setErr("Impact angle from vertical must be between 0 and 90 degrees.");
    clearResults();
    return;
  }
  try {
    const res = await fetchJSON({
      impactor_diameter_m: dia, velocity_km_s: vel, impactor_density_kg_m3: den,
      target_strength_pa: str, target_density_kg_m3: tden, target_class: mat,
      angle_from_vertical_deg: ang, package: $("pkg").value,
    });
    setErr("");
    render(res);
  } catch (e) {
    setErr(`Forward model failed: ${e.message}`);
    clearResults();
  }
}

async function fetchJSON(body) {
  const res = await fetch("/api/v2/forward_point", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      if (Array.isArray(data.detail)) {
        detail = data.detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
      } else if (typeof data.detail === "object" && data.detail !== null) {
        detail = JSON.stringify(data.detail);
      } else if (data.detail) {
        detail = data.detail;
      }
    } catch {
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json();
}

function setErr(msg) {
  const box = $("err");
  if (!box) return;
  if (!msg) { box.hidden = true; box.textContent = ""; return; }
  box.hidden = false;
  box.textContent = msg;
}

for (const id of ["dia", "vel", "den", "mat", "str", "tden", "ang", "pkg"]) {
  $(id).addEventListener("input", update);
  $(id).addEventListener("change", update);
}
$("exag").addEventListener("change", update);
document.querySelector("form").addEventListener("submit", (event) => { event.preventDefault(); update(); });
update();
