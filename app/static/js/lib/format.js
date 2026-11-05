export function fmtSI(value, unit = "", digits = 3) {
  if (!isFinite(value)) return "n/a";
  const prefixes = [
    [1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""], [1e-2, "c"], [1e-3, "m"],
  ];
  const prefixedUnits = new Set(["m", "m/s", "kg/m3", "kg/m³", "Pa", "g"]);
  let v = value;
  let prefix = "";
  if (prefixedUnits.has(unit) && value !== 0) {
    for (const [f, p] of prefixes) {
      if (Math.abs(value) >= f) { v = value / f; prefix = p; break; }
    }
  }
  const abs = Math.abs(v);
  let text;
  if (abs !== 0 && (abs >= 1e5 || abs < 1e-3)) {
    const exp = Math.floor(Math.log10(abs));
    const mant = v / 10 ** exp;
    text = `${mant.toFixed(2)}e${exp}`;
  } else if (Math.abs(v - Math.round(v)) < 1e-6) {
    text = Math.round(v).toString();
  } else if (abs >= 100) {
    text = v.toFixed(Math.max(0, digits - 3));
  } else if (abs >= 1) {
    text = v.toFixed(Math.max(0, digits - 1));
  } else {
    text = v.toPrecision(digits);
  }
  return `${text}${unit ? " " + prefix + unit : prefix ? " " + prefix : ""}`;
}

export function fmtExp(value, digits = 3) {
  if (!isFinite(value) || value === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(value)));
  const mant = value / 10 ** exp;
  return `${mant.toFixed(digits)}e${exp >= 0 ? '+' : ''}${exp}`;
}

export function fmtPct(x, digits = 0) {
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtDex(v, digits = 2) {
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;
}

export function median(arr) {
  const a = [...arr].sort((x, y) => x - y);
  const n = a.length;
  return n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
}

export function quantile(arr, q) {
  const a = [...arr].sort((x, y) => x - y);
  const pos = (a.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

export function fmtSciHTML(value, digits = 2) {
  if (!isFinite(value) || value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 0.01 && abs < 1e4) return value.toFixed(digits);
  const exp = Math.floor(Math.log10(abs));
  const mant = (value / 10 ** exp).toFixed(digits);
  return `${mant} &times; 10<sup>${exp}</sup>`;
}

export function fmtEnergyTNT(megatons, digits = 2) {
  if (!isFinite(megatons) || megatons <= 0) return "0.0 t TNT";
  if (megatons >= 1e9) {
    const pt = megatons / 1e9;
    if (pt >= 1e3) {
      const exp = Math.floor(Math.log10(pt));
      const mant = (pt / 10 ** exp).toFixed(2);
      return `${mant} &times; 10<sup>${exp}</sup> Pt TNT`;
    }
    return `${pt.toFixed(digits)} Pt TNT`;
  } else if (megatons >= 1e6) {
    return `${(megatons / 1e6).toFixed(digits)} Tt TNT`;
  } else if (megatons >= 1e3) {
    return `${(megatons / 1e3).toFixed(digits)} Gt TNT`;
  } else if (megatons >= 1.0) {
    return `${megatons.toFixed(megatons >= 10 ? 1 : 2)} Mt TNT`;
  } else if (megatons >= 1e-3) {
    return `${(megatons * 1e3).toFixed(megatons * 1e3 >= 10 ? 1 : 2)} kt TNT`;
  } else {
    return `${(megatons * 1e6).toFixed(1)} t TNT`;
  }
}

export function formatFormulaHTML(formulaStr) {
  if (!formulaStr) return "";
  let html = formulaStr
    .replace(/\bD_p\b/g, "<i>D</i><sub><i>p</i></sub>")
    .replace(/\bU\b/g, "<i>U</i>")
    .replace(/\bdelta\b/g, "<i>&delta;</i>")
    .replace(/\brho_t\b/g, "<i>&rho;</i><sub><i>t</i></sub>")
    .replace(/\s\*\s/g, " &middot; ")
    .replace(/\^([-\d.]+)/g, "<sup>$1</sup>");

  html = html.replace(/=\s*([0-9.]+)[eE]\+?([0-9]+)/, (match, mant, expStr) => {
    const exp = parseInt(expStr, 10);
    const val = parseFloat(mant) * Math.pow(10, exp);
    if (val >= 0.01 && val < 10000) {
      return `= ${val.toFixed(2)}`;
    }
    return `= ${parseFloat(mant).toFixed(3)} &times; 10<sup>${exp}</sup>`;
  });
  return html;
}
