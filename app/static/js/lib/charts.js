const SVG_NS = "http://www.w3.org/2000/svg";

export function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function mount(container, width, height) {
  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
  });
  container.classList.add("chart");
  container.replaceChildren(svg);
  return svg;
}

function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}

function drawFrame(svg, m, x, y, xTicks, yTicks, fmtX, fmtY, tickClass = "tick-label") {
  for (const t of xTicks) {
    const px = m.left + x(t);
    svg.appendChild(svgEl("line", {
      x1: px, x2: px, y1: m.top, y2: m.top + m.h, class: "gridline",
    }));
    svg.appendChild(svgEl("text", {
      x: px, y: m.top + m.h + 15, "text-anchor": "middle", class: tickClass,
    })).textContent = fmtX(t);
  }
  for (const t of yTicks) {
    const py = m.top + y(t);
    svg.appendChild(svgEl("line", {
      x1: m.left, x2: m.left + m.w, y1: py, y2: py, class: "gridline",
    }));
    svg.appendChild(svgEl("text", {
      x: m.left - 6, y: py + 3.5, "text-anchor": "end", class: tickClass,
    })).textContent = fmtY(t);
  }
  svg.appendChild(svgEl("line", {
    x1: m.left, x2: m.left + m.w, y1: m.top + m.h, y2: m.top + m.h,
    class: "axis-line",
  }));
  svg.appendChild(svgEl("line", {
    x1: m.left, x2: m.left, y1: m.top, y2: m.top + m.h, class: "axis-line",
  }));
}

function axisTitles(svg, m, xlabel, ylabel, titleClass = "axis-title") {
  if (xlabel) {
    svg.appendChild(svgEl("text", {
      x: m.left + m.w / 2, y: m.top + m.h + 34,
      "text-anchor": "middle", class: titleClass,
    })).textContent = xlabel;
  }
  if (ylabel) {
    const y = svg.appendChild(svgEl("text", {
      x: 13, y: m.top + m.h / 2, "text-anchor": "middle", class: titleClass,
      transform: `rotate(-90 13 ${m.top + m.h / 2})`,
    }));
    y.textContent = ylabel;
  }
}

const DEFAULT_M = { left: 56, right: 16, top: 16, bottom: 44 };
function plotBox(width, height, margins = DEFAULT_M) {
  return {
    ...margins,
    w: width - margins.left - margins.right,
    h: height - margins.top - margins.bottom,
  };
}

function setupInspector(svg, m, onMove, onLeave) {
  const overlay = svg.appendChild(svgEl("rect", {
    x: m.left, y: m.top, width: m.w, height: m.h,
    fill: "transparent", cursor: "crosshair",
    style: "touch-action: pan-y; pointer-events: all;",
  }));

  const handleEvent = (evt) => {
    const pt = svg.createSVGPoint();
    if (evt.touches && evt.touches.length > 0) {
      pt.x = evt.touches[0].clientX;
      pt.y = evt.touches[0].clientY;
    } else {
      pt.x = evt.clientX;
      pt.y = evt.clientY;
    }
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const svgP = pt.matrixTransform(ctm.inverse());
    if (svgP.x >= m.left - 4 && svgP.x <= m.left + m.w + 4 &&
        svgP.y >= m.top - 4 && svgP.y <= m.top + m.h + 4) {
      const clampedX = Math.max(m.left, Math.min(m.left + m.w, svgP.x));
      const clampedY = Math.max(m.top, Math.min(m.top + m.h, svgP.y));
      onMove(clampedX, clampedY);
    } else {
      onLeave();
    }
  };

  let lastTouchTime = 0;
  const handleTouch = (evt) => {
    lastTouchTime = Date.now();
    handleEvent(evt);
  };

  const handleLeave = (evt) => {
    if (Date.now() - lastTouchTime < 1500) return;
    if (evt && evt.pointerType === "touch") return;
    onLeave();
  };

  overlay.addEventListener("mousemove", handleEvent);
  overlay.addEventListener("pointermove", (evt) => {
    if (evt.pointerType === "touch") lastTouchTime = Date.now();
    handleEvent(evt);
  });
  overlay.addEventListener("pointerdown", (evt) => {
    if (evt.pointerType === "touch") lastTouchTime = Date.now();
    handleEvent(evt);
  });
  overlay.addEventListener("touchstart", handleTouch, { passive: true });
  overlay.addEventListener("touchmove", handleTouch, { passive: true });
  overlay.addEventListener("mouseleave", handleLeave);
  overlay.addEventListener("pointerleave", handleLeave);
}

function renderTooltip(layer, x, y, lines, clampBox) {
  layer.replaceChildren();
  if (!lines || !lines.length) return;

  const padX = 7;
  const padY = 4;
  const lineH = 13.5;
  const h = lines.length * lineH + padY * 2;
  const maxLen = Math.max(...lines.map((l) => l.length));
  const w = Math.max(maxLen * 6.6 + padX * 2, 56);

  let bx = x - w / 2;
  let by = y - h - 7;

  if (clampBox) {
    if (bx < clampBox.left + 2) bx = clampBox.left + 2;
    if (bx + w > clampBox.left + clampBox.w - 2) bx = clampBox.left + clampBox.w - w - 2;
    if (by < clampBox.top + 2) by = y + 8;
  }

  const bg = svgEl("rect", {
    x: bx, y: by, width: w, height: h, rx: 2,
    fill: "#ffffff", stroke: "var(--line-strong)", "stroke-width": "1",
    style: "filter: drop-shadow(0 1px 3px rgba(0,0,0,0.12));",
  });
  layer.appendChild(bg);

  lines.forEach((txt, idx) => {
    const t = svgEl("text", {
      x: bx + padX, y: by + padY + (idx + 0.82) * lineH,
      style: "font-family: var(--mono); font-size: 10.5px; fill: var(--ink); font-weight: 500;",
    });
    t.textContent = txt;
    layer.appendChild(t);
  });
}

export function histogram(container, { edges, counts, xLabel, median: med,
                                      ci = null, prior = null, width = 560, height = 240 }) {
  const svg = mount(container, width, height);
  const m = plotBox(width, height, { left: 42, right: 12, top: 14, bottom: 38 });
  const xLo = edges[0], xHi = edges[edges.length - 1];
  const cMax = Math.max(...counts, ...(prior || []), 1);
  const x = (v) => ((v - xLo) / (xHi - xLo)) * m.w;
  const y = (c) => m.h - (c / cMax) * (m.h - 6);
  const xTicks = niceTicks(xLo, xHi, 5);
  const yTicks = niceTicks(0, cMax, 3);

  drawFrame(svg, m, x, y, xTicks, yTicks,
            (t) => t.toFixed(1), (t) => Math.round(t));

  const rects = [];
  counts.forEach((c, i) => {
    if (!c) { rects.push(null); return; }
    const r = svg.appendChild(svgEl("rect", {
      x: m.left + x(edges[i]) + 0.5,
      y: m.top + y(c),
      width: Math.max(x(edges[i + 1]) - x(edges[i]) - 0.8, 1),
      height: m.h - y(c),
      class: "series",
    }));
    r.setAttribute("opacity", "0.85");
    rects.push(r);
  });

  if (prior) {
    let d = "";
    prior.forEach((c, i) => {
      const xa = m.left + x(edges[i]); const xb = m.left + x(edges[i + 1]); const yy = m.top + y(c);
      d += `${i ? "L" : "M"}${xa},${yy} L${xb},${yy} `;
    });
    svg.appendChild(svgEl("path", { d, fill: "none", class: "prior-line" }));
  }

  if (ci) {
    for (const v of ci) {
      svg.appendChild(svgEl("line", {
        x1: m.left + x(v), x2: m.left + x(v),
        y1: m.top, y2: m.top + m.h, class: "target-line",
      }));
    }
  }
  if (med !== undefined) {
    svg.appendChild(svgEl("line", {
      x1: m.left + x(med), x2: m.left + x(med),
      y1: m.top, y2: m.top + m.h, class: "axis-line",
    })).setAttribute("stroke-width", "1.6");
  }
  axisTitles(svg, m, xLabel, "samples");

  const inspectGroup = svg.appendChild(svgEl("g", { class: "inspect-layer", style: "pointer-events: none;" }));

  setupInspector(svg, m, (sx) => {
    const frac = Math.max(0, Math.min(1, (sx - m.left) / m.w));
    const v = xLo + frac * (xHi - xLo);
    let binIdx = 0;
    for (let i = 0; i < edges.length - 1; i++) {
      if (v >= edges[i] && v <= edges[i + 1]) { binIdx = i; break; }
      if (i === edges.length - 2) { binIdx = i; }
    }
    rects.forEach((r, idx) => {
      if (!r) return;
      if (idx === binIdx) {
        r.setAttribute("opacity", "1");
        r.setAttribute("stroke", "var(--ink)");
        r.setAttribute("stroke-width", "1.2");
      } else {
        r.setAttribute("opacity", "0.6");
        r.removeAttribute("stroke");
      }
    });

    const bLo = edges[binIdx];
    const bHi = edges[binIdx + 1];
    const count = counts[binIdx];
    const bx = m.left + (x(bLo) + x(bHi)) / 2;
    const by = m.top + y(count);
    renderTooltip(inspectGroup, bx, by, [
      `val: ${bLo.toFixed(2)} to ${bHi.toFixed(2)}`,
      `count: ${count} draws`,
    ], m);
  }, () => {
    rects.forEach((r) => {
      if (!r) return;
      r.setAttribute("opacity", "0.85");
      r.removeAttribute("stroke");
    });
    inspectGroup.replaceChildren();
  });
}

export function scatter(container, { xs, ys, xLabel, yLabel, xMedian = null,
                                     width = 560, height = 280, pointR = 1.8 }) {
  const svg = mount(container, width, height);
  const m = plotBox(width, height, { left: 56, right: 16, top: 16, bottom: 44 });
  const xLo = Math.min(...xs), xHi = Math.max(...xs);
  const yLo = Math.min(...ys), yHi = Math.max(...ys);
  const pad = (h) => (h > 0 ? h * 0.04 : 0.5);
  const x = (v) => ((v - (xLo - pad(xHi - xLo))) /
                    ((xHi + pad(xHi - xLo)) - (xLo - pad(xHi - xLo)))) * m.w;
  const y = (v) => m.h - ((v - (yLo - pad(yHi - yLo))) /
                    ((yHi + pad(yHi - yLo)) - (yLo - pad(yHi - yLo)))) * m.h;
  const xTicks = niceTicks(xLo, xHi, 5);
  const yTicks = niceTicks(yLo, yHi, 5);

  drawFrame(svg, m, x, y, xTicks, yTicks,
            (t) => t.toFixed(2), (t) => t.toFixed(2));

  for (let i = 0; i < xs.length; i++) {
    svg.appendChild(svgEl("circle", {
      cx: m.left + x(xs[i]), cy: m.top + y(ys[i]), r: pointR, class: "pt",
    }));
  }
  if (xMedian !== null) {
    svg.appendChild(svgEl("line", {
      x1: m.left + x(xMedian), x2: m.left + x(xMedian),
      y1: m.top, y2: m.top + m.h, class: "target-line",
    }));
  }
  axisTitles(svg, m, xLabel, yLabel);

  const inspectGroup = svg.appendChild(svgEl("g", { class: "inspect-layer", style: "pointer-events: none;" }));

  setupInspector(svg, m, (sx, sy) => {
    let bestDist = Infinity;
    let bestIdx = 0;
    for (let i = 0; i < xs.length; i++) {
      const px = m.left + x(xs[i]);
      const py = m.top + y(ys[i]);
      const d = (px - sx) ** 2 + (py - sy) ** 2;
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    inspectGroup.replaceChildren();
    const px = m.left + x(xs[bestIdx]);
    const py = m.top + y(ys[bestIdx]);

    inspectGroup.appendChild(svgEl("line", {
      x1: px, x2: px, y1: m.top, y2: m.top + m.h,
      class: "gridline", stroke: "var(--blue)", "stroke-dasharray": "2 2",
    }));
    inspectGroup.appendChild(svgEl("line", {
      x1: m.left, x2: m.left + m.w, y1: py, y2: py,
      class: "gridline", stroke: "var(--blue)", "stroke-dasharray": "2 2",
    }));
    inspectGroup.appendChild(svgEl("circle", {
      cx: px, cy: py, r: 4.5, fill: "var(--blue)", stroke: "#ffffff", "stroke-width": "1.5",
    }));

    const dVal = 10 ** xs[bestIdx];
    const uKmS = (10 ** ys[bestIdx]) / 1000;
    const dStr = dVal >= 1000 ? `${(dVal / 1000).toFixed(2)} km` : `${dVal < 10 ? dVal.toFixed(1) : Math.round(dVal)} m`;
    renderTooltip(inspectGroup, px, py, [
      `Dp: ${dStr} (log ${xs[bestIdx].toFixed(2)})`,
      `U:  ${uKmS.toFixed(1)} km/s (log ${ys[bestIdx].toFixed(2)})`,
    ], m);
  }, () => {
    inspectGroup.replaceChildren();
  });
}

export function lineChart(container, { xs, series, xLabel, yLabel, yTarget =
    null, logX = false, width = 640, height = 280, fmtX = null }) {
  const svg = mount(container, width, height);
  const m = plotBox(width, height, { left: 56, right: 16, top: 32, bottom: 46 });
  const allY = series.flatMap((s) => s.ys).concat(yTarget !== null ? [yTarget] : []);
  const xLo = Math.min(...xs), xHi = Math.max(...xs);
  const yLo = Math.min(...allY), yHi = Math.max(...allY);
  const ySpan = yHi - yLo;
  const yPad = ySpan * 0.15 || 0.05;
  const tf = (v) => (logX ? Math.log10(v) : v);
  const x = (v) => ((tf(v) - tf(xLo)) / (tf(xHi) - tf(xLo) || 1)) * m.w;
  const y = (v) => m.h - ((v - (yLo - yPad)) / ((yHi + yPad) - (yLo - yPad))) * m.h;
  const xTicks = logX
    ? xs.filter((_, i) => i % Math.max(1, Math.floor(xs.length / 5)) === 0)
    : niceTicks(xLo, xHi, 5);
  const yTicks = niceTicks(yLo - yPad, yHi + yPad, 4);
  const yDigits = ySpan < 0.08 ? 3 : 2;

  drawFrame(svg, m, x, y, xTicks, yTicks,
            (t) => (fmtX ? fmtX(t) : logX ? String(t) :
                   String(Math.round(t / 1000)) + "k"),
            (t) => t.toFixed(yDigits));

  if (yTarget !== null) {
    svg.appendChild(svgEl("line", {
      x1: m.left, x2: m.left + m.w, y1: m.top + y(yTarget),
      y2: m.top + y(yTarget), class: "target-line",
    }));
    svg.appendChild(svgEl("text", {
      x: m.left + m.w - 4, y: m.top + y(yTarget) - 5,
      "text-anchor": "end", class: "tick-label",
    })).textContent = `nominal target (${yTarget})`;
  }

  series.forEach((s, si) => {
    const pts = s.ys.map((v, i) =>
      `${m.left + x(xs[i])},${m.top + y(v)}`).join(" ");
    svg.appendChild(svgEl("polyline", {
      points: pts, fill: "none", class: `series-line ${si ? "s" + (si + 1) : ""}`,
    }));
    const lx = m.left + 4 + (si % 5) * 112;
    const ly = 14;
    svg.appendChild(svgEl("line", {
      x1: lx, x2: lx + 14, y1: ly - 3, y2: ly - 3,
      class: `series-line ${si ? "s" + (si + 1) : ""}`,
    }));
    svg.appendChild(svgEl("text", {
      x: lx + 18, y: ly, class: "legend-label",
    })).textContent = s.name;
  });
  axisTitles(svg, m, xLabel, yLabel);

  const inspectGroup = svg.appendChild(svgEl("g", { class: "inspect-layer", style: "pointer-events: none;" }));

  setupInspector(svg, m, (sx) => {
    let closestIdx = 0;
    let closestDist = Infinity;
    xs.forEach((v, i) => {
      const dist = Math.abs((m.left + x(v)) - sx);
      if (dist < closestDist) {
        closestDist = dist;
        closestIdx = i;
      }
    });

    inspectGroup.replaceChildren();
    const cx = m.left + x(xs[closestIdx]);

    inspectGroup.appendChild(svgEl("line", {
      x1: cx, x2: cx, y1: m.top, y2: m.top + m.h,
      class: "gridline", stroke: "var(--line-strong)", "stroke-dasharray": "2 2",
    }));

    const lines = [`Budget: ${Math.round(xs[closestIdx] / 1000)}k sims`];
    series.forEach((s) => {
      const cy = m.top + y(s.ys[closestIdx]);
      inspectGroup.appendChild(svgEl("circle", {
        cx, cy, r: 3.5, fill: "var(--sheet)", stroke: "var(--ink)", "stroke-width": "1.2",
      }));
      lines.push(`${s.name}: ${(s.ys[closestIdx] * 100).toFixed(1)}%`);
    });

    renderTooltip(inspectGroup, cx, m.top + 20, lines, m);
  }, () => {
    inspectGroup.replaceChildren();
  });
}

export function stripPlot(container, { groups, yTarget = 1, yLabel,
                                        width = 640, height = 280 }) {
  const svg = mount(container, width, height);
  const m = plotBox(width, height, { left: 56, right: 16, top: 20, bottom: 44 });
  const all = groups.flatMap((g) => g.values);
  const yLo = Math.min(yTarget, ...all) * 0.92;
  const yHi = Math.max(yTarget, ...all) * 1.08;
  const gw = m.w / groups.length;
  const y = (v) => m.h - ((v - yLo) / (yHi - yLo)) * m.h;
  const yTicks = niceTicks(yLo, yHi, 4);

  for (const t of yTicks) {
    const py = m.top + y(t);
    svg.appendChild(svgEl("line", {
      x1: m.left, x2: m.left + m.w, y1: py, y2: py, class: "gridline",
    }));
    svg.appendChild(svgEl("text", {
      x: m.left - 6, y: py + 3.5, "text-anchor": "end", class: "tick-label",
    })).textContent = t.toFixed(2);
  }
  svg.appendChild(svgEl("line", {
    x1: m.left, x2: m.left + m.w, y1: m.top + y(yTarget),
    y2: m.top + y(yTarget), class: "target-line",
  }));
  svg.appendChild(svgEl("text", {
    x: m.left + m.w - 4, y: m.top + y(yTarget) - 5,
    "text-anchor": "end", class: "tick-label",
  })).textContent = "exact agreement (1.00)";

  const plottedPoints = [];

  groups.forEach((g, gi) => {
    const cx = m.left + gw * (gi + 0.5);
    svg.appendChild(svgEl("text", {
      x: cx, y: m.top + m.h + 16, "text-anchor": "middle", class: "axis-title",
    })).textContent = g.name;
    g.values.forEach((v, i) => {
      const jitter = Math.sin(i * 12.9898) * gw * 0.28;
      const px = cx + jitter;
      const py = m.top + y(v);
      svg.appendChild(svgEl("circle", {
        cx: px, cy: py, r: 2.2, class: "pt",
      }));
      plottedPoints.push({ x: px, y: py, val: v, group: g.name });
    });
  });

  svg.appendChild(svgEl("line", {
    x1: m.left, x2: m.left, y1: m.top, y2: m.top + m.h, class: "axis-line",
  }));
  axisTitles(svg, m, null, yLabel);

  const inspectGroup = svg.appendChild(svgEl("g", { class: "inspect-layer", style: "pointer-events: none;" }));

  setupInspector(svg, m, (sx, sy) => {
    let bestDist = Infinity;
    let bestPt = plottedPoints[0];
    plottedPoints.forEach((pt) => {
      const d = (pt.x - sx) ** 2 + (pt.y - sy) ** 2;
      if (d < bestDist) {
        bestDist = d;
        bestPt = pt;
      }
    });
    if (bestPt) {
      inspectGroup.replaceChildren();
      inspectGroup.appendChild(svgEl("circle", {
        cx: bestPt.x, cy: bestPt.y, r: 4.5, fill: "var(--blue)", stroke: "#ffffff", "stroke-width": "1.5",
      }));
      renderTooltip(inspectGroup, bestPt.x, bestPt.y, [
        `${bestPt.group}`,
        `Ratio: ${bestPt.val.toFixed(3)}`,
      ], m);
    }
  }, () => {
    inspectGroup.replaceChildren();
  });
}

export function profileChart(container, { radii, curves, observed = null,
                                           xLabel = null,
                                           yLabel = "elevation (m)",
                                           width = 560, height = 280 }) {
  const svg = mount(container, width, height);
  const m = plotBox(width, height, { left: 56, right: 16, top: 16, bottom: 44 });
  const rMax = radii[radii.length - 1];
  const useKm = rMax >= 2000;
  const resolvedXLabel = xLabel || (useKm ? "radius (km)" : "radius (m)");
  const all = curves.flatMap((c) => c.values).concat(observed && observed.curve ? observed.curve : []);
  const hMin = Math.min(0, ...all) * 1.06;
  const hMax = Math.max(1, ...all) * 1.15;
  const x = (r) => (r / rMax) * m.w;
  const y = (h) => m.h - ((h - hMin) / (hMax - hMin)) * m.h;
  const rTicks = niceTicks(0, rMax, 4);
  const hTicks = niceTicks(hMin, hMax, 4);

  drawFrame(svg, m, x, y, rTicks, hTicks,
            (t) => (useKm ? (t / 1000).toFixed(rMax >= 20000 ? 0 : 1) : t.toFixed(0)),
            (t) => (Math.abs(t) >= 10000 ? (t / 1000).toFixed(1) + "k" : t.toFixed(0)));

  svg.appendChild(svgEl("line", {
    x1: m.left, x2: m.left + m.w, y1: m.top + y(0), y2: m.top + y(0),
    class: "gridline",
  }));

  curves.forEach((c, ci) => {
    const pts = c.values.map((h, i) =>
      `${m.left + x(radii[i])},${m.top + y(h)}`).join(" ");
    svg.appendChild(svgEl("polyline", {
      points: pts, fill: "none", stroke: "var(--blue)",
      "stroke-width": ci === 0 ? 1.8 : 0.9, opacity: ci === 0 ? 1 : 0.35,
    }));
  });

  if (observed && observed.curve) {
    const pts = observed.curve.map((h, i) => `${m.left + x(radii[i])},${m.top + y(h)}`).join(" ");
    svg.appendChild(svgEl("polyline", { points: pts, fill: "none", class: "observed-line" }));
  }

  if (observed) {
    const rr = observed.rim_radius;
    svg.appendChild(svgEl("line", {
      x1: m.left + x(rr), x2: m.left + x(rr), y1: m.top, y2: m.top + m.h,
      class: "target-line",
    }));
    svg.appendChild(svgEl("text", {
      x: m.left + x(rr) - 6, y: m.top + m.h - 10, "text-anchor": "end", class: "tick-label",
    })).textContent = "rim radius";
  }
  axisTitles(svg, m, resolvedXLabel, yLabel);

  const inspectGroup = svg.appendChild(svgEl("g", { class: "inspect-layer", style: "pointer-events: none;" }));

  setupInspector(svg, m, (sx) => {
    const frac = Math.max(0, Math.min(1, (sx - m.left) / m.w));
    const rIdx = Math.round(frac * (radii.length - 1));
    const rVal = radii[rIdx];
    const hVal = curves[0].values[rIdx];
    const px = m.left + x(rVal);
    const py = m.top + y(hVal);

    inspectGroup.replaceChildren();
    inspectGroup.appendChild(svgEl("line", {
      x1: px, x2: px, y1: m.top, y2: m.top + m.h,
      class: "gridline", stroke: "var(--line-strong)", "stroke-dasharray": "2 2",
    }));
    inspectGroup.appendChild(svgEl("circle", {
      cx: px, cy: py, r: 4, fill: "var(--blue)", stroke: "#ffffff", "stroke-width": "1.4",
    }));

    renderTooltip(inspectGroup, px, py, [
      `radius: ${rVal >= 1000 ? (rVal / 1000).toFixed(2) + " km" : rVal.toFixed(1) + " m"}`,
      `elevation: ${hVal.toFixed(1)} m`,
    ], m);
  }, () => {
    inspectGroup.replaceChildren();
  });
}
