"""Render docs/VALIDATION.md from results/benchmarks_v2.json (numbers are never typed by hand)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def s3(x):
    return f"{x:+.3f}"


def _label(x):
    return x.replace("v2 exact engine", "v2 engine")


def main():
    r = json.loads((REPO / "results" / "benchmarks_v2.json").read_text())
    B = r["blocks"]
    p = r["provenance"]
    L = []
    w = L.append
    w("# Validation report (generated)\n")
    w(f"Generated from `results/benchmarks_v2.json` by `scripts/write_validation_doc.py`. Benchmark run "
      f"{p['started_utc']}, script sha256 {p['script_sha256'][:12]}, Python {p['python']}, NumPy {p['numpy']}, "
      f"quick={p['quick']}.\n")
    w("Test types:\n")
    for k, v in r["legend"].items():
        w(f"- **{k}**: {v}")
    w("\nOnly P, W and L involve real measurements. **None of them measures per-crater accuracy for natural "
      "impactors**, because no natural lunar crater has an independently known impactor.\n")

    if "P" in B:
        b = B["P"]
        k = "compact (2700 kg/m3, aluminium)"
        w("## P. Measured impacts (hollow, slow spacecraft)\n")
        w("| Event | Q | D obs (m) | D pred median [90%] (m) | log10 obs/pred | m known (kg) | m recovered median [90%] (kg) | log10 err | PIT |")
        w("|---|---|---|---|---|---|---|---|---|")
        for e in b["events"]:
            f = e["forward"][k]
            i = e["inverse_mass"]
            mk = f"{e['mass_range_kg'][0]}–{e['mass_range_kg'][1]}" if e.get("mass_range_kg") else f"{e['known_mass_kg']}"
            w(f"| {e['label']} | {e['quality']} | {e['observed_d_m']:.1f} | {f['predicted_median_m']:.1f} "
              f"[{f['predicted_90_m'][0]:.1f}, {f['predicted_90_m'][1]:.1f}] | {s3(f['log10_residual_obs_minus_pred'])} | "
              f"{mk} | {i['median_kg']:.3g} [{i['q05_kg']:.2g}, {i['q95_kg']:.2g}] | "
              f"{s3(i['log10_error_median_minus_true'])} | {i['pit_of_true_mass']:.2f} |")
        s = b["summary"]
        w(f"\nQuality A/B (n = {s['n_quality_AB']}): forward median residual {s3(s['forward_median_log10_residual_AB'])} dex, "
          f"inside 90%: {s['forward_inside_90_AB']}. Inverse median mass error {s3(s['inverse_median_log10_error_AB'])} dex, "
          f"inside 90%: {s['inverse_inside_90_AB']}.")
        if "inverse_pit_sd_AB" in s:
            w(f"Spread: forward RMS residual {s['forward_rms_log10_residual_AB']:.3f} dex, inverse RMS mass error "
              f"{s['inverse_rms_log10_error_AB']:.3f} dex; SD of the PIT values {s['inverse_pit_sd_AB']:.2f} (0.29 for "
              "calibrated intervals; smaller means the intervals are wider than these errors need). "
              f"Events no session had examined before the test (held out): {s['n_held_out']}.")
        w("Excluded: " + "; ".join(f"{x['id']} ({x['reason']})" for x in b["excluded"]) + ".")
        w("\nCaveats:\n"
          f"- All impactors are hollow and travelled at {min(e['velocity_km_s'] for e in b['events']):.1f}–"
          f"{max(e['velocity_km_s'] for e in b['events']):.1f} km/s, outside the compact hypervelocity regime.\n"
          "- n is tiny, and the intervals are wide.\n"
          "- σ_mf = 0.1 dex was chosen with the Apollo 13/14/16 S-IVB craters in view. No other default parameter was set from these events, but all 11 were examined during development (they fit the opt-in SIGMA_MF_ARTIFICIAL table), so this is not a held-out test (development_use in data/validation/spacecraft_impacts.json).\n")

    if "W" in B:
        b = B["W"]
        w("## W. Population test: LROC new craters vs bolide flux × scaling\n")
        bins = ["≥" + str(a) if c > 1e8 else f"{a}–{c}" for a, c in b["bins_m"]]
        w("| | " + " | ".join(bins) + " | total |")
        w("|---|" + "---|" * (len(bins) + 1))
        w("| observed | " + " | ".join(str(x) for x in b["observed_counts"]) + f" | {sum(b['observed_counts'])} |")
        for k, v in b["packages"].items():
            w(f"| predicted, {k} | " + " | ".join(f"{x:.1f}" for x in v["predicted_counts_at_delta0_F1"])
              + f" | {v['predicted_total_ge7m_at_delta0_F1']:.1f} |")
        w("\n| Package | Δ log10 D, median [90%] | log10 Y_regolith (Pa), median [90%] | N≥7/N≥12 pred vs obs |")
        w("|---|---|---|---|")
        for k, v in b["packages"].items():
            d, y, sl = v["delta_log10_D_posterior"], v["regolith_strength_log10_pa_posterior"], v["slope_check"]
            w(f"| {k} | {s3(d['median'])} [{s3(d['q05'])}, {s3(d['q95'])}] | {y['median']:.2f} [{y['q05']:.2f}, {y['q95']:.2f}] | "
              f"{sl.get('predicted_N_ge7_over_N_ge12', float('nan')):.2f} vs {sl.get('observed_N_ge7_over_N_ge12', float('nan')):.2f} |")
        if "sensitivity_lpi2007" in b:
            w("\nSensitivity (LPI 2007 package):\n")
            w("| Variant | observed | predicted (Δ=0, F=1) | Δ median [90%] |")
            w("|---|---|---|---|")
            for k, v in b["sensitivity_lpi2007"].items():
                d = v["delta_posterior"]
                w(f"| {k} | {v['observed_total']} | {v['predicted_total_delta0_F1']:.1f} | {s3(d['median'])} [{s3(d['q05'])}, {s3(d['q95'])}] |")
        w(f"\n{b['flux_note']}\n")
        w("Exposure and counts were digitised from Speyerer et al. (2016) Fig. 2, because the Methods are paywalled. "
          "Flux and crater-size normalisation are degenerate, so this test checks consistency and constrains one combination.\n")

    if "L" in B:
        b = B["L"]
        rr = b["recovered"]
        w("## L. Law recovery on laboratory data\n")
        w(f"{b['data']}.\n")
        w(f"- Recovered μ = {rr['mu']:.3f} (90% {rr['mu_90'][0]:.3f}–{rr['mu_90'][1]:.3f}); published dry sand 0.41.")
        w(f"- Recovered ν = {rr['nu']:.3f} (90% {rr['nu_90'][0]:.3f}–{rr['nu_90'][1]:.3f}).")
        g = b["gravity_trend_log10_D10G_over_D502G"]
        w(f"- Held-out cross-laboratory gravity trend: observed {g['observed']:.3f}, predicted {g['predicted']:.3f}; "
          "level offsets " + ", ".join(s3(x["log10_pred_over_obs"]) for x in b["held_out"]) + " dex.")
        if "engine_constants_held_out" in b:
            w("\nThe engine's **shipped** regolith constants (not the fit above), strength ≈ 0, vertical impacts, "
              "apparent radius:\n")
            w("| Material | In sample? | n | mean log10 obs/pred | SD (dex) | inside 90% |")
            w("|---|---|---|---|---|---|")
            for name, x in b["engine_constants_held_out"].items():
                if name == "all_held_out":
                    continue
                sd = "–" if x["sd_dex"] is None else f"{x['sd_dex']:.3f}"
                w(f"| {name} | {'yes' if x['in_sample'] else 'no'} | {x['n']} | {s3(x['mean_log10_obs_over_pred'])} | "
                  f"{sd} | {x['inside_90']} |")
            a = b["engine_constants_held_out"]["all_held_out"]
            w(f"\nAll held-out granular shots: mean {s3(a['mean_log10_obs_over_pred'])} dex, RMS {a['rms_dex']:.3f} dex, "
              f"inside 90%: {a['inside_90']}. The level offset of the Flintshot-only fit above (about +0.06 dex on the "
              "Ormö shots) is a property of that fit, not of the engine.\n")
        for n in b["notes"]:
            w(f"- {n}")
        w("")

    if "T" in B:
        b = B["T"]
        c = b["cross_validation"]
        w("## T. Location test\n")
        w(f"Leave-one-out log-RMSE of local regolith thickness ({c['n_points']} measurements): interpolated map "
          f"{c['rmse_ln_idw']:.3f}; mare/highland mean {c['rmse_ln_terrain_mean']:.3f}; global mean "
          f"{c['rmse_ln_global_mean']:.3f}. Decision: {b['decision']}\n")

    if "M" in B:
        w("## M. Mathematical correctness\n")
        w("| Crater (m) | fiber q05/q50/q95 | independent | max diff (dex) |")
        w("|---|---|---|---|")
        for x in B["M"]["results"]:
            w(f"| {x['crater_m']} | {' / '.join(f'{v:.2f}' for v in x['fiber_q05_q50_q95'])} | "
              f"{' / '.join(f'{v:.2f}' for v in x['independent_q05_q50_q95'])} | {x['max_abs_diff_dex']:.3f} |")
        w("")

    for key, title in (("C", "C. In-model calibration"), ("R", "R. Robustness under misspecification")):
        if key not in B:
            continue
        w(f"## {title}\n")
        nd = {x.get("n_draws_per_crater") for x in B[key]["results"]}
        if nd != {None}:
            w(f"Importance-sampling draws per crater: {', '.join(f'{v:,}' for v in sorted(v for v in nd if v))}.\n")
        w("| Truth set | quantity | 90% coverage (±MC se) | 68% coverage | envelope coverage | median 90% width (dex) | n |")
        w("|---|---|---|---|---|---|---|")
        for x in B[key]["results"]:
            for q, v in x["per_quantity"].items():
                w(f"| {x['label']} | {q} | {v['coverage90']:.3f} ± {v['coverage90_mc_se']:.3f} | {v['coverage68']:.3f} | "
                  f"{v['envelope_coverage90']:.3f} | {v['median_width90_dex']:.2f} | {x['n']} |")
        w("")

    if "B" in B:
        b = B["B"]["results"]
        w("## B. Baselines (common synthetic test set; truth drawn from the declared model set)\n")
        w("| Method | quantity | 90% coverage | median abs error (dex) | median 90% width | interval score | time (s) |")
        w("|---|---|---|---|---|---|---|")
        names = {"lm": "mass", "le": "energy", "lu": "velocity"}
        for m, v in b["methods"].items():
            for q, x in v.items():
                t = x["seconds_total"]
                t = "–" if t is None or t != t else f"{t:.1f}"
                w(f"| {_label(m)} | {names[q]} | {x['coverage90']:.3f} | {x['median_abs_error_dex']:.3f} | "
                  f"{x['median_width90_dex']:.2f} | {x['interval_score90']:.2f} | {t} |")
        w(f"\nn = {b['n']}" + (f", {b['n_draws_per_crater']:,} importance-sampling draws per crater" if "n_draws_per_crater" in b else "")
          + ". Because the synthetic truth comes from the engine's own "
          "declared model set, this comparison favours the engine by construction; it is not evidence about real "
          "craters. The time column is wall-clock for the whole set and depends on machine load; the clean "
          "sequential per-crater timing is in results/comparison_original_vs_v2.json (runtime block). Velocity is never better than the prior-only baseline.\n")

    (REPO / "docs" / "VALIDATION.md").write_text("\n".join(L) + "\n")
    print("wrote docs/VALIDATION.md")


if __name__ == "__main__":
    main()
