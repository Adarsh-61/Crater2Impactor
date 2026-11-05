"""Deterministic symmetry-reduced convolution (SRC) backend for one scenario.

Computes the same posterior as sampler.run_scenario by quadrature instead of importance sampling. It uses the
engine's fibre solver (packages.solve_log10_m), the crater-dependent log10-mass support sampler.mass_support(D),
integrates the density exponent nu exactly, and raises SRCUnsupported for inputs it cannot represent exactly so
that callers can fall back to Monte Carlo.

Mathematics: at fixed crater size the rim law depends on the target density and strength only through psi = Y rho_ref / rho_t (density-ratio packages; lpi2007: psi = Y (delta /
rho_t)^(1/3) rho_ref / rho_t), and log10 m = x(s, psi) + target-density shift + kinematic shift, so the posterior of
log10 m is a convolution of a target pmf with line-supported kinematic pmfs on a 0.005 dex grid, followed by the
mass-prior factor 10^(-b log10 m). No random numbers are used.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfcx, ndtr, roots_legendre

from .flash import FLASH_DEFINITIONS
from .observations import ejecta_pattern_loglik
from .packages import PACKAGES, effective_strength, solve_log10_m
from .populations import POPULATIONS
from .sampler import EventData, ModelForm, log_ndtr_diff, mass_support, validate_event
from .targets import (CLASS_PRIORS, COMPETENT, DH_BEDROCK_MIN, DH_REGOLITH_MAX, FRACTURED, GLOBAL_CONTEXT,
                      LAYERED_EXTRA_SIGMA_DEX, REGOLITH, SiteContext, regolith_probability)

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


class SRCUnsupported(ValueError):
    """The SRC backend cannot represent this input exactly; use the Monte Carlo sampler."""



def _G2(z: np.ndarray) -> np.ndarray:
    """Antiderivative of Phi(z): integral_{-inf}^z Phi(t) dt = z * Phi(z) + phi(z)."""
    return z * ndtr(z) + INV_SQRT_2PI * np.exp(-0.5 * np.square(z))


def _G2_tail_scaled(u: np.ndarray) -> np.ndarray:
    """G2(-u) exp(u^2 / 2) for u >= 0, free of cancellation: G2(-u) = phi(u) - u Phi(-u) and
    Phi(-u) = erfcx(u / sqrt 2) exp(-u^2 / 2) / 2."""
    return INV_SQRT_2PI - 0.5 * u * erfcx(u / math.sqrt(2.0))


def _log_g2_second_difference(t1, t2, t3, t4):
    """log(G2(t1) - G2(t2) - G2(t3) + G2(t4)) for t1 >= t2, t3 >= t4 (the trapezoid integral of the flash
    likelihood). Where all four arguments have the same sign the terms are evaluated in the far tail, scaled by
    the largest one: G2(t) = t + G2(-t), and the linear parts cancel in the second difference, so for t >= 0 the
    difference equals that of G2(-t). Elsewhere the direct formula has no cancellation problem."""
    ts = np.stack(np.broadcast_arrays(*(np.asarray(t, dtype=np.float64) for t in (t1, t2, t3, t4))))
    sgn = np.array([1.0, -1.0, -1.0, 1.0])[:, None]
    tail = (ts.min(axis=0) >= 0.0) | (ts.max(axis=0) <= 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = np.log(np.sum(sgn * _G2(ts), axis=0))
        u = np.abs(ts)
        umin = u.min(axis=0)
        scaled = np.sum(sgn * _G2_tail_scaled(u) * np.exp(-0.5 * (u**2 - umin**2)), axis=0)
        tail_val = -0.5 * umin**2 + np.log(scaled)
    return np.where(tail, tail_val, direct)


def analytic_flash_loglik(
    log10_flash_energy_j: float,
    flash_sigma_dex: float,
    lek: np.ndarray,
    eta_range: tuple[float, float] = (5e-4, 6e-3),
    f_range: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Exact closed-form marginal log-likelihood of a lunar impact flash over (log10 eta, log10 F).

    Normalised like the sampler's unnormalised Gaussian term exp(-z^2 / 2), i.e. sqrt(2 pi) sigma times the
    density. Accurate in the far tails (a flash many sigma from the crater's prediction keeps a finite, correctly
    sloped log-likelihood instead of a constant floor)."""
    eta_lo = math.log10(eta_range[0])
    eta_hi = math.log10(eta_range[1])
    w_eta = eta_hi - eta_lo
    f_lo, f_hi = f_range
    w_f = f_hi - f_lo
    sig = float(flash_sigma_dex)

    delta = (float(log10_flash_energy_j) - np.asarray(lek, dtype=np.float64)) / sig
    a0 = eta_lo / sig
    a1 = eta_hi / sig

    if w_f <= 1e-12:
        f0 = f_lo / sig
        return math.log(math.sqrt(2.0 * math.pi) * sig / w_eta) + log_ndtr_diff(delta + f0 - a1, delta + f0 - a0)

    b0 = f_lo / sig
    b1 = f_hi / sig
    log_sd = _log_g2_second_difference(delta + b1 - a0, delta + b0 - a0, delta + b1 - a1, delta + b0 - a1)
    return math.log(math.sqrt(2.0 * math.pi) * (sig ** 2) / (w_eta * w_f)) + log_sd


_SRC_DX = 0.005
# x = log10 m + 3 mu log10 U reaches ~24-25 for D ~ 100 km; the upper edge leaves margin (overflow is an error,
# see _splat_1d)
_SRC_X_LO, _SRC_X_HI = -16.0, 34.0
_SRC_N_X = int(round((_SRC_X_HI - _SRC_X_LO) / _SRC_DX)) + 1
_SRC_GRID_X = np.linspace(_SRC_X_LO, _SRC_X_HI, _SRC_N_X)

# the kinematic shift -3 mu log10 U (U in m/s, solver reference 1 m/s) reaches -8.0 for mu = 0.55 at 72 km/s, so the
# lower edge must lie well below -8
_SRC_B_LO, _SRC_B_HI = -12.0, 6.0
_SRC_N_B = int(round((_SRC_B_HI - _SRC_B_LO) / _SRC_DX)) + 1
_SRC_ZERO_B_IDX = int(round((0.0 - _SRC_B_LO) / _SRC_DX))
# grids for log10 kinetic energy (J) and log10 momentum (kg m/s)
_SRC_E_LO, _SRC_E_HI = -14.0, 42.0
_SRC_N_E = int(round((_SRC_E_HI - _SRC_E_LO) / _SRC_DX)) + 1
_SRC_GRID_E = np.linspace(_SRC_E_LO, _SRC_E_HI, _SRC_N_E)
_SRC_P_LO, _SRC_P_HI = -15.0, 37.0
_SRC_N_P = int(round((_SRC_P_HI - _SRC_P_LO) / _SRC_DX)) + 1
_SRC_GRID_P = np.linspace(_SRC_P_LO, _SRC_P_HI, _SRC_N_P)


def _gl01(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = roots_legendre(n)
    return 0.5 * (x + 1.0), 0.5 * w


_U6, _W6 = _gl01(6)
_U8, _W8 = _gl01(8)
_U10, _W10 = _gl01(10)
_U12, _W12 = _gl01(12)
_U16, _W16 = _gl01(16)
_U20, _W20 = _gl01(20)
_U24, _W24 = _gl01(24)
_U36, _W36 = _gl01(36)
_U48, _W48 = _gl01(48)

# Crater-size nodes. The likelihood of the noise-free size s is N(y, sig^2), but under the mass prior
# 10^(-b log10 m), with log10 m ~ s / slope, the posterior of s is that normal shifted down by
# sig^2 b ln10 / slope (slope >= 2 / (3 (2 + mu)) > 1 / TAIL_SLOPE_INV), which exceeds 5 sig once sig > ~0.3 dex.
# The node window therefore reaches below y by 5 sig plus that shift, but not below the sizes made at the
# lower mass bound (s_floor = lo / 3 - 1, a 1 mm crater for 1 mg).
TAIL_SLOPE_INV = 4.0


def _s_window_below(y: float, sig: float, b: float, lo_m: float) -> float:
    """Distance (in units of sig) from y to the lowest crater-size node."""
    z = 5.0 + TAIL_SLOPE_INV * b * math.log(10.0) * sig
    return max(5.0, min(z, (y - (lo_m / 3.0 - 1.0)) / sig))


def _normal_nodes(n: int, z_lo: float = 5.0):
    """Gauss-Legendre nodes on [-z_lo, 5] with weights proportional to the standard normal density; the node count
    grows with the window so that the spacing stays that of n nodes on [-5, 5]."""
    n = int(math.ceil(n * (z_lo + 5.0) / 10.0))
    u, w = _gl01(n)
    z = -z_lo + (z_lo + 5.0) * u
    w = (z_lo + 5.0) * w * np.exp(-0.5 * z**2) * INV_SQRT_2PI
    return z, w / w.sum()


N_S_MIX_BINS = 400   # q05 converges to ~2e-4 dex at 400 bins (160 bins: ~0.002 dex; 1000 bins reference)


def _mixture_nodes(y: float, sig: np.ndarray, w: np.ndarray, n_bins: int, z_lo: float = 6.0):
    """Nodes and weights for s ~ sum_k w_k N(y, sig_k^2).

    Bin edges are sinh-stretched around y, out to -z_lo max(sig) below and +6 max(sig) above, so narrow and wide
    components are both resolved. Each bin carries its exact mixture mass sum_k w_k [Phi(b_k) - Phi(a_k)] and sits
    at the exact conditional mean of the mixture in that bin (Gaussian truncated means), so mass and first moment
    are exact bin by bin. The tails beyond the window (prior mass < 2e-9, posterior mass negligible by the choice
    of z_lo) are folded into the outer bins."""
    w = np.asarray(w, float) / np.sum(w)
    smax = float(np.max(sig))
    z_lo = max(6.0, z_lo)
    n_lo = int(math.ceil(n_bins / 2 * z_lo / 6.0))
    u = np.concatenate([np.linspace(-1.0, 0.0, n_lo + 1)[:-1], np.linspace(0.0, 1.0, n_bins // 2 + 1)])
    edges = y + np.where(u < 0, z_lo, 6.0) * smax * np.sinh(3.0 * u) / math.sinh(3.0)
    edges[0], edges[-1] = -np.inf, np.inf
    z = (edges[None, :] - y) / sig[:, None]                 # (K, n_bins + 1)
    cdf = ndtr(z)
    pdf = np.where(np.isfinite(z), np.exp(-0.5 * np.where(np.isfinite(z), z, 0.0) ** 2) * INV_SQRT_2PI, 0.0)
    mass_k = cdf[:, 1:] - cdf[:, :-1]                        # (K, n_bins)
    first_k = sig[:, None] * (pdf[:, :-1] - pdf[:, 1:])      # int (s - y) N ds over each bin
    mass = w @ mass_k
    first = w @ first_k
    keep = mass > 0
    return y + first[keep] / mass[keep], mass[keep]


def _observed_nodes(obs: float, sd: float, lo: float, hi: float):
    """48 Gauss-Legendre nodes and weights for a normal measurement density N(obs, sd^2) restricted to the prior
    support [lo, hi] (weights = node weight x measurement density, so they sum to the probability mass inside).
    The window is the support within 8 sd of obs; when obs lies outside the support it starts at the nearest
    bound and is shortened to 30 / z sd, z the distance in sd, where the measurement density falls by e^-30."""
    z_out = max((lo - obs) / sd, (obs - hi) / sd, 0.0)
    width = 8.0 if z_out == 0.0 else min(8.0, 30.0 / z_out)
    if obs < lo:
        a, b_ = lo, min(hi, lo + width * sd)
    elif obs > hi:
        a, b_ = max(lo, hi - width * sd), hi
    else:
        a, b_ = max(lo, obs - 8.0 * sd), min(hi, obs + 8.0 * sd)
    x = a + (b_ - a) * _U48
    return x, (b_ - a) * _W48 * np.exp(-0.5 * ((x - obs) / sd) ** 2) * INV_SQRT_2PI / sd


# Quadrature sizes for the crater-size variable s and the target strength Y. With the speed, angle and impactor
# density all observed, the kinematic kernel is narrow and does not smooth discrete s- and Y-node spikes, so the node
# counts must be large: with 36 and 24/16 nodes the Apollo 16 S-IVB case (speed 2 %, angle 3 deg, density known) is
# off by up to 0.014 dex, while 96/48/32 and 256/96/64 agree to 1e-4 dex.
N_S_NODES, N_Y_NODES, N_Y_NODES_LPI = 96, 48, 32


def _splat_1d(vals: np.ndarray, weights: np.ndarray, lo: float, n_grid: int, dx: float) -> np.ndarray:
    """Fast O(N) vectorized tent-kernel splatting via np.bincount."""
    pos = (vals - lo) / dx
    idx = np.floor(pos).astype(np.int64)
    frac = pos - idx
    valid = (idx >= 0) & (idx < n_grid - 1)
    if not np.all(valid):
        # A share of the weight above 1e-6 off the grid aborts the SRC evaluation (infer_crater backend "auto"
        # then falls back to Monte Carlo). Below 1e-6 (e.g. the angle floor sin(theta) = 1e-6 at grazing
        # incidence) the loss is negligible and the points are dropped
        tot = float(np.sum(np.abs(weights)))
        lost = float(np.sum(np.abs(np.where(valid, 0.0, weights))))
        if tot > 0 and lost > 1e-6 * tot:
            raise SRCUnsupported(f"SRC grid overflow: {lost / tot:.2e} of the weight falls outside [{lo}, {lo + (n_grid - 1) * dx}]")
        idx = np.clip(idx, 0, n_grid - 2)
        frac = np.clip(frac, 0.0, 1.0)
        weights = np.where(valid, weights, 0.0)
    out = np.bincount(idx, weights=weights * (1.0 - frac), minlength=n_grid).astype(np.float64)
    out += np.bincount(idx + 1, weights=weights * frac, minlength=n_grid)
    return out


def _lconv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full linear convolution computed directly on the non-zero spans of a and b.

    An FFT convolution spreads round-off of ~1e-16 x max over the whole grid; the mass-prior factor 10^(-b log10 m)
    then amplifies that floor at low masses by up to 10^(0.9 x 15) for km-size craters, enough to move q05 of a
    3 km crater by up to 0.016 dex depending on the grid bounds. The splatted pmfs are exactly zero
    outside their support, so a direct convolution of the non-zero spans is exact and cheap."""
    out = np.zeros(a.size + b.size - 1)
    ia, ib = np.flatnonzero(a), np.flatnonzero(b)
    if ia.size == 0 or ib.size == 0:
        return out
    a0, a1, b0, b1 = ia[0], ia[-1] + 1, ib[0], ib[-1] + 1
    out[a0 + b0: a1 + b1 - 1] = np.convolve(a[a0:a1], b[b0:b1])
    return out


# lpi2007 target pmf by psi windows. ln psi = ln Y + (ln delta - ln rho)/3 - ln rho + ln rho_ref and Y is
# log-uniform, so for each (rho, delta) node pair ln psi is uniform on a window of fixed length ln(y_hi / y_lo).
# x0(s, psi) is solved once on a fine psi grid; each psi node's s-pmf is splatted once; a pair's pmf is the
# hat-function-weighted sum of node pmfs over its window (one matrix product), shifted by
# log10(rho / rho_ref) - (mu / 2) log10(delta / rho). Replaces the 3-D (rho, Y, delta) point cloud (~1e6 points).
LPI_WINDOW_METHOD = True
N_PSI_LPI = 96   # 96 nodes agree with the 3-D quadrature to 2e-4 dex (48-128 tested)


def _lpi_target_pmf(s_all, ws_all, rho_nodes, w_rho, delta_nodes, w_delta, ln_y_lo, ln_y_hi, rho_ref, ln_psi_lo,
                    ln_psi_hi, mu_c, k1, k2, kr, g, rim_factor):
    lg = np.linspace(ln_psi_lo, ln_psi_hi, N_PSI_LPI)
    h = lg[1] - lg[0]
    x0, isl = _x0_isl_surface(s_all, np.exp(lg), 0, np.array([k1]), np.array([k2]), np.array([mu_c]), np.array([kr]),
                              np.array([False]), rho_ref, g, False, rim_factor)
    wsl = ws_all[:, None] * isl
    # node pmfs on a local window of the x grid
    i_lo = int(np.floor((x0.min() - 2 * _SRC_DX - _SRC_X_LO) / _SRC_DX))
    i_hi = int(np.ceil((x0.max() + 2 * _SRC_DX - _SRC_X_LO) / _SRC_DX)) + 1
    nloc = i_hi - i_lo
    pos = (x0 - (_SRC_X_LO + i_lo * _SRC_DX)) / _SRC_DX
    idx = np.floor(pos).astype(np.int64)
    frac = pos - idx
    rows = np.broadcast_to(np.arange(N_PSI_LPI)[None, :], x0.shape)
    flat = (rows * nloc + idx).ravel()
    P = np.bincount(flat, weights=(wsl * (1.0 - frac)).ravel(), minlength=N_PSI_LPI * nloc)
    P += np.bincount(flat + 1, weights=(wsl * frac).ravel(), minlength=N_PSI_LPI * nloc)
    P = P[: N_PSI_LPI * nloc].reshape(N_PSI_LPI, nloc)
    # hat-function weights of each (rho, delta) window on the psi grid: exact integral of the piecewise-linear
    # interpolant over [a, b], divided by the window length (uniform density in ln Y)
    lr, ld = np.meshgrid(np.log(rho_nodes), np.log(delta_nodes), indexing="ij")
    c = (ld - lr) / 3.0 + math.log(rho_ref) - lr
    a, b = (ln_y_lo + c).ravel(), (ln_y_hi + c).ravel()
    t = (lg[None, :] - lg[0]) / h
    ta, tb = (a[:, None] - lg[0]) / h, (b[:, None] - lg[0]) / h

    def hat_cdf(x):   # integral of the hat centred at t from -inf to x (units of h)
        d = x - t
        return np.where(d <= -1, 0.0, np.where(d <= 0, 0.5 * (d + 1) ** 2, np.where(d <= 1, 1.0 - 0.5 * (1 - d) ** 2, 1.0)))
    W = (hat_cdf(tb) - hat_cdf(ta)) / (tb - ta)                    # (pairs, N_PSI_LPI), rows sum to 1
    W *= (w_rho[:, None] * w_delta[None, :]).ravel()[:, None]
    Q = W @ P                                                       # (pairs, nloc)
    shift = ((lr - math.log(rho_ref)) - 0.5 * mu_c * (ld - lr)).ravel() / math.log(10.0)
    k = np.floor(shift / _SRC_DX).astype(np.int64)
    f = shift / _SRC_DX - k
    out = np.zeros(_SRC_N_X + nloc + 2)          # padded, overflow checked below
    base = i_lo + k
    if base.min() < 0 or base.max() + nloc + 1 > _SRC_N_X:
        raise SRCUnsupported("SRC grid overflow in the lpi2007 target pmf")
    for j in range(Q.shape[0]):
        o = base[j]
        out[o: o + nloc] += (1.0 - f[j]) * Q[j]
        out[o + 1: o + 1 + nloc] += f[j] * Q[j]
    return out[:_SRC_N_X]


# Strength-scale symmetry of the point-source law (verified numerically to 1e-14 for every package and class):
# Y -> lam Y with m -> lam^(3(2+mu)/2) m keeps pi2 / st fixed and scales D by lam, hence at the SRC reference state
#   x0(s, psi) = x0(s - l, psi_ref) + 1.5 (2 + mu) l,   slope(s, psi) = slope(s - l, psi_ref),
#   l = log10(psi / psi_ref) [- 0.25 max(s + 1, 0) for the size-dependent IMPACTS rock strength].
# The (s, psi) surface is therefore one 1-D curve, solved on a dense tau grid and interpolated linearly
# (2048 nodes: interpolation error ~1e-6 dex). USE_SCALE_SYMMETRY = False restores the 2-D solve.
USE_SCALE_SYMMETRY = True
N_TAU = 2048
_PSI_REF = 1.0e5


def _x0_isl_surface(s_all, psi_grid, c, k1a, k2a, mua, kra, sda, rho_ref, g, density_ratio, rim_factor):
    """x0(s, psi) and 1 / slope (with the size-dependent-strength Jacobian) on the (s_all x psi_grid) grid."""
    mu_c, sd = float(mua[c]), bool(sda[c])
    S, PSI = np.meshgrid(s_all, psi_grid, indexing="ij")
    if not USE_SCALE_SYMMETRY:
        n = S.size
        x0, sl, pi2, st = solve_log10_m(S.ravel(), u_m_s=np.ones(n), delta=np.full(n, rho_ref), rho_t=np.full(n, rho_ref),
                                        y_eff=effective_strength(PSI.ravel(), np.power(10.0, S.ravel()), sd),
                                        k1=np.full(n, k1a[c]), k2=np.full(n, k2a[c]), mu=np.full(n, mu_c),
                                        kr=np.full(n, kra[c]), g=g, density_ratio=density_ratio, nu=np.full(n, 1.0 / 3.0),
                                        elev_deg=np.full(n, 90.0), angle_q=np.ones(n), rim_factor=rim_factor)
        if sd:
            sl = sl / (1.0 - 0.125 * mu_c * (st / (pi2 + st)))
        return x0.reshape(S.shape), (1.0 / sl).reshape(S.shape)
    ell = np.log10(PSI / _PSI_REF) - (0.25 * np.maximum(S + 1.0, 0.0) if sd else 0.0)
    tau = S - ell
    tg = np.linspace(float(tau.min()) - 1e-3, float(tau.max()) + 1e-3, N_TAU)
    n = N_TAU
    x1, sl1, pi21, st1 = solve_log10_m(tg, u_m_s=np.ones(n), delta=np.full(n, rho_ref), rho_t=np.full(n, rho_ref),
                                       y_eff=np.full(n, _PSI_REF), k1=np.full(n, k1a[c]), k2=np.full(n, k2a[c]),
                                       mu=np.full(n, mu_c), kr=np.full(n, kra[c]), g=g, density_ratio=density_ratio,
                                       nu=np.full(n, 1.0 / 3.0), elev_deg=np.full(n, 90.0), angle_q=np.ones(n),
                                       rim_factor=rim_factor)
    if sd:
        sl1 = sl1 / (1.0 - 0.125 * mu_c * (st1 / (pi21 + st1)))
    x0 = np.interp(tau.ravel(), tg, x1).reshape(S.shape) + 1.5 * (2.0 + mu_c) * ell
    isl = np.interp(tau.ravel(), tg, 1.0 / sl1).reshape(S.shape)
    return x0, isl


def _log_laplace(pmf: np.ndarray, lo: float, alpha: float) -> float:
    """log sum_i pmf_i exp(-alpha (lo + i dx)) for a pmf on a uniform grid (stable; -inf if empty)."""
    nz = np.flatnonzero(pmf)
    if nz.size == 0:
        return -math.inf
    e = -alpha * (lo + nz * _SRC_DX)
    m = float(e.max())
    return m + math.log(float(np.sum(pmf[nz] * np.exp(e - m))))


def _target_branch_weights(d_obs_m: float, site: SiteContext, ev: EventData) -> dict[tuple[int, bool], float]:
    """Exact 3-interval Gauss-Legendre quadrature over log(h) split at D/20 and D/4 for (cls, is_layered)."""
    if getattr(ev, "rocky_ejecta", None) is not None and site.fixed_class is not None:
        raise ValueError("rocky_ejecta cannot be used with a fixed target class")
    out = {(c, lay): 0.0 for c in (REGOLITH, FRACTURED, COMPETENT) for lay in (False, True)}
    if site.fixed_class is not None:
        out[(int(site.fixed_class), False)] = 1.0
        return out
    units = (
        [
            (True, site.p_mare, site.h_by_unit[0][0], site.h_by_unit[0][1]),
            (False, 1.0 - site.p_mare, site.h_by_unit[1][0], site.h_by_unit[1][1]),
        ]
        if site.h_by_unit is not None
        else [
            (True, site.p_mare, site.h_median_m, site.h_sigma_ln),
            (False, 1.0 - site.p_mare, site.h_median_m, site.h_sigma_ln),
        ]
    )
    h_k1, h_k2 = d_obs_m / 20.0, d_obs_m / 4.0
    for is_mare, p_unit, h_med, h_sig in units:
        if p_unit <= 0.0:
            continue
        sub_cls = COMPETENT if is_mare else FRACTURED
        z1 = math.log(max(h_k1, 1e-12) / h_med) / h_sig
        z2 = math.log(max(h_k2, 1e-12) / h_med) / h_sig
        knots = sorted(list({-8.0, min(max(z1, -8.0), 8.0), min(max(z2, -8.0), 8.0), 8.0}))
        for za, zb in zip(knots[:-1], knots[1:]):
            if zb <= za + 1e-12:
                continue
            z_seg = za + (zb - za) * _U20
            wz_seg = (zb - za) * _W20 * np.exp(-0.5 * z_seg**2) * INV_SQRT_2PI
            h_seg = h_med * np.exp(h_sig * z_seg)
            if getattr(ev, "rocky_ejecta", None) is not None:
                b1 = 2.22 if is_mare else 1.01
                zr = b1 * (0.084 * d_obs_m - h_seg)
                wz_seg = wz_seg * np.exp(-np.logaddexp(0.0, -zr if ev.rocky_ejecta else zr))
            p_reg = regolith_probability(d_obs_m / h_seg)
            z_mid = 0.5 * (za + zb)
            lay = (z_mid > z1) and (z_mid < z2)
            out[(REGOLITH, lay)] += p_unit * float(np.sum(wz_seg * p_reg))
            out[(sub_cls, lay)] += p_unit * float(np.sum(wz_seg * (1.0 - p_reg)))
    return out



def _class_coupling_probability(d_obs_m: float, sigma_meas_dex: float, site: SiteContext) -> float:
    """Prior probability that the regolith thickness puts this crater, or the other-class crater of the same
    impactor, near the layered band (D/h within a factor F = 10^(1 + 5 sigma_meas) of 4-20). Only outside it does the
    coherent class rule of sampler.run_scenario reduce to the uncoupled rule SRC implements (P_reg is 0 or 1 for every
    size that matters)."""
    from . import sampler
    if not sampler.COHERENT_CLASS_MODEL or site.fixed_class is not None:
        return 0.0
    f = 10.0 ** (1.0 + 5.0 * sigma_meas_dex)
    lo, hi = math.log(d_obs_m / (DH_BEDROCK_MIN * f)), math.log(d_obs_m * f / DH_REGOLITH_MAX)
    units = (((site.p_mare,) + site.h_by_unit[0]), ((1.0 - site.p_mare,) + site.h_by_unit[1])) \
        if site.h_by_unit is not None else ((1.0, site.h_median_m, site.h_sigma_ln),)
    return float(sum(w * (ndtr((hi - math.log(m)) / sd) - ndtr((lo - math.log(m)) / sd)) for w, m, sd in units))


def run_scenario_src(
    d_obs_m: float,
    sigma_meas_dex: float = 0.03,
    population: str = "asteroidal",
    package: str = "holsapple2003",
    *,
    site: SiteContext = GLOBAL_CONTEXT,
    model: ModelForm = ModelForm(),
    event: EventData | None = None,
    gravity_m_s2: float = 1.62,
) -> dict:
    """Deterministic posterior of log10 mass for one scenario (same model as sampler.run_scenario).

    Returns q05/median/q95 (midpoint CDF), log_evidence (same normalisation as run_scenario), class_probs,
    grid_x and posterior_pmf; log_evidence_d_only (diameter-only evidence from the same pass, or None);
    grid_energy/energy_pmf and grid_momentum/momentum_pmf (log10 kinetic energy and momentum). Raises
    SRCUnsupported for inputs it cannot represent exactly (luminous-efficiency draws from a table; craters whose
    target class is coupled to the impactor near the layered band, see _class_coupling_probability)."""
    ev = event or EventData()
    validate_event(ev)
    if ev.eta_draws is not None:
        raise SRCUnsupported("luminous-efficiency draws (eta_draws) are not supported by the SRC backend")
    if not (d_obs_m > 0) or not math.isfinite(d_obs_m):
        raise ValueError("crater diameter must be positive and finite")
    if _class_coupling_probability(d_obs_m, sigma_meas_dex, site) > 1e-9:
        raise SRCUnsupported("the coherent target-class rule couples the regolith and substrate branches "
                             "(sampler.COHERENT_CLASS_MODEL); SRC implements only the uncoupled rule")
    pop = POPULATIONS[population]
    pkg = PACKAGES[package]
    k1a, k2a, mua, kra, sda = pkg.arrays()
    b = pop.mass_prior_b if model.mass_prior_b is None else model.mass_prior_b
    y_obs = math.log10(d_obs_m)

    branch_w = _target_branch_weights(d_obs_m, site, ev)

    # 1. Velocity quadrature
    if ev.velocity_m_s is not None:
        # ln U over the measurement density restricted to the speed support, so that a measured speed several
        # sigma outside the support still gets nodes where the prior is positive
        su = math.log(1.0 + ev.velocity_sigma_rel)
        lnu_n, w_meas = _observed_nodes(math.log(ev.velocity_m_s), su, math.log(pop.u_lo), math.log(pop.u_hi))
        u_raw = np.clip(np.exp(lnu_n), pop.u_lo, pop.u_hi)
        lp_u = pop.logpdf_u(u_raw) + np.log(u_raw)
        valid_u = np.isfinite(lp_u) & (w_meas > 0)
        u_nodes = u_raw[valid_u]
        w_u = w_meas[valid_u] * np.exp(lp_u[valid_u])
    else:
        ln_u_lo, ln_u_hi = math.log(pop.u_lo), math.log(pop.u_hi)
        ln_u_nodes = ln_u_lo + (ln_u_hi - ln_u_lo) * _U48
        u_nodes = np.exp(ln_u_nodes)
        if pop.u_sigma_ln > 0:
            z_u = (ln_u_nodes - math.log(pop.u_median)) / pop.u_sigma_ln
            z_lo, z_hi = pop._z(pop.u_lo), pop._z(pop.u_hi)
            w_u = (
                (ln_u_hi - ln_u_lo)
                * _W48
                * np.exp(-0.5 * z_u**2)
                * INV_SQRT_2PI
                / (pop.u_sigma_ln * (ndtr(z_hi) - ndtr(z_lo)))
            )
        else:
            w_u = _W48.copy()
    u_log10 = np.log10(u_nodes)

    # 2. Elevation & angle-exponent q quadrature
    if ev.elev_deg is not None:
        el_raw, w_norm = _observed_nodes(ev.elev_deg, ev.elev_sigma_deg, max(pop.elev_lo, 0.001),
                                         min(pop.elev_hi, 89.999))
        lp_el = pop.logpdf_elev(el_raw)
        valid_el = np.isfinite(lp_el) & (w_norm > 0)
        el_nodes = np.clip(el_raw[valid_el], 0.5, 90.0)
        w_el = w_norm[valid_el] * np.exp(lp_el[valid_el])
    else:
        if pop.angle_law == "sin2":
            el_nodes = np.degrees(np.arcsin(_U48**2))
            w_el = 4.0 * (_U48**3) * _W48
        else:
            el_nodes = pop.elev_lo + (pop.elev_hi - pop.elev_lo) * _U48
            w_el = _W48.copy()
    if ev.ejecta_pattern is not None:
        w_el = w_el * np.exp(ejecta_pattern_loglik(ev.ejecta_pattern, el_nodes))

    q_lo, q_hi = model.angle_q
    if q_hi > q_lo:
        q_nodes, w_q = q_lo + (q_hi - q_lo) * _U12, _W12
    else:
        q_nodes, w_q = np.array([q_lo]), np.array([1.0])
    log10_sin_el = np.log10(np.maximum(np.sin(np.radians(el_nodes)), 1e-6))
    lq_vals = (log10_sin_el[:, None] * q_nodes[None, :]).ravel()
    lq_weights = (w_el[:, None] * w_q[None, :]).ravel()

    # 3. Impactor density delta & exponent nu quadrature
    if ev.delta_kg_m3 is not None:
        # a known density is weighted by the population's prior density there, as in the sampler
        w_known = float(np.exp(pop.logpdf_delta(float(ev.delta_kg_m3))))
        if not w_known > 0:
            raise SRCUnsupported("the known impactor density lies outside the population's density prior")
        delta_nodes, w_delta = np.array([float(ev.delta_kg_m3)]), np.array([w_known])
    else:
        ln_d_lo, ln_d_hi = math.log(pop.delta_lo), math.log(pop.delta_hi)
        d_stony = np.exp(ln_d_lo + (ln_d_hi - ln_d_lo) * _U16)
        w_stony = (1.0 - pop.iron_fraction) * _W16
        if pop.iron_fraction > 0:
            d_iron = 7000.0 + 900.0 * _U6
            w_iron = pop.iron_fraction * _W6
            delta_nodes = np.concatenate([d_stony, d_iron])
            w_delta = np.concatenate([w_stony, w_iron])
        else:
            delta_nodes, w_delta = d_stony, w_stony
    log10_delta = np.log10(delta_nodes)

    nu_lo, nu_hi = model.nu
    if pkg.density_ratio and nu_hi > nu_lo:
        nu_nodes, w_nu = nu_lo + (nu_hi - nu_lo) * _U10, _W10
    else:
        nu_nodes, w_nu = np.array([0.5 * (nu_lo + nu_hi)]), np.array([1.0])

    # Diameter-only evidence in the same pass. With the mass prior exp(-alpha log10 m), alpha = b ln 10, the
    # Laplace transform of the convolution target (*) angle (*) density (*) speed is the product of the factors'
    # transforms, so log Z(D only) needs only the prior
    # kinematic kernels, not a second SRC pass. Valid when the event leaves the target block unchanged: not with
    # rocky ejecta (target branch weights) or a known density under lpi2007 (delta enters psi). The mass-support
    # mask is not applied in the transform (mass there is < 1e-6 by the grid-overflow guard).
    alpha = b * math.log(10.0)
    d_only = (event is not None and ev != EventData() and ev.rocky_ejecta is None
              and not (ev.delta_kg_m3 is not None and not pkg.density_ratio))
    if d_only:
        ln_u_lo0, ln_u_hi0 = math.log(pop.u_lo), math.log(pop.u_hi)
        lnu0 = ln_u_lo0 + (ln_u_hi0 - ln_u_lo0) * _U48
        if pop.u_sigma_ln > 0:
            zu0 = (lnu0 - math.log(pop.u_median)) / pop.u_sigma_ln
            w_u0 = ((ln_u_hi0 - ln_u_lo0) * _W48 * np.exp(-0.5 * zu0**2) * INV_SQRT_2PI
                    / (pop.u_sigma_ln * (ndtr(pop._z(pop.u_hi)) - ndtr(pop._z(pop.u_lo)))))
        else:
            w_u0 = _W48.copy()
        u0_log10 = lnu0 / math.log(10.0)
        if pop.angle_law == "sin2":
            el0, w_el0 = np.degrees(np.arcsin(_U48**2)), 4.0 * (_U48**3) * _W48
        else:
            el0, w_el0 = pop.elev_lo + (pop.elev_hi - pop.elev_lo) * _U48, _W48.copy()
        lq0 = (np.log10(np.maximum(np.sin(np.radians(el0)), 1e-6))[:, None] * q_nodes[None, :]).ravel()
        lqw0 = (w_el0[:, None] * w_q[None, :]).ravel()
        ln_d_lo0, ln_d_hi0 = math.log(pop.delta_lo), math.log(pop.delta_hi)
        d0 = np.exp(ln_d_lo0 + (ln_d_hi0 - ln_d_lo0) * _U16)
        wd0 = (1.0 - pop.iron_fraction) * _W16
        if pop.iron_fraction > 0:
            d0, wd0 = np.concatenate([d0, 7000.0 + 900.0 * _U6]), np.concatenate([wd0, pop.iron_fraction * _W6])
        log10_delta0 = np.log10(d0)
    log_zd_classes = []

    # 4. Crater size s quadrature for (layered=False, layered=True)
    layered_sig = LAYERED_EXTRA_SIGMA_DEX if model.layered_extra_sigma_dex is None else model.layered_extra_sigma_dex
    if model.sigma_mf_draws is not None:
        smf_arr = np.asarray(model.sigma_mf_draws, dtype=np.float64)
        smf_nodes, w_smf = smf_arr, np.full(len(smf_arr), 1.0 / len(smf_arr))
    elif model.sigma_mf_range is not None:
        # as sampler.draw_sigma_mf: uniform over the range; a degenerate range (a, a) is the fixed value a
        r_lo, r_hi = model.sigma_mf_range
        if r_hi > r_lo:
            smf_nodes, w_smf = r_lo + (r_hi - r_lo) * _U8, _W8
        else:
            smf_nodes, w_smf = np.array([float(r_lo)]), np.array([1.0])
    else:
        smf_nodes, w_smf = np.array([model.sigma_mf_dex]), np.array([1.0])

    lo_m, hi_m = mass_support(d_obs_m)
    s_by_lay = {}
    for lay in (False, True):
        sig_tot = np.sqrt(sigma_meas_dex**2 + smf_nodes**2 + (layered_sig**2 if lay else 0.0))
        z_lo = _s_window_below(y_obs, float(np.max(sig_tot)), b, lo_m)
        if sig_tot.size > 1:
            # A sigma_mf range or table makes the crater-size density a Gaussian scale mixture; one shared node
            # set (exact bin masses and bin means) replaces len(sigma) x N_S nodes
            s_by_lay[lay] = _mixture_nodes(y_obs, sig_tot, w_smf, N_S_MIX_BINS, z_lo=max(6.0, z_lo))
            continue
        zs, ws_ = _normal_nodes(N_S_NODES, z_lo)
        s_grid = (y_obs + sig_tot[:, None] * zs[None, :]).ravel()
        w_s_grid = (w_smf[:, None] * ws_[None, :]).ravel()
        s_by_lay[lay] = (s_grid, w_s_grid)

    # 5. Per-class 2D-Buckingham-Pi evaluation and 1D-Z convolution
    class_pmfs = {}
    kin = {}   # class -> (log10 m at the joint (z, U) points, their weights before the mass prior)
    for c in (REGOLITH, FRACTURED, COMPETENT):
        w_nolay = branch_w[(c, False)]
        w_lay = branch_w[(c, True)]
        if w_nolay + w_lay <= 1e-15:
            class_pmfs[c] = np.zeros(_SRC_N_X, dtype=np.float64)
            continue

        mu_c = float(mua[c])
        pr = CLASS_PRIORS[c]
        ln_y_lo, ln_y_hi = math.log(pr.y_lo), math.log(pr.y_hi)

        s_parts, ws_parts = [], []
        if smf_nodes.size > 1 and w_nolay > 1e-15 and w_lay > 1e-15:
            # one shared mixture grid for both branches (2K-component scale mixture) instead of concatenating two
            # N_S_MIX_BINS grids
            sig0 = np.sqrt(sigma_meas_dex**2 + smf_nodes**2)
            sig1 = np.sqrt(sigma_meas_dex**2 + smf_nodes**2 + layered_sig**2)
            s_c, w_c = _mixture_nodes(y_obs, np.concatenate([sig0, sig1]),
                                      np.concatenate([w_nolay * w_smf, w_lay * w_smf]), N_S_MIX_BINS,
                                      z_lo=max(6.0, _s_window_below(y_obs, float(np.max(sig1)), b, lo_m)))
            s_parts, ws_parts = [s_c], [w_c * (w_nolay + w_lay)]
        elif w_nolay > 1e-15:
            s0, ws0 = s_by_lay[False]
            s_parts.append(s0)
            ws_parts.append(w_nolay * ws0)
        if w_lay > 1e-15 and not (smf_nodes.size > 1 and w_nolay > 1e-15):
            s1, ws1 = s_by_lay[True]
            s_parts.append(s1)
            ws_parts.append(w_lay * ws1)
        s_all = np.concatenate(s_parts)
        ws_all = np.concatenate(ws_parts)

        rho_ref = math.sqrt(pr.rho_lo * pr.rho_hi)
        s_theta_vals = -3.0 * mu_c * lq_vals
        pmf_theta = _splat_1d(s_theta_vals, lq_weights, _SRC_B_LO, _SRC_N_B, _SRC_DX)

        if pkg.density_ratio:
            rho_nodes = pr.rho_lo + (pr.rho_hi - pr.rho_lo) * _U12
            w_rho = _W12
            log10_rho_ref = math.log10(rho_ref)

            ln_psi_lo = math.log(pr.y_lo * (rho_ref / pr.rho_hi))
            ln_psi_hi = math.log(pr.y_hi * (rho_ref / pr.rho_lo))
            n_psi = 40
            ln_psi_grid = np.linspace(ln_psi_lo, ln_psi_hi, n_psi)
            d_ln_psi = ln_psi_grid[1] - ln_psi_grid[0]
            psi_grid = np.exp(ln_psi_grid)

            x0_2d, isl_2d = _x0_isl_surface(s_all, psi_grid, c, k1a, k2a, mua, kra, sda, rho_ref, gravity_m_s2, True,
                                            pkg.rim_factor)

            uy, wy = _gl01(N_Y_NODES)
            ln_r_2d, ln_y_2d = np.meshgrid(np.log(rho_nodes), ln_y_lo + (ln_y_hi - ln_y_lo) * uy, indexing="ij")
            w_ry = (w_rho[:, None] * wy[None, :]).ravel()
            ln_psi_ry = (ln_y_2d - ln_r_2d + math.log(rho_ref)).ravel()
            pos_p = np.clip((ln_psi_ry - ln_psi_lo) / d_ln_psi, 0.0, n_psi - 1.000001)
            idx_p = np.floor(pos_p).astype(np.int64)
            frac_p = pos_p - idx_p
            x_base = (1.0 - frac_p)[None, :] * x0_2d[:, idx_p] + frac_p[None, :] * x0_2d[:, idx_p + 1]
            isl_interp = (1.0 - frac_p)[None, :] * isl_2d[:, idx_p] + frac_p[None, :] * isl_2d[:, idx_p + 1]
            wx_full = (ws_all[:, None] * w_ry[None, :] * isl_interp).ravel()
            # log10 m = x(s, psi) + 3 nu log10(rho_t / rho_ref) + (1 - 3 nu) log10(delta / rho_ref) + angle/speed terms:
            # nu couples the target-density and impactor-density shifts, so it is integrated node by node.
            # Grouped by target-density node k, the nu-dependent part of the shift is
            # 3 nu l10_r_k + (1 - 3 nu) S = S + 3 nu (l10_r_k - S), S = log10(delta / rho_ref), so it folds into one
            # (nu, delta) kernel per k and x is splatted once per k instead of once per nu node
            n_y = len(wy)
            l10_r_nodes = (np.log(rho_nodes) - math.log(rho_ref)) / math.log(10.0)
            s_d = log10_delta - log10_rho_ref
            w_nd = (w_nu[:, None] * w_delta[None, :]).ravel()
            x_blocks = x_base.reshape(len(s_all), len(rho_nodes), n_y)
            w_blocks = wx_full.reshape(len(s_all), len(rho_nodes), n_y)
            pmf_z = np.zeros(_SRC_N_X)
            lz_k = []
            if d_only:
                s_d0 = log10_delta0 - log10_rho_ref
                w_nd0 = (w_nu[:, None] * wd0[None, :]).ravel()
            for k in range(len(rho_nodes)):
                pmf_x = _splat_1d(x_blocks[:, k, :].ravel(), w_blocks[:, k, :].ravel(), _SRC_X_LO, _SRC_N_X, _SRC_DX)
                kv = (s_d[None, :] + 3.0 * nu_nodes[:, None] * (l10_r_nodes[k] - s_d[None, :])).ravel()
                pmf_k = _splat_1d(kv, w_nd, _SRC_B_LO, _SRC_N_B, _SRC_DX)
                pmf_sd = _lconv(pmf_theta, pmf_k)[_SRC_ZERO_B_IDX: _SRC_ZERO_B_IDX + _SRC_N_B]
                pmf_z += _lconv(pmf_x, pmf_sd)[_SRC_ZERO_B_IDX: _SRC_ZERO_B_IDX + _SRC_N_X]
                if d_only:
                    kv0 = (s_d0[None, :] + 3.0 * nu_nodes[:, None] * (l10_r_nodes[k] - s_d0[None, :])).ravel()
                    lz_k.append(_log_laplace(pmf_x, _SRC_X_LO, alpha)
                                + _log_laplace(_splat_1d(kv0, w_nd0, _SRC_B_LO, _SRC_N_B, _SRC_DX), _SRC_B_LO, alpha))
            if d_only:
                log_lx_lk = float(np.logaddexp.reduce(lz_k))
        else:
            rho_nodes = pr.rho_lo + (pr.rho_hi - pr.rho_lo) * _U8
            w_rho = _W8
            d_min, d_max = float(np.min(delta_nodes)), float(np.max(delta_nodes))
            ln_psi_lo = math.log(pr.y_lo * ((d_min / pr.rho_hi) ** (1.0 / 3.0)) * (rho_ref / pr.rho_hi))
            ln_psi_hi = math.log(pr.y_hi * ((d_max / pr.rho_lo) ** (1.0 / 3.0)) * (rho_ref / pr.rho_lo))
            if LPI_WINDOW_METHOD:
                pmf_x = _lpi_target_pmf(s_all, ws_all, rho_nodes, w_rho, delta_nodes, w_delta, ln_y_lo, ln_y_hi,
                                        rho_ref, ln_psi_lo, ln_psi_hi, mu_c, k1a[c], k2a[c], kra[c], gravity_m_s2,
                                        pkg.rim_factor)
                pmf_sd = pmf_theta
            else:   # 3-D (rho, Y, delta) quadrature, kept for comparison (LPI_WINDOW_METHOD = False)
                n_psi = 48
                ln_psi_grid = np.linspace(ln_psi_lo, ln_psi_hi, n_psi)
                d_ln_psi = ln_psi_grid[1] - ln_psi_grid[0]
                psi_grid = np.exp(ln_psi_grid)

                S_2d, PSI_2d = np.meshgrid(s_all, psi_grid, indexing="ij")
                s_f, psi_f = S_2d.ravel(), PSI_2d.ravel()
                n_pts = len(s_f)
                x0_f, sl_f, _, _ = solve_log10_m(
                    s_f,
                    u_m_s=np.ones(n_pts),
                    delta=np.full(n_pts, rho_ref),
                    rho_t=np.full(n_pts, rho_ref),
                    y_eff=psi_f,
                    k1=np.full(n_pts, k1a[c]),
                    k2=np.full(n_pts, k2a[c]),
                    mu=np.full(n_pts, mu_c),
                    kr=np.full(n_pts, kra[c]),
                    g=gravity_m_s2,
                    density_ratio=False,
                    nu=np.full(n_pts, 1.0 / 3.0),
                    elev_deg=np.full(n_pts, 90.0),
                    angle_q=np.ones(n_pts),
                    rim_factor=pkg.rim_factor,
                )
                x0_2d = x0_f.reshape(len(s_all), n_psi)
                isl_2d = (1.0 / sl_f).reshape(len(s_all), n_psi)

                ln_r_3d, ln_y_3d, ln_d_3d = np.meshgrid(
                    np.log(rho_nodes),
                    ln_y_lo + (ln_y_hi - ln_y_lo) * _gl01(N_Y_NODES_LPI)[0],
                    np.log(delta_nodes),
                    indexing="ij",
                )
                w_ryd = (w_rho[:, None, None] * _gl01(N_Y_NODES_LPI)[1][None, :, None] * w_delta[None, None, :]).ravel()
                ln_psi_ryd = (ln_y_3d + (ln_d_3d - ln_r_3d) / 3.0 + math.log(rho_ref) - ln_r_3d).ravel()
                pos_p = np.clip((ln_psi_ryd - ln_psi_lo) / d_ln_psi, 0.0, n_psi - 1.000001)
                idx_p = np.floor(pos_p).astype(np.int64)
                frac_p = pos_p - idx_p
                shift_rd = ((ln_r_3d.ravel() - math.log(rho_ref)) - 0.5 * mu_c * (ln_d_3d.ravel() - ln_r_3d.ravel())) / math.log(10.0)
                pmf_sd = pmf_theta
                x_interp = (1.0 - frac_p)[None, :] * x0_2d[:, idx_p] + frac_p[None, :] * x0_2d[:, idx_p + 1] + shift_rd[None, :]
                isl_interp = (1.0 - frac_p)[None, :] * isl_2d[:, idx_p] + frac_p[None, :] * isl_2d[:, idx_p + 1]
                wx_full = ws_all[:, None] * w_ryd[None, :] * isl_interp
                pmf_x = _splat_1d(x_interp.ravel(), wx_full.ravel(), _SRC_X_LO, _SRC_N_X, _SRC_DX)

            pmf_z = _lconv(pmf_x, pmf_sd)[_SRC_ZERO_B_IDX: _SRC_ZERO_B_IDX + _SRC_N_X]
            if d_only:
                log_lx_lk = _log_laplace(pmf_x, _SRC_X_LO, alpha)

        if d_only:   # prior angle and prior speed factors of the diameter-only transform
            log_zd_classes.append(
                log_lx_lk
                + _log_laplace(_splat_1d(-3.0 * mu_c * lq0, lqw0, _SRC_B_LO, _SRC_N_B, _SRC_DX), _SRC_B_LO, alpha)
                + _log_laplace(_splat_1d(-3.0 * mu_c * u0_log10, w_u0, _SRC_B_LO, _SRC_N_B, _SRC_DX), _SRC_B_LO, alpha))

        if ev.flash_energy_j is None:
            pmf_u = _splat_1d(-3.0 * mu_c * u_log10, w_u, _SRC_B_LO, _SRC_N_B, _SRC_DX)
            pmf_m_full = _lconv(pmf_z, pmf_u)
            class_pmfs[c] = pmf_m_full[_SRC_ZERO_B_IDX : _SRC_ZERO_B_IDX + _SRC_N_X]
            # joint (z, U) points for the energy and momentum posteriors (log10 m = z - 3 mu log10 U)
            nz_idx = np.flatnonzero(pmf_z > 1e-18 * np.max(pmf_z))
            m_mat = _SRC_GRID_X[nz_idx][:, None] - 3.0 * mu_c * u_log10[None, :]
            kin[c] = (m_mat, pmf_z[nz_idx][:, None] * w_u[None, :])
        else:
            if ev.flash_definition not in FLASH_DEFINITIONS:
                raise ValueError(f"unknown flash definition {ev.flash_definition}")
            f_lo, f_hi = FLASH_DEFINITIONS[ev.flash_definition]["log10_factor"]
            nz_idx = np.flatnonzero(pmf_z > 1e-18 * np.max(pmf_z))
            z_sub = _SRC_GRID_X[nz_idx]
            wz_sub = pmf_z[nz_idx]
            m_mat = z_sub[:, None] - 3.0 * mu_c * u_log10[None, :]
            lek_mat = z_sub[:, None] + (2.0 - 3.0 * mu_c) * u_log10[None, :] - math.log10(2.0)
            ll_f = analytic_flash_loglik(
                math.log10(ev.flash_energy_j),
                ev.flash_sigma_dex,
                lek_mat.ravel(),
                eta_range=ev.eta_range,
                f_range=(f_lo, f_hi),
            ).reshape(m_mat.shape)
            w_mat = wz_sub[:, None] * w_u[None, :] * np.exp(ll_f)
            class_pmfs[c] = _splat_1d(m_mat.ravel(), w_mat.ravel(), _SRC_X_LO, _SRC_N_X, _SRC_DX)
            kin[c] = (m_mat, w_mat)

    def log_prior(m):   # mass prior (+ mass observation) on log10 m, -inf outside the support
        lw = np.where((m >= lo_m) & (m <= hi_m), -b * math.log(10.0) * m, -np.inf)
        if ev.mass_kg is not None:
            lw = lw - 0.5 * ((m - math.log10(ev.mass_kg)) / ev.mass_sigma_dex) ** 2
        return lw

    ins = (_SRC_GRID_X >= lo_m) & (_SRC_GRID_X <= hi_m)
    lw_grid = log_prior(_SRC_GRID_X)
    mx = float(np.max(lw_grid[ins]))
    prior_mult = np.where(ins, np.exp(lw_grid - mx), 0.0)

    # deterministic posteriors of log10 kinetic energy (m + 2 log10 U - log10 2) and log10 momentum (m + log10 U)
    pmf_e, pmf_p = np.zeros(_SRC_N_E), np.zeros(_SRC_N_P)
    for c, (m_mat, w_mat) in kin.items():
        if w_mat.size == 0:
            continue
        wp = w_mat * np.exp(log_prior(m_mat) - mx)
        lu = u_log10
        pmf_e += _splat_1d((m_mat + 2.0 * lu[None, :] - math.log10(2.0)).ravel(), wp.ravel(), _SRC_E_LO, _SRC_N_E, _SRC_DX)
        pmf_p += _splat_1d((m_mat + lu[None, :]).ravel(), wp.ravel(), _SRC_P_LO, _SRC_N_P, _SRC_DX)

    for c in (REGOLITH, FRACTURED, COMPETENT):
        class_pmfs[c] = class_pmfs[c] * prior_mult
    tot_pmf = class_pmfs[REGOLITH] + class_pmfs[FRACTURED] + class_pmfs[COMPETENT]
    tot_sum = float(tot_pmf.sum())

    if b == 0:
        log_norm = -math.log((hi_m - lo_m) * math.log(10.0))
    else:
        bl = b * math.log(10.0)
        log_norm = math.log(b) + bl * lo_m - math.log(-math.expm1(-bl * (hi_m - lo_m)))
    if not (tot_sum > 0 and math.isfinite(tot_sum)):
        # observations outside every prior support (e.g. 18 km/s for the artificial population) leave no mass;
        # the Monte Carlo path reports such inputs as incompatible
        raise SRCUnsupported("SRC found no posterior mass: the observations are outside the prior support")
    log_ev = float(mx + math.log(tot_sum) + log_norm + math.log(math.log(10.0)))

    # Midpoint CDF eliminates the -0.5 * DX = -0.0025 dex left-edge bias
    cdf_mid = (np.cumsum(tot_pmf) - 0.5 * tot_pmf) / tot_sum
    q05, q50, q95 = [float(np.interp(q_val, cdf_mid, _SRC_GRID_X)) for q_val in (0.05, 0.50, 0.95)]
    cls_probs = [float(class_pmfs[c].sum() / tot_sum) for c in (REGOLITH, FRACTURED, COMPETENT)]
    return dict(
        q05=q05,
        median=q50,
        q95=q95,
        log_evidence=log_ev,
        class_probs=cls_probs,
        grid_x=_SRC_GRID_X,
        posterior_pmf=tot_pmf / tot_sum,
        log_evidence_d_only=(float(np.logaddexp.reduce(log_zd_classes)) + log_norm + math.log(math.log(10.0))
                             if d_only and log_zd_classes else None),
        grid_energy=_SRC_GRID_E,
        energy_pmf=pmf_e / pmf_e.sum() if pmf_e.sum() > 0 else pmf_e,
        grid_momentum=_SRC_GRID_P,
        momentum_pmf=pmf_p / pmf_p.sum() if pmf_p.sum() > 0 else pmf_p,
    )

