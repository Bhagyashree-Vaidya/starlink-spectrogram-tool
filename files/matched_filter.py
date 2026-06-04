"""
Matched-filter faint-track recovery  (Phase 3)
==============================================
For: FunLab, Prof. Sumit Roy & Jesse Chiu, UW ECE
By: Bhagyashree Vaidya

The connected-component detector only finds tracks whose pixels cross an
amplitude threshold. A real Starlink pass that sits just under that threshold
is invisible to it even though, integrated along its whole S-curve, there is
plenty of energy.

There are two ways to use it:

1. Prediction-guided recovery (`recover_faint_predictions`, recommended).
   After the main detection+matching pass, walk along each *unmatched* TLE
   prediction curve and test whether faint energy is present. This answers
   "did the receiver catch this predicted pass, even below the detection
   threshold?" It can only fill gaps, never steal an existing match, and it is
   precise by construction because it only looks where a real orbit predicts a
   track.

2. Blind scan (`find_faint_tracks`). Slide an S-curve template bank over the
   whole spectrogram and pick peaks. Useful for exploration when no prediction
   is available, but it has the usual recall/precision trade-off and needs
   tuning against real captures.

The S-curve

    f(t) = f0 + (Δf / 2) * tanh((t - t0) / τ)

is a fixed *shape* for a given (Δf, τ); changing f0 / t0 just translates it in
frequency / time, so a 2-D cross-correlation (FFT-based) of the enhanced image
with each template gives the energy along every candidate curve.
"""

import numpy as np
from scipy.signal import fftconvolve

from scurve_extractor import scurve_model, fit_scurve, _curves_consistent


# ---------------------------------------------------------------------------
# Prediction-guided faint recovery  (recommended)
# ---------------------------------------------------------------------------
def recover_faint_predictions(enhanced, unmatched_predictions, leakage_mask=None,
                              freq_band=2, min_coverage=0.45,
                              min_contiguous=0.35, k_sigma=1.5):
    """
    For every prediction the detector/matcher missed, test whether faint energy
    is present along its predicted Doppler curve.

    Walks each prediction curve, takes the max enhanced power in a small
    frequency band around it, and measures how much of the curve carries signal
    (coverage) and the longest continuous lit arc (contiguity). A prediction is
    declared "recovered (faint)" when both clear their thresholds.

    Because it only inspects pixels a real TLE predicts, it cannot steal an
    existing match and cannot invent tracks where no orbit passes.

    Returns a list of dicts: prediction_label, coverage, contiguous, mean_excess,
    confidence, plus the sampled curve (time_bins/freq_bins) for display.
    """
    n_freq, n_time = enhanced.shape
    med = np.median(enhanced)
    mad = np.median(np.abs(enhanced - med)) * 1.4826 + 1e-12
    bg_level = med + k_sigma * mad

    recovered = []
    for pred in unmatched_predictions:
        t = np.round(pred["time_bins"]).astype(int)
        f = np.asarray(pred["freq_bins"], dtype=float)
        inb = (t >= 0) & (t < n_time) & (f >= 0) & (f < n_freq)
        t, f = t[inb], f[inb]
        if len(t) < 8:
            continue
        # skip the portion that lies inside the leakage band (no info there)
        if leakage_mask is not None:
            keep = ~leakage_mask[np.round(f).astype(int)]
            if keep.sum() < 8:
                continue
            t, f = t[keep], f[keep]

        cov, cont, excess = _curve_coverage(enhanced, t, f, bg_level,
                                            band=freq_band)
        if cov >= min_coverage and cont >= min_contiguous:
            recovered.append({
                "prediction_label": pred["label"],
                "coverage": cov,
                "contiguous": cont,
                "mean_excess": excess,
                "confidence": float(min(1.0, 0.5 * cov + 0.5 * cont)),
                "time_bins": t.astype(float),
                "freq_bins": f,
                "source": "matched_filter",
            })
    # strongest first
    recovered.sort(key=lambda r: r["confidence"], reverse=True)
    return recovered


# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------
def _build_template(df, tau, n_freq, n_time, line_width=1):
    """
    Rasterise one centred S-curve shape into a small mask.

    The curve passes through the mask centre (f0=0 at t0=0); correlating the
    mask with the image therefore reports, at output pixel (r, c), the energy
    along the curve with f0=r, t0=c.

    Returns (mask, center_row, center_col). Mask is normalised to unit sum so
    the correlation yields the *mean* power along the curve.
    """
    # window: integrate over roughly +/- 3 tau around the inflection
    half_w = int(min(n_time // 2, max(20, round(3 * tau))))
    t_rel = np.arange(-half_w, half_w + 1)
    f_rel = 0.5 * df * np.tanh(t_rel / tau)

    f_extent = int(np.ceil(np.max(np.abs(f_rel)))) + line_width + 1
    H = 2 * f_extent + 1
    W = len(t_rel)
    mask = np.zeros((H, W), dtype=float)
    cr = f_extent           # centre row
    for j, fr in enumerate(f_rel):
        r0 = cr + fr
        for dlw in range(-line_width, line_width + 1):
            r = int(round(r0 + dlw))
            if 0 <= r < H:
                mask[r, j] += np.exp(-0.5 * (dlw / max(line_width, 1)) ** 2)
    s = mask.sum()
    if s > 0:
        mask /= s
    return mask, cr, W // 2


def default_bank(n_freq, n_time):
    """A small spread of (Δf, τ) shapes covering shallow→steep LEO passes."""
    df_mag = np.array([0.15, 0.3, 0.5, 0.7]) * n_freq
    taus = np.array([0.04, 0.09, 0.18]) * n_time
    bank = []
    for tau in taus:
        for mag in df_mag:
            bank.append((+float(mag), float(tau)))   # rising pass
            bank.append((-float(mag), float(tau)))   # falling pass
    return bank


# ---------------------------------------------------------------------------
# Core matched filter
# ---------------------------------------------------------------------------
def matched_filter_response(enhanced, bank, line_width=1):
    """
    Cross-correlate the enhanced image with every template in the bank.

    Returns:
        best       : (n_freq, n_time) max mean-power response over all shapes
        best_shape : (n_freq, n_time) int index into `bank` of the winning shape
    """
    n_freq, n_time = enhanced.shape
    # Background-subtract: the enhanced image sits on a non-zero noise floor, so
    # the mean power along *any* curve is dominated by that DC level and the
    # track contrast is swamped. Remove the floor (clip negatives) so the
    # response reflects excess energy along a curve.
    e = enhanced - np.median(enhanced)
    e = np.clip(e, 0.0, None)

    best = np.full((n_freq, n_time), -np.inf, dtype=float)
    best_shape = np.zeros((n_freq, n_time), dtype=int)

    for k, (df, tau) in enumerate(bank):
        mask, _, _ = _build_template(df, tau, n_freq, n_time, line_width)
        # correlation = convolution with the flipped kernel
        resp = fftconvolve(e, mask[::-1, ::-1], mode="same")
        upd = resp > best
        best[upd] = resp[upd]
        best_shape[upd] = k
    return best, best_shape


# ---------------------------------------------------------------------------
# Peak picking with non-maximum suppression
# ---------------------------------------------------------------------------
def _robust_threshold(resp, k_sigma):
    med = np.median(resp)
    mad = np.median(np.abs(resp - med)) * 1.4826 + 1e-12
    return med + k_sigma * mad


def _curve_coverage(enhanced, t_fit, f_fit, bg_level, band=1):
    """
    Fraction of the candidate curve that actually carries signal.

    A genuine track lights up most of its curve; a curve fitted through
    scattered noise only clips a few bright pixels. For each curve sample we
    take the max enhanced value in a small frequency band around it and count
    how many exceed the background level.

    Returns (coverage_fraction, mean_excess).
    """
    n_freq, n_time = enhanced.shape
    rows = np.round(f_fit).astype(int)
    cols = np.round(t_fit).astype(int)
    vals = []
    for r, c in zip(rows, cols):
        r0, r1 = max(0, r - band), min(n_freq, r + band + 1)
        vals.append(enhanced[r0:r1, c].max())
    vals = np.asarray(vals)
    hits = vals > bg_level
    coverage = float(np.mean(hits))
    # longest contiguous covered run, as a fraction of the curve — a genuine
    # track lights up a continuous arc; noise only clips scattered pixels
    best_run = run = 0
    for h in hits:
        run = run + 1 if h else 0
        best_run = max(best_run, run)
    contiguous = float(best_run / max(len(hits), 1))
    excess = float(np.mean(np.clip(vals - bg_level, 0, None)))
    return coverage, contiguous, excess


def find_faint_tracks(enhanced, existing_curves=None, bank=None,
                      k_sigma=5.0, max_candidates=20,
                      nms_freq=8, nms_time=25, line_width=1,
                      consistency_tol_px=3.0, min_coverage=0.6,
                      min_contiguous=0.45, leakage_mask=None):
    """
    Recover faint S-curve tracks the blob detector missed.

    Parameters
    ----------
    enhanced : 2-D array
        Background-subtracted, normalised spectrogram (same image the detector
        thresholds).
    existing_curves : list of curve dicts, optional
        Already-found trajectories ({time_bins, freq_bins, params,...}); any
        matched-filter candidate consistent with one of these is dropped as a
        duplicate.
    k_sigma : float
        Detection sensitivity. Higher = stricter (fewer faint tracks).

    Returns
    -------
    list of fit-style dicts (success, params, t_fit, f_fit, score, source).
    """
    n_freq, n_time = enhanced.shape
    if bank is None:
        bank = default_bank(n_freq, n_time)

    best, best_shape = matched_filter_response(enhanced, bank, line_width)
    thr = _robust_threshold(best, k_sigma)

    # background level for the coverage test (same floor the response uses)
    bg_level = np.median(enhanced) + \
        0.5 * 1.4826 * np.median(np.abs(enhanced - np.median(enhanced)))

    # candidate peaks above threshold, strongest first
    ys, xs = np.where(best >= thr)
    if len(ys) == 0:
        return []
    # suppress peaks whose closest-approach frequency sits in the leakage band
    # (its residual produces spurious high responses)
    if leakage_mask is not None:
        keep = ~leakage_mask[ys]
        ys, xs = ys[keep], xs[keep]
        if len(ys) == 0:
            return []
    scores = best[ys, xs]
    order = np.argsort(scores)[::-1]
    ys, xs, scores = ys[order], xs[order], scores[order]

    existing = list(existing_curves) if existing_curves else []
    picked = []
    out = []
    for y, x, sc in zip(ys, xs, scores):
        # non-maximum suppression vs already-picked peaks
        if any(abs(y - py) <= nms_freq and abs(x - px) <= nms_time
               for py, px in picked):
            continue

        df, tau = bank[best_shape[y, x]]
        half_w = int(min(n_time // 2, max(20, round(3 * tau))))
        t0, f0 = float(x), float(y)
        t_fit = np.arange(max(0, int(t0 - half_w)),
                          min(n_time, int(t0 + half_w) + 1), dtype=float)
        f_fit = scurve_model(t_fit, f0, df, t0, tau)
        # keep only the part inside the frequency band
        inb = (f_fit >= 0) & (f_fit < n_freq)
        if inb.sum() < 5:
            continue
        t_fit, f_fit = t_fit[inb], f_fit[inb]

        # coverage gate: reject curves fitted through scattered noise
        coverage, contiguous, excess = _curve_coverage(
            enhanced, t_fit, f_fit, bg_level)
        if coverage < min_coverage or contiguous < min_contiguous:
            continue

        cand = {
            "success": True,
            "params": {"f0": f0, "df": float(df), "t0": t0, "tau": float(tau)},
            "t_fit": t_fit, "f_fit": f_fit,
            "t_ridge": t_fit, "f_ridge": f_fit,
            "rmse": float("nan"), "r2": float("nan"),
            "n_points": int(len(t_fit)),
            "score": float(sc),
            "coverage": coverage,
            "contiguous": contiguous,
            "source": "matched_filter",
            "merged_from": [],
        }

        # drop if it coincides with an already-known trajectory
        if any(_curves_consistent(cand, ex, n_time, tol_px=consistency_tol_px)
               for ex in existing):
            continue

        picked.append((y, x))
        existing.append(cand)
        out.append(cand)
        if len(out) >= max_candidates:
            break

    return out
