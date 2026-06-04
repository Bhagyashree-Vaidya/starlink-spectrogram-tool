"""
Residual analysis  (Phase 5)
============================
For: FunLab, Prof. Sumit Roy & Jesse Chiu, UW ECE
By: Bhagyashree Vaidya

For each matched detection-prediction pair, compute the frequency residual
(measured minus predicted) at every shared time bin. The residual time-series
is where real calibration insight lives:

  - A flat zero residual = perfect match, TLE is fresh, receiver is healthy.
  - A constant offset    = frequency bias (LO error or systematic TLE drift).
  - A linear ramp        = clock drift in the receiver or SDR sample-rate error.
  - A shift at the zero-crossing = timing error (capture timestamp is wrong).
  - A frequency-dependent curve  = ionospheric or tropospheric refraction.
  - Growing scatter       = stale TLE (orbit has drifted from the propagated state).

This module computes the raw residuals, fits a simple linear model to each
(offset + drift), and provides summary statistics the GUI can display.
"""

import numpy as np
from scipy.stats import linregress


def compute_residuals(matches, detected_curves, predictions):
    """
    For each match, compute per-time-bin frequency residuals.

    Parameters
    ----------
    matches : list of dicts from the correlation module
        Each has detected_id, prediction_label, distance_px, confidence.
    detected_curves : list of dicts with time_bins, freq_bins, track_id
        Fitted S-curve outputs (fit_to_curve adapters) or raw blob coords.
    predictions : list of dicts with time_bins, freq_bins, label

    Returns
    -------
    list of residual dicts, one per match:
        match          : the original match dict
        time_bins      : shared time bins (sorted)
        residual_freq  : measured - predicted frequency at each time bin
        det_freq       : measured frequency at each time bin
        pred_freq      : predicted frequency at each time bin
        stats          : {mean, std, median, max_abs, slope, intercept, r_value}
    """
    # build lookups
    det_lookup = {}
    for c in detected_curves:
        tid = c.get("track_id")
        t = np.round(c["time_bins"]).astype(int)
        f = np.asarray(c["freq_bins"], dtype=float)
        det_lookup[tid] = dict(zip(t.tolist(), f.tolist()))

    pred_lookup = {}
    for p in predictions:
        t = np.round(p["time_bins"]).astype(int)
        f = np.asarray(p["freq_bins"], dtype=float)
        pred_lookup[p["label"]] = dict(zip(t.tolist(), f.tolist()))

    results = []
    for m in matches:
        det_id = m["detected_id"]
        pred_label = m["prediction_label"]
        d_map = det_lookup.get(det_id, {})
        p_map = pred_lookup.get(pred_label, {})
        shared = sorted(set(d_map.keys()) & set(p_map.keys()))
        if len(shared) < 3:
            continue

        t_arr = np.array(shared, dtype=float)
        det_f = np.array([d_map[t] for t in shared])
        pred_f = np.array([p_map[t] for t in shared])
        resid = det_f - pred_f

        # linear fit: residual = slope * t + intercept
        stats = _fit_residual(t_arr, resid)

        results.append({
            "match": m,
            "time_bins": t_arr,
            "residual_freq": resid,
            "det_freq": det_f,
            "pred_freq": pred_f,
            "stats": stats,
        })

    return results


def _fit_residual(t, resid):
    """Fit a line to the residual and return summary statistics."""
    stats = {
        "mean": float(np.mean(resid)),
        "std": float(np.std(resid)),
        "median": float(np.median(resid)),
        "max_abs": float(np.max(np.abs(resid))),
        "n_points": int(len(resid)),
    }
    if len(t) >= 3:
        try:
            slope, intercept, r_value, _, std_err = linregress(t, resid)
            stats.update({
                "slope": float(slope),
                "intercept": float(intercept),
                "r_value": float(r_value),
                "slope_std_err": float(std_err),
            })
        except Exception:
            stats.update({"slope": 0.0, "intercept": 0.0,
                          "r_value": 0.0, "slope_std_err": 0.0})
    else:
        stats.update({"slope": 0.0, "intercept": 0.0,
                      "r_value": 0.0, "slope_std_err": 0.0})
    return stats


def diagnose(stats):
    """
    Return a short human-readable diagnosis string from residual stats.

    This is a rough heuristic for the GUI tooltip, not a rigorous test.
    """
    notes = []
    mean = stats["mean"]
    std = stats["std"]
    slope = stats["slope"]
    r = abs(stats.get("r_value", 0))

    if abs(mean) < 0.5 and std < 1.0:
        notes.append("Excellent match")
    elif abs(mean) < 2.0 and std < 3.0:
        notes.append("Good match")

    if abs(mean) > 3.0:
        notes.append(f"Frequency offset: {mean:+.1f} bins")

    if abs(slope) > 0.02 and r > 0.7:
        direction = "increasing" if slope > 0 else "decreasing"
        notes.append(f"Linear drift ({direction}, {slope:+.3f} bins/sample, R={r:.2f})")

    if std > 5.0:
        notes.append(f"High scatter (std={std:.1f} bins) - possible stale TLE or RFI")

    if not notes:
        notes.append("Moderate match")

    return "; ".join(notes)


def aggregate_summary(residual_results):
    """
    Compute fleet-level summary across all matched residuals.
    Useful for detecting systematic receiver issues.
    """
    if not residual_results:
        return {"n_matches": 0}

    means = [r["stats"]["mean"] for r in residual_results]
    stds = [r["stats"]["std"] for r in residual_results]
    slopes = [r["stats"]["slope"] for r in residual_results]

    return {
        "n_matches": len(residual_results),
        "fleet_mean_offset": float(np.mean(means)),
        "fleet_std_offset": float(np.std(means)),
        "fleet_mean_scatter": float(np.mean(stds)),
        "fleet_mean_drift": float(np.mean(slopes)),
        "fleet_std_drift": float(np.std(slopes)),
        "systematic_offset": abs(float(np.mean(means))) > 2.0,
        "systematic_drift": (abs(float(np.mean(slopes))) > 0.01
                             and float(np.std(slopes)) < abs(float(np.mean(slopes)))),
    }
