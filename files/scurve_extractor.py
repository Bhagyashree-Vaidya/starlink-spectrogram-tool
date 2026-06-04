"""
S-curve trajectory extractor
============================
For: FunLab, Prof. Sumit Roy & Jesse Chiu, UW ECE
By: Bhagyashree Vaidya

Phase 1 of the trajectory-automation work Jesse asked for.

The connected-component detector in `starlink_pipeline.py` returns noisy
pixel blobs. A real Starlink Doppler pass is a smooth S-curve, so instead of
treating each blob as a track we:

  1. Extract a 1-D ridge from each blob (one frequency value per time bin).
  2. Fit a parametric Doppler S-curve to that ridge.
  3. Merge blob fragments whose fits describe the same trajectory
     (this is the "multiple detections along the same trajectory" case
     Jesse saw in the real captures).

S-curve model (Doppler shift of a LEO pass):

    f(t) = f0 + (df / 2) * tanh((t - t0) / tau)

  f0  : centre frequency bin (closest-approach frequency)
  df  : total frequency swing across the pass (approach -> recede)
  t0  : time bin of closest approach (the inflection point)
  tau : time constant (how sharp the S is)

The fitted parameters give each detection a clean, complete, analytic curve
that is far more robust to match against TLE predictions than raw pixels.
"""

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def scurve_model(t, f0, df, t0, tau):
    """Doppler S-curve: f(t) = f0 + (df/2) * tanh((t - t0) / tau)."""
    return f0 + 0.5 * df * np.tanh((t - t0) / tau)


# ---------------------------------------------------------------------------
# Ridge extraction — one frequency per time bin for a single blob
# ---------------------------------------------------------------------------
def extract_ridge(track_label_image, track_id, intensity_image=None):
    """
    Collapse a detected blob into a 1-D ridge.

    For each time column the blob occupies, take the intensity-weighted mean
    frequency (centroid of the blob in that column). Falls back to the plain
    mean when no intensity image is supplied.

    Returns (t_array, f_array) sorted by time. Empty arrays if the blob is
    missing.
    """
    coords = np.column_stack(np.where(track_label_image == track_id))
    if len(coords) == 0:
        return np.array([]), np.array([])

    f_pix = coords[:, 0]
    t_pix = coords[:, 1]

    t_vals = np.unique(t_pix)
    f_ridge = np.empty(len(t_vals), dtype=float)
    for i, t in enumerate(t_vals):
        sel = t_pix == t
        fs = f_pix[sel]
        if intensity_image is not None:
            w = intensity_image[fs, t]
            if w.sum() > 0:
                f_ridge[i] = np.average(fs, weights=w)
            else:
                f_ridge[i] = fs.mean()
        else:
            f_ridge[i] = fs.mean()
    return t_vals.astype(float), f_ridge


# ---------------------------------------------------------------------------
# Fit one S-curve to a ridge
# ---------------------------------------------------------------------------
def fit_scurve(t, f, n_freq=None, n_time=None):
    """
    Robust least-squares fit of the S-curve model to ridge points (t, f).

    Uses a soft-L1 (Huber-like) loss so a few outlier pixels do not drag the
    fit. Returns a dict with the fitted params, a resampled smooth curve,
    and goodness-of-fit (rmse, r2). `success=False` when the blob is too
    short or the optimiser fails.
    """
    out = {"success": False, "params": None, "rmse": np.nan, "r2": np.nan,
           "t_fit": np.array([]), "f_fit": np.array([]), "n_points": int(len(t))}

    if len(t) < 5:
        return out

    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    span = max(t.max() - t.min(), 1.0)

    # --- initial guess from the data -----------------------------------
    # df: difference between the late-half and early-half mean frequency
    order = np.argsort(t)
    f_sorted = f[order]
    n = len(f_sorted)
    early = f_sorted[: max(1, n // 3)].mean()
    late = f_sorted[-max(1, n // 3):].mean()
    df0 = (late - early)
    if abs(df0) < 1.0:
        df0 = np.sign(df0 if df0 != 0 else 1.0) * max(f.max() - f.min(), 1.0)
    f0_0 = float(np.median(f))
    t0_0 = float(np.median(t))
    tau0 = max(span / 4.0, 1.0)
    p0 = [f0_0, df0, t0_0, tau0]

    # --- bounds --------------------------------------------------------
    fmax = n_freq if n_freq else (f.max() + abs(df0) + 50)
    lo = [-abs(fmax), -2 * abs(fmax) - 1, t.min() - span, 0.5]
    hi = [abs(fmax) * 2, 2 * abs(fmax) + 1, t.max() + span, span * 5 + 1]
    # keep p0 inside bounds
    p0 = [min(max(p0[i], lo[i] + 1e-6), hi[i] - 1e-6) for i in range(4)]

    def resid(p):
        return scurve_model(t, *p) - f

    try:
        res = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1",
                            f_scale=3.0, max_nfev=2000)
    except Exception:
        return out

    f0, df, t0, tau = res.x
    f_pred = scurve_model(t, f0, df, t0, tau)
    residuals = f - f_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((f - f.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    t_fit = np.arange(int(round(t.min())), int(round(t.max())) + 1, dtype=float)
    f_fit = scurve_model(t_fit, f0, df, t0, tau)

    out.update({
        "success": True,
        "params": {"f0": float(f0), "df": float(df),
                   "t0": float(t0), "tau": float(tau)},
        "rmse": rmse,
        "r2": float(r2),
        "t_fit": t_fit,
        "f_fit": f_fit,
    })
    return out


# ---------------------------------------------------------------------------
# Fit every detected blob
# ---------------------------------------------------------------------------
def fit_track_scurves(track_label_image, track_props, intensity_image=None):
    """
    Run ridge extraction + S-curve fit on every detected track.

    Returns a list of fit dicts (one per track) carrying the original
    track_id, ridge points, fitted params, smooth curve, and fit quality.
    """
    n_freq, n_time = track_label_image.shape
    fits = []
    for tp in track_props:
        tid = tp["track_id"]
        # track_props were re-numbered 1..N but the label image still holds
        # the original region labels; fall back to centroid match if needed.
        t, f = extract_ridge(track_label_image, tp.get("_label", tid),
                             intensity_image)
        if len(t) == 0:
            # the re-numbered id may not exist in the label image; skip cleanly
            continue
        fit = fit_scurve(t, f, n_freq=n_freq, n_time=n_time)
        fit["track_id"] = tid
        fit["t_ridge"] = t
        fit["f_ridge"] = f
        fit["centroid_time"] = tp.get("centroid_time")
        fit["centroid_freq"] = tp.get("centroid_freq")
        fits.append(fit)
    return fits


# ---------------------------------------------------------------------------
# Merge fragments that describe the same trajectory
# ---------------------------------------------------------------------------
def _curves_consistent(a, b, n_time, tol_px=3.0, min_overlap=4):
    """
    Two fits belong to the same trajectory if, over the time span they jointly
    cover, their predicted frequencies agree to within `tol_px` on average.

    Works whether their time ranges overlap or are disjoint (a broken track):
    we evaluate both fitted curves over the union span and compare.
    """
    if not (a["success"] and b["success"]):
        return False
    pa, pb = a["params"], b["params"]

    ta = a["t_ridge"]
    tb = b["t_ridge"]
    t_lo = int(max(0, min(ta.min(), tb.min())))
    t_hi = int(min(n_time - 1, max(ta.max(), tb.max())))
    if t_hi <= t_lo:
        return False
    tt = np.arange(t_lo, t_hi + 1, dtype=float)
    fa = scurve_model(tt, pa["f0"], pa["df"], pa["t0"], pa["tau"])
    fb = scurve_model(tt, pb["f0"], pb["df"], pb["t0"], pb["tau"])
    mean_diff = float(np.mean(np.abs(fa - fb)))

    # require the swing directions to agree (same kind of pass)
    same_dir = np.sign(pa["df"]) == np.sign(pb["df"]) or \
        abs(pa["df"]) < 5 or abs(pb["df"]) < 5
    return mean_diff <= tol_px and same_dir


def merge_fragments(fits, track_label_image, intensity_image=None,
                    tol_px=3.0):
    """
    Greedy union-find over fitted fragments: any two consistent fits are merged
    into one trajectory, then the combined ridge is re-fit.

    Returns (merged_fits, merge_map) where merge_map[new_id] = [old_ids...].
    Fits that fail to converge are passed through unmerged.
    """
    n_freq, n_time = track_label_image.shape
    good = [f for f in fits if f["success"]]
    bad = [f for f in fits if not f["success"]]

    n = len(good)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if _curves_consistent(good[i], good[j], n_time, tol_px=tol_px):
                union(i, j)

    # group
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged_fits = []
    merge_map = {}
    new_id = 1
    for _, idxs in sorted(groups.items()):
        old_ids = [good[i]["track_id"] for i in idxs]
        if len(idxs) == 1:
            fit = dict(good[idxs[0]])
            fit["track_id"] = new_id
            fit["merged_from"] = old_ids
            merged_fits.append(fit)
        else:
            # combine all ridge points and re-fit
            t_all = np.concatenate([good[i]["t_ridge"] for i in idxs])
            f_all = np.concatenate([good[i]["f_ridge"] for i in idxs])
            order = np.argsort(t_all)
            refit = fit_scurve(t_all[order], f_all[order],
                              n_freq=n_freq, n_time=n_time)
            refit["track_id"] = new_id
            refit["t_ridge"] = t_all[order]
            refit["f_ridge"] = f_all[order]
            refit["merged_from"] = old_ids
            refit["centroid_time"] = float(np.mean(t_all))
            refit["centroid_freq"] = float(np.mean(f_all))
            merged_fits.append(refit)
        merge_map[new_id] = old_ids
        new_id += 1

    # pass failed fits through (still useful to show as unmatched)
    for f in bad:
        f = dict(f)
        f["track_id"] = new_id
        f["merged_from"] = [f.get("track_id")]
        merge_map[new_id] = f["merged_from"]
        merged_fits.append(f)
        new_id += 1

    return merged_fits, merge_map


# ---------------------------------------------------------------------------
# Convenience: turn a fit into the {time_bins, freq_bins, label} dict the
# correlation module already understands.
# ---------------------------------------------------------------------------
def fit_to_curve(fit, label=None):
    """Adapt an S-curve fit to the prediction-style dict used by correlation."""
    return {
        "label": label if label is not None else f"FIT-{fit['track_id']}",
        "time_bins": np.asarray(fit["t_fit"], dtype=float),
        "freq_bins": np.asarray(fit["f_fit"], dtype=float),
        "track_id": fit["track_id"],
    }
