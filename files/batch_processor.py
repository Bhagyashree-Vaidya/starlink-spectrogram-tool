"""
Batch processor  (Phase 6)
==========================
For: FunLab, Prof. Sumit Roy & Jesse Chiu, UW ECE
By: Bhagyashree Vaidya

Process a folder of captures in one shot. For each file the full pipeline
runs: load -> leakage removal -> detection -> S-curve fitting -> matching
-> residual analysis. Results are collected into a single summary table
(one row per capture) and a detailed per-match table, both exportable as
CSV or JSON.

Supports mixed input: .npy spectrograms and .sigmf-meta IQ captures in
the same folder.
"""

import os
import glob
import time
import numpy as np
from datetime import datetime

from starlink_pipeline import (
    remove_leakage,
    detect_tracks,
    detect_leakage_band,
)
from capture_loader import (
    load_sigmf,
    iq_to_spectrogram,
    load_npy_spectrogram,
)
from scurve_extractor import fit_track_scurves, merge_fragments, fit_to_curve
from correlation import match_curves_hungarian, correlation_summary
from matched_filter import recover_faint_predictions
from residual_analysis import compute_residuals, aggregate_summary


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------
def process_one(file_path, predictions=None, params=None):
    """
    Run the full pipeline on one capture file.

    Parameters
    ----------
    file_path : str
        Path to .npy or .sigmf-meta file.
    predictions : list of prediction dicts, optional
        TLE-based predictions to correlate against.
    params : dict, optional
        Pipeline parameters (defaults filled in for any missing keys).

    Returns
    -------
    dict with keys: file, status, spectrogram_shape, n_detected, n_trajectories,
    n_matched, recall, precision, avg_distance, n_faint_recovered,
    fleet_mean_offset, fleet_mean_scatter, processing_time_s, error,
    matches (list), residuals (list).
    """
    p = _defaults(params)
    result = {
        "file": os.path.basename(file_path),
        "path": file_path,
        "status": "error",
        "error": None,
        "spectrogram_shape": None,
        "n_detected": 0,
        "n_trajectories": 0,
        "n_matched": 0,
        "recall": 0.0,
        "precision": 0.0,
        "avg_distance": float("nan"),
        "n_faint_recovered": 0,
        "fleet_mean_offset": float("nan"),
        "fleet_mean_scatter": float("nan"),
        "processing_time_s": 0.0,
        "matches": [],
        "residuals": [],
    }

    t0 = time.time()
    try:
        # load
        spec = _load_file(file_path)
        result["spectrogram_shape"] = f"{spec.shape[0]}x{spec.shape[1]}"

        # leakage removal
        cleaned, leakage_mask = remove_leakage(
            spec, method=p["removal_method"])

        # detection
        labels, props, enhanced = detect_tracks(
            cleaned, min_track_length=p["min_track_length"],
            power_threshold=None)
        result["n_detected"] = len(props)

        # S-curve fitting + merge
        raw_fits = fit_track_scurves(labels, props, intensity_image=enhanced)
        merged, _ = merge_fragments(
            raw_fits, labels, intensity_image=enhanced,
            tol_px=p["merge_tol_px"])
        curves = [
            fit_to_curve(f) for f in merged
            if f["success"]
            and f.get("r2", 0) >= p["min_r2"]
            and f.get("n_points", 0) >= p["min_ridge_pts"]
        ]
        result["n_trajectories"] = len(curves)

        # matching
        if predictions:
            matches, un_det, un_pred = match_curves_hungarian(
                curves, predictions, max_distance_px=p["max_match_dist"])
            cs = correlation_summary(
                matches, un_det, un_pred,
                n_predicted=len(predictions), n_detected=len(curves))
            result["n_matched"] = cs["n_matched"]
            result["recall"] = cs["recall"]
            result["precision"] = cs["precision"]
            result["avg_distance"] = cs["avg_distance_px"]
            result["matches"] = matches

            # faint recovery
            label_to_pred = {pr["label"]: pr for pr in predictions}
            un_pred_dicts = [label_to_pred[l] for l in un_pred
                            if l in label_to_pred]
            faint = recover_faint_predictions(
                enhanced, un_pred_dicts, leakage_mask=leakage_mask,
                k_sigma=p["faint_k_sigma"])
            result["n_faint_recovered"] = len(faint)

            # residuals
            if matches and curves:
                resids = compute_residuals(matches, curves, predictions)
                result["residuals"] = resids
                agg = aggregate_summary(resids)
                result["fleet_mean_offset"] = agg.get(
                    "fleet_mean_offset", float("nan"))
                result["fleet_mean_scatter"] = agg.get(
                    "fleet_mean_scatter", float("nan"))

        result["status"] = "ok"

    except Exception as e:
        result["error"] = str(e)

    result["processing_time_s"] = round(time.time() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def process_folder(folder_path, predictions=None, params=None,
                   progress_callback=None):
    """
    Find all .npy and .sigmf-meta files in folder_path and run the pipeline
    on each. Returns (summary_rows, all_results).

    progress_callback(i, n, filename) is called before each file if provided.
    """
    files = _find_captures(folder_path)
    if not files:
        return [], []

    results = []
    for i, fp in enumerate(files):
        if progress_callback:
            progress_callback(i, len(files), os.path.basename(fp))
        r = process_one(fp, predictions=predictions, params=params)
        results.append(r)

    summary = _build_summary(results)
    return summary, results


def _find_captures(folder):
    """Discover .npy and .sigmf-meta files recursively."""
    npy = sorted(glob.glob(os.path.join(folder, "**", "*.npy"), recursive=True))
    sigmf = sorted(glob.glob(os.path.join(folder, "**", "*.sigmf-meta"),
                              recursive=True))
    # skip datetime axis files
    npy = [f for f in npy if "datetime" not in os.path.basename(f).lower()]
    return npy + sigmf


def _load_file(path):
    """Load a single capture file, return 2D spectrogram array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.ndim == 0:
            raise ValueError(f"File contains a scalar/dict, not a spectrogram: {path}")
        if arr.ndim == 1:
            raise ValueError(
                f"1D array (shape {arr.shape}) -- this looks like a timestamp "
                f"axis, not a spectrogram. Look for a companion file without "
                f"the 'datetime_' prefix.")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D, got shape {arr.shape}")
        return arr.astype(np.float64)
    elif ext == ".sigmf-meta":
        from capture_loader import load_sigmf, iq_to_spectrogram
        iq, info = load_sigmf(path)
        max_samples = 5_000_000
        if len(iq) > max_samples:
            iq = iq[::len(iq) // max_samples]
        spec, _, _ = iq_to_spectrogram(np.asarray(iq), info["sample_rate"])
        return spec
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _defaults(params):
    """Fill in default pipeline parameters."""
    d = {
        "removal_method": "interpolate",
        "min_track_length": 15,
        "merge_tol_px": 3.0,
        "min_r2": 0.3,
        "min_ridge_pts": 8,
        "max_match_dist": 12.0,
        "faint_k_sigma": 1.5,
    }
    if params:
        d.update(params)
    return d


def _build_summary(results):
    """One row per file for the summary table."""
    rows = []
    for r in results:
        rows.append({
            "File": r["file"],
            "Status": r["status"],
            "Shape": r.get("spectrogram_shape", ""),
            "Detected": r["n_detected"],
            "Trajectories": r["n_trajectories"],
            "Matched": r["n_matched"],
            "Recall": f"{r['recall']*100:.0f}%" if r["recall"] else "0%",
            "Precision": f"{r['precision']*100:.0f}%" if r["precision"] else "0%",
            "Avg dist (px)": f"{r['avg_distance']:.1f}"
                             if np.isfinite(r["avg_distance"]) else "-",
            "Faint recovered": r["n_faint_recovered"],
            "Mean offset (bins)": f"{r['fleet_mean_offset']:.2f}"
                                  if np.isfinite(r["fleet_mean_offset"]) else "-",
            "Time (s)": r["processing_time_s"],
            "Error": r.get("error") or "",
        })
    return rows
