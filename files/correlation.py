"""
Correlation engine — match detected tracks against predicted Doppler S-curves.

Two complementary scoring methods:
  1. Curve-distance score:    mean Euclidean distance (in pixel space)
                              between a predicted curve and the nearest
                              detected pixels along its time axis.
  2. Image cross-correlation: render the predicted curve into a binary mask,
                              correlate against the enhanced spectrogram.

Outputs ranked matches with confidence scores so the GUI can colour-code
detections by which TLE they most likely belong to.
"""

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Helper: rasterise a predicted curve into a 2-D mask
# ---------------------------------------------------------------------------
def rasterize_prediction(prediction, shape, line_width=2):
    """
    Convert a predicted (time_bins, freq_bins) curve into a 2-D binary mask
    matching `shape = (n_freq, n_time)`.
    """
    n_freq, n_time = shape
    mask = np.zeros(shape, dtype=bool)
    t = prediction["time_bins"].astype(int)
    f = prediction["freq_bins"]
    for ti, fi in zip(t, f):
        if 0 <= ti < n_time:
            fi_int = int(np.clip(round(fi), 0, n_freq - 1))
            f_lo = max(0, fi_int - line_width)
            f_hi = min(n_freq, fi_int + line_width + 1)
            mask[f_lo:f_hi, ti] = True
    return mask


# ---------------------------------------------------------------------------
# Curve-distance score (predicted curve  ↔ detected pixels)
# ---------------------------------------------------------------------------
def curve_distance_score(prediction, track_label_image, track_props):
    """
    For each detected track, compute the mean perpendicular distance from
    the predicted curve to the closest detected pixels (sharing time bins).

    Returns dict {detected_track_id: mean_distance_px}.  Lower = better match.
    """
    pred_t = prediction["time_bins"].astype(int)
    pred_f = prediction["freq_bins"]
    pred_lookup = dict(zip(pred_t, pred_f))

    out = {}
    for tp in track_props:
        coords = np.column_stack(np.where(track_label_image == tp["track_id"]))
        if len(coords) == 0:
            continue
        # coords: rows = freq, cols = time
        diffs = []
        for f_pix, t_pix in coords:
            if t_pix in pred_lookup:
                diffs.append(abs(f_pix - pred_lookup[t_pix]))
        if diffs:
            out[tp["track_id"]] = float(np.mean(diffs))
    return out


# ---------------------------------------------------------------------------
# Match each detected track to its best-fitting prediction
# ---------------------------------------------------------------------------
def match_tracks_to_predictions(track_label_image, track_props, predictions,
                                max_distance_px=12.0):
    """
    Greedy matcher: for every detected track, find the prediction with the
    smallest mean curve distance. Predictions can match multiple detections
    (and vice-versa) if multiple are within `max_distance_px`.

    Returns:
        matches: list of dicts with keys
                 detected_id, prediction_label, distance_px, n_overlap_pts
        unmatched_detected: list of detected track ids with no good match
        unmatched_predicted: list of prediction labels with no detection
    """
    matches = []
    matched_pred = set()
    matched_det = set()

    # Build distance table {(det_id, pred_label): mean_dist}
    table = {}
    for pred in predictions:
        scores = curve_distance_score(pred, track_label_image, track_props)
        for det_id, dist in scores.items():
            table[(det_id, pred["label"])] = dist

    # Sort all candidate pairs by distance
    pairs = sorted(table.items(), key=lambda kv: kv[1])
    for (det_id, pred_label), dist in pairs:
        if dist > max_distance_px:
            continue
        if det_id in matched_det or pred_label in matched_pred:
            continue
        # Count overlap
        pred = next(p for p in predictions if p["label"] == pred_label)
        coords = np.column_stack(np.where(track_label_image == det_id))
        det_times = set(coords[:, 1].tolist())
        pred_times = set(pred["time_bins"].astype(int).tolist())
        n_overlap = len(det_times & pred_times)

        matches.append({
            "detected_id": int(det_id),
            "prediction_label": pred_label,
            "distance_px": float(dist),
            "n_overlap_pts": int(n_overlap),
            "confidence": float(np.exp(-dist / 4.0)),  # 0-1, drops with distance
        })
        matched_det.add(det_id)
        matched_pred.add(pred_label)

    unmatched_det = [tp["track_id"] for tp in track_props
                     if tp["track_id"] not in matched_det]
    unmatched_pred = [p["label"] for p in predictions
                      if p["label"] not in matched_pred]

    return matches, unmatched_det, unmatched_pred


# ---------------------------------------------------------------------------
# 2-D image cross-correlation (alternative scoring)
# ---------------------------------------------------------------------------
def image_correlation_score(enhanced_image, prediction, search_shift=10):
    """
    Slide the predicted curve mask over the enhanced spectrogram and
    return the maximum normalised cross-correlation along with the
    optimal (df, dt) shift.
    """
    mask = rasterize_prediction(prediction, enhanced_image.shape).astype(float)
    if mask.sum() == 0:
        return 0.0, (0, 0)

    img = enhanced_image - enhanced_image.mean()
    msk = mask - mask.mean()

    best = -np.inf
    best_shift = (0, 0)
    for df in range(-search_shift, search_shift + 1):
        for dt in range(-search_shift, search_shift + 1):
            shifted = np.roll(np.roll(msk, df, axis=0), dt, axis=1)
            num = np.sum(img * shifted)
            den = np.sqrt(np.sum(img ** 2) * np.sum(shifted ** 2)) + 1e-12
            score = num / den
            if score > best:
                best = score
                best_shift = (df, dt)
    return float(best), best_shift


# ---------------------------------------------------------------------------
# Summary statistics for the GUI
# ---------------------------------------------------------------------------
def correlation_summary(matches, unmatched_det, unmatched_pred,
                        n_predicted, n_detected):
    """Build a one-shot summary dict for the metric bar."""
    n_matched = len(matches)
    if matches:
        avg_conf = float(np.mean([m["confidence"] for m in matches]))
        avg_dist = float(np.mean([m["distance_px"] for m in matches]))
    else:
        avg_conf = 0.0
        avg_dist = float("nan")

    return {
        "n_predicted": n_predicted,
        "n_detected": n_detected,
        "n_matched": n_matched,
        "n_missed": len(unmatched_pred),
        "n_false_alarm": len(unmatched_det),
        "recall": n_matched / n_predicted if n_predicted else 0.0,
        "precision": n_matched / n_detected if n_detected else 0.0,
        "avg_confidence": avg_conf,
        "avg_distance_px": avg_dist,
    }
