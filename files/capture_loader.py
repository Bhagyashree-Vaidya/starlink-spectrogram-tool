"""
Capture loader — read measured Starlink signal captures into spectrograms.

Supports three input formats:
  1. SigMF pair (.sigmf-meta + .sigmf-data, complex64 IQ)
  2. CSV from Jesse's doppler-predictor (predicted Doppler track)
  3. NumPy .npy spectrogram (already-computed power array)

Adapted from the capture pipeline in the FunLab Drive folder
(plot_sigmf3.py, plot_sigmf4.py, correlation_preprocessing.py).
"""

import json
import os
import numpy as np
from datetime import datetime
from scipy.signal import spectrogram

# ---------------------------------------------------------------------------
# Defaults (matching the FunLab capture rig)
# ---------------------------------------------------------------------------
DEFAULT_NFFT = 1024
DEFAULT_NOVERLAP = 512
DEFAULT_SAMPLE_RATE = 500_000.0  # Hz, from correlation_preprocessing.py
DEFAULT_CENTER_FREQ = 11.2e9     # Hz, typical Starlink downlink chunk
NOTCH_WIDTH = 40                 # DC notch (bins)


# ---------------------------------------------------------------------------
# SigMF loader
# ---------------------------------------------------------------------------
def load_sigmf(meta_path):
    """
    Load a SigMF capture: returns (iq_complex64, metadata_dict).

    meta_path: path to .sigmf-meta JSON file (data file is auto-discovered).
    """
    if not meta_path.endswith(".sigmf-meta"):
        if meta_path.endswith(".sigmf-data"):
            raise FileNotFoundError(
                f"You pasted the .sigmf-data file — paste the .sigmf-meta "
                f"file instead. It should be right next to it:\n"
                f"  {meta_path.replace('.sigmf-data', '.sigmf-meta')}"
            )
        raise FileNotFoundError(
            f"Expected a .sigmf-meta file, got: {meta_path}"
        )

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Meta file not found: {meta_path}\n"
            f"Make sure to use the full absolute path, e.g.:\n"
            f"  /Users/bhagyashree/Desktop/r001_f11.200GHz_20251121T015107.sigmf-meta"
        )

    with open(meta_path) as f:
        meta = json.load(f)

    data_path = meta_path.replace(".sigmf-meta", ".sigmf-data")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found next to the meta file:\n"
            f"  Missing: {data_path}\n"
            f"Download BOTH files from the Drive folder into the same directory."
        )

    # complex64 = 8 bytes per sample
    iq = np.memmap(data_path, dtype=np.complex64, mode="r")

    global_info = meta.get("global", {})
    captures = meta.get("captures", [{}])
    info = {
        "sample_rate": float(global_info.get("core:sample_rate", DEFAULT_SAMPLE_RATE)),
        "center_freq": float(captures[0].get("core:frequency", DEFAULT_CENTER_FREQ)),
        "datetime": captures[0].get("core:datetime"),
        "n_samples": len(iq),
        "data_path": data_path,
    }
    return iq, info


def iq_to_spectrogram(iq, sample_rate, nfft=DEFAULT_NFFT,
                      noverlap=DEFAULT_NOVERLAP, notch_dc=True):
    """
    Compute a power spectrogram (dB) from complex IQ samples.

    Returns (Sxx_db, freqs_hz, times_s) — frequencies are baseband (centred 0).
    """
    f, t, Sxx = spectrogram(
        iq, fs=sample_rate, nperseg=nfft, noverlap=noverlap,
        return_onesided=False, scaling="density", mode="psd",
    )
    # Shift so DC is in the middle
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f = np.fft.fftshift(f)

    Sxx_db = 10 * np.log10(Sxx + 1e-20)

    if notch_dc:
        mid = nfft // 2
        half = NOTCH_WIDTH // 2
        Sxx_db[mid - half:mid + half + 1, :] = np.median(Sxx_db)

    # Reorient: rows = freq (low to high), cols = time
    return Sxx_db.astype(np.float32), f, t


# ---------------------------------------------------------------------------
# CSV loader (Jesse's predicted-pass format)
# ---------------------------------------------------------------------------
def load_predicted_csv(csv_path):
    """
    Load a per-satellite CSV from doppler-predictor output.

    Returns dict of numpy arrays + metadata.
    """
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    cols = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)

    arr = {}
    for k in ("azimuth_deg", "elevation_deg", "distance_km",
              "relative_velocity_kms", "doppler_shift_hz",
              "rx_freq_hz", "time_minutes"):
        if k in cols:
            arr[k] = np.array(cols[k], dtype=float)

    arr["satellite"] = cols.get("satellite", [""])[0]
    arr["timestamps"] = cols.get("timestamp", [])
    arr["tx_freq_ghz"] = float(cols.get("tx_freq_ghz", ["10.5"])[0])
    return arr


def load_predicted_directory(dir_path):
    """Load all SAT-*.csv files in a doppler-predictor output directory."""
    import glob
    paths = sorted(glob.glob(os.path.join(dir_path, "SAT-*.csv")))
    return [load_predicted_csv(p) for p in paths]


# ---------------------------------------------------------------------------
# NumPy / generic loader
# ---------------------------------------------------------------------------
def load_npy_spectrogram(path):
    """Load a pre-computed spectrogram (.npy). Returns (array, info)."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    info = {"shape": arr.shape, "dtype": str(arr.dtype),
            "min": float(arr.min()), "max": float(arr.max())}
    return arr, info


# ---------------------------------------------------------------------------
# Synthetic measured-data generator (when no real capture is available)
# ---------------------------------------------------------------------------
def synthetic_measured_capture(predictions, n_time=512, n_freq=256,
                               leakage_freq_bin=128, leakage_power=50.0,
                               noise_floor=-20.0, noise_std=3.0,
                               track_jitter_freq=2.0, track_jitter_time=0,
                               miss_fraction=0.0, false_alarm_count=0,
                               seed=42):
    """
    Synthesise a 'measured' spectrogram by painting predicted tracks with
    realistic distortions: noise, leakage, jitter, missing tracks, false alarms.

    Used so the correlation pipeline can be demoed without real SigMF data.
    """
    rng = np.random.RandomState(seed)
    spec = noise_floor + noise_std * rng.randn(n_freq, n_time)

    # Leakage band
    leak = leakage_power * np.exp(
        -0.5 * (np.arange(n_freq) - leakage_freq_bin) ** 2 / 9.0)
    spec += leak[:, None] * (1 + 0.1 * rng.randn(n_time)[None, :])

    measured_tracks = []
    for pred in predictions:
        if rng.random() < miss_fraction:
            continue  # this track was missed by the receiver
        t_idx = pred["time_bins"].astype(int)
        f_idx = pred["freq_bins"] + rng.normal(0, track_jitter_freq, size=len(t_idx))
        t_shift = int(rng.normal(0, track_jitter_time))
        for ti, fi in zip(t_idx, f_idx):
            ti2 = ti + t_shift
            fi2 = int(np.clip(fi, 0, n_freq - 1))
            if 0 <= ti2 < n_time:
                power = rng.uniform(4, 10)
                for df in (-2, -1, 0, 1, 2):
                    if 0 <= fi2 + df < n_freq:
                        w = np.exp(-0.5 * df ** 2)
                        spec[fi2 + df, ti2] += power * w
        measured_tracks.append(pred["label"])

    # False alarms (RFI bursts that aren't real satellites)
    for _ in range(false_alarm_count):
        bt = rng.randint(0, n_time)
        bf = rng.randint(0, n_freq)
        spec[max(0, bf - 3):bf + 3, max(0, bt - 3):bt + 3] += rng.uniform(8, 15)

    return spec, measured_tracks
