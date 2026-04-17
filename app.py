"""
Starlink Spectrogram Processing Tool — Interactive UI
======================================================
For: FunLab, Prof. Sumit Roy & Jesse Chiu, UW ECE
By: Bhagyashree Vaidya | March 2026

Streamlit wrapper around the spectrogram processing pipeline.
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

MAX_PLOT_PIXELS = 1500  # max pixels on each axis for Plotly heatmaps

# Import pipeline functions from the files/ directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"))
from starlink_pipeline import (
    generate_synthetic_spectrogram,
    detect_leakage_band,
    remove_leakage,
    enhance_tracks,
    detect_tracks,
)
from doppler_predictor import (
    DopplerPredictor,
    load_tle_file,
    find_visible_satellites,
    generate_synthetic_prediction,
    SIEG_HALL,
    C_M_S,
)
from capture_loader import (
    load_sigmf,
    iq_to_spectrogram,
    load_predicted_csv,
    load_npy_spectrogram,
    synthetic_measured_capture,
)
from correlation import (
    match_tracks_to_predictions,
    correlation_summary,
    rasterize_prediction,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Starlink Spectrogram Tool",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — compact, professional look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* tighter spacing */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    /* metric cards */
    div[data-testid="stMetric"] {
        background: #0e1117; border: 1px solid #262730;
        border-radius: 8px; padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { font-size: 0.8rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    /* sidebar header */
    .sidebar-header { font-size: 0.75rem; color: #888; text-transform: uppercase;
                      letter-spacing: 0.08em; margin-top: 1rem; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — All controls                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.title("📡 Control Panel")

    # --- Data source --------------------------------------------------
    st.markdown('<p class="sidebar-header">Data Source</p>', unsafe_allow_html=True)

    data_source = st.radio(
        "Input data",
        [
            "Generate synthetic",
            "Synthetic + Predicted overlay",
            "Load .npy spectrogram",
            "Load SigMF capture (.sigmf-meta)",
        ],
        index=1,
        help=(
            "• Synthetic = self-contained demo\n"
            "• Synthetic + Predicted = drives the correlation pipeline\n"
            "• .npy / SigMF = real captures from the FunLab Drive folder"
        ),
    )

    uploaded_file = None
    npy_sample_choice = None
    npy_local_path = None
    sigmf_meta_path = None
    tle_path = None

    if data_source == "Load .npy spectrogram":
        npy_source = st.radio(
            "NPY source",
            ["Use a sample file", "Paste file path (for large files)", "Upload small file"],
            index=0,
            help="Sample = instant demo.  Paste path = best for GB-sized Drive files.  Upload = small files only (<200 MB).",
        )
        if npy_source == "Use a sample file":
            npy_sample_choice = st.selectbox(
                "Sample spectrogram",
                [
                    "medium_10sats.npy — 10 satellites, default settings",
                    "easy_5sats_lownoise.npy — 5 satellites, low noise",
                    "hard_15sats_noisy.npy — 15 satellites, heavy noise",
                    "medium_10sats_cleaned.npy — pre-cleaned (no leakage)",
                ],
                index=0,
            )
            st.caption(
                "These are pre-generated synthetic spectrograms in `sample_data/`."
            )
        elif npy_source == "Paste file path (for large files)":
            npy_local_path = st.text_input(
                "Absolute path to .npy file",
                value="",
                help="Reads directly from disk — no size limit, no upload wait.",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload spectrogram (.npy)",
                type=["npy"],
                help="A 2D numpy array, shape (n_freq, n_time), values in dB.  Max ~200 MB.",
            )
    elif data_source == "Load SigMF capture (.sigmf-meta)":
        sigmf_meta_path = st.text_input(
            "SigMF meta file path",
            value="",
            help="Path to a .sigmf-meta file. The .sigmf-data file must sit alongside it.",
        )
        st.caption(
            "Download a `starlink_sigmf_*` folder from the FunLab Drive and "
            "paste the path to any `.sigmf-meta` file.  \n"
            "Example: `~/Downloads/starlink_sigmf_20251120_175016/"
            "r001_f11.200GHz_20251121T015107.sigmf-meta`"
        )

    # --- Synthetic generation params ----------------------------------
    # Used by both "Generate synthetic" and "Synthetic + Predicted overlay"
    is_synthetic = data_source in ("Generate synthetic", "Synthetic + Predicted overlay")
    if is_synthetic:
        st.markdown('<p class="sidebar-header">Synthetic Data</p>', unsafe_allow_html=True)

        n_satellites = st.slider("Number of satellites", 1, 20, 10)
        n_time = st.select_slider("Time bins", [256, 512, 1024], value=512)
        n_freq = st.select_slider("Frequency bins", [128, 256, 512], value=256)
        leakage_power = st.slider(
            "Leakage power (dB)", 10.0, 80.0, 50.0, step=5.0,
            help="Amplitude of the bright leakage band",
        )
        leakage_freq_pct = st.slider(
            "Leakage position (% of freq axis)", 10, 90, 50,
            help="Where the leakage band sits in frequency",
        )
        noise_std = st.slider("Noise level (σ)", 0.5, 8.0, 3.0, step=0.5)
        seed = st.number_input("Random seed", 0, 9999, 42, step=1)
    else:
        # Provide harmless defaults so the variables exist for any code path
        n_satellites = 10
        n_time = 512
        n_freq = 256
        leakage_power = 50.0
        leakage_freq_pct = 50
        noise_std = 3.0
        seed = 42

    # --- Doppler prediction (Skyfield / TLE) --------------------------
    st.markdown('<p class="sidebar-header">Doppler Prediction</p>', unsafe_allow_html=True)

    enable_prediction = st.checkbox(
        "Enable predicted S-curve overlay",
        value=(data_source == "Synthetic + Predicted overlay"),
        help="Compute Doppler curves and correlate them against detected tracks.",
    )

    prediction_mode = "synthetic"
    tle_file_path = None
    capture_datetime_str = ""
    obs_lat = SIEG_HALL["latitude"]
    obs_lon = SIEG_HALL["longitude"]
    obs_alt = SIEG_HALL["altitude"]
    pass_duration_min = 10.0
    pass_step_s = 1.0
    max_visible_sats = 30
    n_predicted = 10
    pred_seed = 42
    miss_fraction = 0.2
    false_alarms = 2

    if enable_prediction:
        prediction_mode = st.radio(
            "Prediction source",
            ["Real TLE (Skyfield)", "Synthetic (demo)"],
            index=0,
            help="Real TLE uses Jesse's predictor with actual orbital data. "
                 "Synthetic generates fake curves for quick demos.",
        )

        if prediction_mode == "Real TLE (Skyfield)":
            default_tle = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "doppler-predictor", "starlink.txt",
            )
            tle_file_path = st.text_input(
                "TLE file path",
                value=default_tle if os.path.exists(default_tle) else "",
                help="3-line TLE file. The repo ships `doppler-predictor/starlink.txt` (~9000 sats).",
            )
            capture_datetime_str = st.text_input(
                "Capture time (UTC)",
                value="2025-11-21 01:51:07",
                help="When the spectrogram was captured. "
                     "Nov 2025 captures: 2025-11-21 01:51:07. "
                     "Jan 2026 capture: 2026-01-20 23:06:06.",
            )

            st.markdown('<p class="sidebar-header">Ground Station</p>', unsafe_allow_html=True)
            obs_lat = st.number_input("Latitude (°N)", value=47.6553, format="%.4f")
            obs_lon = st.number_input("Longitude (°E)", value=-122.3035, format="%.4f")
            obs_alt = st.number_input("Altitude (m)", value=60.0, step=1.0)

            st.markdown('<p class="sidebar-header">Pass Parameters</p>', unsafe_allow_html=True)
            pass_duration_min = st.slider("Pass window (min)", 1.0, 30.0, 10.0, step=1.0,
                                          help="How many minutes of Doppler curve to compute per satellite")
            pass_step_s = st.select_slider("Time step (s)", [0.5, 1.0, 2.0, 5.0], value=1.0)
            max_visible_sats = st.slider("Max visible satellites", 5, 100, 30,
                                         help="Cap how many satellites to process (saves time)")

        else:
            n_predicted = st.slider("Predicted satellites", 1, 30, 10)
            pred_seed = st.number_input("Prediction seed", 0, 9999, 42, step=1)
            miss_fraction = st.slider(
                "Receiver miss rate", 0.0, 0.5, 0.2, step=0.05,
                help="Synthetic-only: fraction of predicted tracks the receiver will miss",
            )
            false_alarms = st.slider(
                "False alarms", 0, 5, 2,
                help="Synthetic-only: number of RFI bursts injected to test precision",
            )

    max_match_dist = st.slider(
        "Max match distance (px)", 2.0, 30.0, 12.0, step=1.0,
        disabled=not enable_prediction,
        help="Detected tracks within this distance of a predicted curve are matched",
    )

    # --- Leakage removal params ---------------------------------------
    st.markdown('<p class="sidebar-header">Leakage Removal</p>', unsafe_allow_html=True)

    leakage_percentile = st.slider(
        "Detection percentile", 80, 99, 95,
        help="Frequency bins above this percentile of mean power are flagged as leakage",
    )
    leakage_dilation = st.slider(
        "Mask dilation (px)", 0, 10, 3,
        help="Expand the leakage mask to catch sidelobes",
    )
    removal_method = st.selectbox(
        "Removal method",
        ["interpolate", "median", "zero"],
        index=0,
        help="How the leakage band is filled after masking",
    )

    # --- Track detection params ---------------------------------------
    st.markdown('<p class="sidebar-header">Track Detection</p>', unsafe_allow_html=True)

    detection_sigma = st.slider(
        "Smoothing σ", 0.5, 5.0, 1.5, step=0.25,
        help="Gaussian smoothing applied during enhancement (higher = fewer false positives, but may merge tracks)",
    )
    threshold_multiplier = st.slider(
        "Threshold (× σ above mean)", 1.0, 5.0, 2.0, step=0.25,
        help="Pixels above mean + k·σ of the enhanced image are kept",
    )
    min_track_length = st.slider(
        "Min track length (px)", 5, 60, 15,
        help="Connected components shorter than this are discarded as noise",
    )
    median_bg_size = st.slider(
        "Background filter size", 5, 31, 15, step=2,
        help="Median filter kernel for background estimation (must be odd)",
    )

    # --- Display options ----------------------------------------------
    st.markdown('<p class="sidebar-header">Display</p>', unsafe_allow_html=True)

    colorscale = st.selectbox(
        "Colorscale", ["Viridis", "Plasma", "Inferno", "Hot", "Cividis", "Electric"], index=0
    )
    show_ground_truth = False
    if is_synthetic:
        show_ground_truth = st.checkbox("Show ground truth tracks", value=False)

    st.divider()
    run_btn = st.button("▶  Run Pipeline", type="primary", width="stretch")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PROCESSING — runs on button click or first load                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@st.cache_data(show_spinner=False)
def _generate(n_time, n_freq, n_satellites, leakage_power, leakage_freq_bin,
              noise_std, seed):
    return generate_synthetic_spectrogram(
        n_time=n_time, n_freq=n_freq, n_satellites=n_satellites,
        leakage_power=leakage_power, leakage_freq_bin=leakage_freq_bin,
        noise_std=noise_std, seed=seed,
    )


def _run_leakage_removal(spectrogram, percentile, dilation, method):
    """Leakage removal with user-controlled percentile and dilation."""
    from scipy import ndimage as _ndi

    mean_power = np.mean(spectrogram, axis=1)
    threshold = np.percentile(mean_power, percentile)
    mask = mean_power > threshold
    if dilation > 0:
        mask = _ndi.binary_dilation(mask, iterations=dilation)
    cleaned, _ = remove_leakage(spectrogram, leakage_mask=mask, method=method)
    return cleaned, mask


def _run_detection(spectrogram, sigma, threshold_k, min_len, bg_size):
    """Track detection with user-controlled parameters."""
    from scipy.ndimage import gaussian_filter, median_filter
    from skimage.morphology import remove_small_objects
    from skimage.measure import label, regionprops
    from scipy import ndimage as _ndi

    # Enhancement
    background = median_filter(spectrogram, size=bg_size)
    enhanced = np.clip(spectrogram - background, 0, None)
    enhanced = gaussian_filter(enhanced, sigma=sigma)
    if enhanced.max() > enhanced.min():
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

    # Threshold
    thr = np.mean(enhanced) + threshold_k * np.std(enhanced)
    binary = enhanced > thr

    # Morphology cleanup
    binary = remove_small_objects(binary, min_size=min_len)
    binary = _ndi.binary_dilation(binary, iterations=1)
    binary = _ndi.binary_erosion(binary, iterations=1)

    track_labels, n_tracks = label(binary, return_num=True)
    regions = regionprops(track_labels, intensity_image=enhanced)

    props = []
    for r in regions:
        if r.axis_major_length < min_len:
            continue
        coords = r.coords
        props.append({
            "track_id": r.label,
            "centroid_freq": float(r.centroid[0]),
            "centroid_time": float(r.centroid[1]),
            "bbox": {
                "freq_min": int(r.bbox[0]), "time_min": int(r.bbox[1]),
                "freq_max": int(r.bbox[2]), "time_max": int(r.bbox[3]),
            },
            "area_pixels": int(r.area),
            "length": float(r.axis_major_length),
            "width": float(r.axis_minor_length),
            "orientation_deg": float(np.degrees(r.orientation)),
            "mean_power": float(r.intensity_mean),
            "max_power": float(r.intensity_max),
            "eccentricity": float(r.eccentricity),
            "freq_range": [int(coords[:, 0].min()), int(coords[:, 0].max())],
            "time_range": [int(coords[:, 1].min()), int(coords[:, 1].max())],
        })

    props.sort(key=lambda x: x["length"], reverse=True)
    for i, p in enumerate(props):
        p["track_id"] = i + 1

    return track_labels, props, enhanced


@st.cache_data(show_spinner=False)
def _run_tle_prediction(tle_path, capture_time_str, lat, lon, alt,
                        duration_min, step_s, max_sats, elevation_mask=10.0):
    """
    Scan TLEs for visible satellites at the capture time, compute each
    satellite's Doppler pass, and return predictions in the same format
    the correlation engine expects.
    """
    from datetime import datetime
    from skyfield.api import utc

    entries = load_tle_file(tle_path)
    obs_time = datetime.strptime(capture_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc)
    loc = {"latitude": lat, "longitude": lon, "altitude": alt}

    # Phase 1: find visible satellites
    visible = find_visible_satellites(
        entries, obs_time, ue_location=loc,
        elevation_mask=elevation_mask, max_sats=max_sats,
    )

    # Phase 2: compute Doppler pass for each
    predictions = []
    pass_data_list = []
    duration_s = duration_min * 60

    for dp in visible:
        pdata = dp.compute_pass(
            obs_time, duration_s=duration_s, step_s=step_s,
            elevation_mask=elevation_mask,
        )
        if len(pdata["time_s"]) < 5:
            continue  # skip satellites with very short visibility

        # Convert to the (time_bins, freq_bins) format the correlation engine uses.
        # Map doppler_hz → freq bin: we normalise over the full set later.
        predictions.append({
            "label": dp.sat_name,
            "time_s": pdata["time_s"],
            "doppler_hz": pdata["doppler_hz"],
            "velocity_m_s": pdata["velocity_m_s"],
            "az_deg": pdata["az_deg"],
            "el_deg": pdata["el_deg"],
            "dist_km": pdata["dist_km"],
        })
        pass_data_list.append(pdata)

    return predictions, pass_data_list, len(entries), len(visible), obs_time.isoformat()


def _map_predictions_to_spectrogram(predictions, n_freq, n_time):
    """
    Convert real TLE predictions (time in seconds, doppler in Hz) into
    pixel-space (time_bins, freq_bins) so they can overlay a spectrogram
    and be fed to the correlation engine.

    Maps the time range of all predictions to [0, n_time) and the Doppler
    range to [0, n_freq).
    """
    if not predictions:
        return []

    # Collect global time and Doppler ranges across all predictions
    all_t = np.concatenate([p["time_s"] for p in predictions])
    all_d = np.concatenate([p["doppler_hz"] for p in predictions])
    t_min, t_max = all_t.min(), all_t.max()
    d_min, d_max = all_d.min(), all_d.max()

    # Slight padding so tracks don't land on the exact edge
    d_pad = max(abs(d_max - d_min) * 0.05, 1000)  # at least 1 kHz
    d_min -= d_pad
    d_max += d_pad

    mapped = []
    for pred in predictions:
        t_bins = (pred["time_s"] - t_min) / max(t_max - t_min, 1) * (n_time - 1)
        f_bins = (pred["doppler_hz"] - d_min) / max(d_max - d_min, 1) * (n_freq - 1)
        mapped.append({
            "label": pred["label"],
            "time_bins": t_bins,
            "freq_bins": f_bins,
            "velocity_m_s": pred["velocity_m_s"],
            "doppler_hz": pred["doppler_hz"],
            "time_s": pred["time_s"],
        })
    return mapped


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN AREA                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown("## 📡 Starlink Spectrogram Processing Tool")
st.caption("Prototype v2  |  Doppler prediction + correlation pipeline")

# --- Load / generate data -----------------------------------------------
spectrogram = None
metadata = None
predictions = []   # list of {label, time_bins, freq_bins, velocity_m_s}
sigmf_info = None

if data_source == "Generate synthetic":
    leakage_freq_bin = int(n_freq * leakage_freq_pct / 100)
    spectrogram, metadata = _generate(
        n_time, n_freq, n_satellites, leakage_power, leakage_freq_bin,
        noise_std, int(seed),
    )

elif data_source == "Synthetic + Predicted overlay":
    # 1) Run Jesse's predictor (synthetic Doppler S-curves)
    predictions = generate_synthetic_prediction(
        n_time=n_time, n_freq=n_freq,
        n_tracks=n_predicted, seed=int(pred_seed),
    )
    # 2) Synthesise a measured spectrogram by painting predictions with noise/leakage
    leakage_freq_bin = int(n_freq * leakage_freq_pct / 100)
    spectrogram, gt_labels = synthetic_measured_capture(
        predictions,
        n_time=n_time, n_freq=n_freq,
        leakage_freq_bin=leakage_freq_bin,
        leakage_power=leakage_power,
        noise_std=noise_std,
        miss_fraction=miss_fraction,
        false_alarm_count=int(false_alarms),
        seed=int(seed),
    )
    metadata = {
        "n_satellites": len(predictions),
        "n_painted": len(gt_labels),
        "tracks": [],
        "note": "Synthesised measured capture with predicted-curve ground truth",
    }

elif data_source == "Load .npy spectrogram":
    _npy_arr = None  # raw loaded array before validation

    if npy_sample_choice is not None:
        sample_fname = npy_sample_choice.split(" —")[0].strip()
        sample_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "sample_data", sample_fname
        )
        _npy_arr = np.load(sample_path)
        _npy_label = f"sample: {sample_fname}"

    elif npy_local_path:
        if not os.path.exists(npy_local_path):
            st.error(f"File not found: `{npy_local_path}`")
            st.stop()
        with st.spinner(f"Loading `{os.path.basename(npy_local_path)}`…"):
            _npy_arr = np.load(npy_local_path, allow_pickle=True, mmap_mode="r")
            # If mmap'd, convert to a real array (may be large — downsample first)
            if hasattr(_npy_arr, "filename"):
                # Downsample huge arrays to manageable size before copying to RAM
                if _npy_arr.ndim == 2:
                    max_proc = 4096
                    f_stride = max(1, _npy_arr.shape[0] // max_proc)
                    t_stride = max(1, _npy_arr.shape[1] // max_proc)
                    _npy_arr = np.array(_npy_arr[::f_stride, ::t_stride], dtype=np.float64)
                else:
                    _npy_arr = np.array(_npy_arr)
        _npy_label = os.path.basename(npy_local_path)

    elif uploaded_file is not None:
        try:
            import io
            _npy_arr = np.load(io.BytesIO(uploaded_file.read()), allow_pickle=True)
        except Exception as _e:
            st.error(
                f"**Could not read `{uploaded_file.name}`** — {_e}\n\n"
                "Make sure the file is a valid NumPy `.npy` array saved with "
                "`np.save()`. If the file is larger than 200 MB, use the "
                "**Paste file path** option instead."
            )
            st.stop()
        _npy_label = uploaded_file.name

    # Validate shape
    if _npy_arr is not None:
        if _npy_arr.ndim != 2:
            st.error(
                f"**Expected a 2D spectrogram array, got shape `{_npy_arr.shape}` "
                f"(dtype `{_npy_arr.dtype}`).**\n\n"
                + (
                    "This looks like a **datetime/timestamp axis**, not the "
                    "spectrogram power data. Look for a companion file in the "
                    "same Drive folder without the `datetime_` prefix — e.g. "
                    "`updated_waterfall_CF_12.075GHz.npy`."
                    if "datetime" in _npy_arr.dtype.name or _npy_arr.ndim == 1
                    else "The file must be a 2D array with shape `(n_freq, n_time)` "
                         "containing power values in dB."
                )
            )
            st.stop()
        spectrogram = _npy_arr.astype(np.float64) if _npy_arr.dtype != np.float64 else _npy_arr
        metadata = {"n_satellites": "?", "tracks": [],
                    "note": f"Loaded: {_npy_label}"}

elif data_source == "Load SigMF capture (.sigmf-meta)" and sigmf_meta_path:
    try:
        iq, sigmf_info = load_sigmf(sigmf_meta_path)
        # downsample to ~5 M samples max so the FFT is responsive
        max_samples = 5_000_000
        if len(iq) > max_samples:
            stride = len(iq) // max_samples
            iq = iq[::stride]
        spec_db, freqs, times = iq_to_spectrogram(
            np.asarray(iq), sigmf_info["sample_rate"],
        )
        spectrogram = spec_db
        metadata = {
            "n_satellites": "—",
            "tracks": [],
            "sigmf": sigmf_info,
            "note": f"SigMF capture: {sigmf_info.get('datetime', '?')} "
                    f"@ {sigmf_info['center_freq']/1e9:.3f} GHz",
        }
    except Exception as e:
        st.error(f"Failed to load SigMF: {e}")
        st.stop()

if spectrogram is None:
    st.info(
        "Pick a data source in the sidebar:\n\n"
        "• **Synthetic + Predicted overlay** — best demo, runs the full correlation pipeline\n"
        "• **SigMF capture** — for real Starlink captures from the FunLab Drive folder"
    )
    st.stop()

# Show spectrogram dimensions (helpful for SigMF / large files)
n_f, n_t = spectrogram.shape
total_px = n_f * n_t
if total_px > MAX_PLOT_PIXELS ** 2:
    st.info(
        f"Spectrogram is {n_f} x {n_t} ({total_px/1e6:.1f} M pixels). "
        f"Plots will be downsampled to ~{MAX_PLOT_PIXELS} x {MAX_PLOT_PIXELS} for display; "
        f"processing runs on the full-resolution data."
    )

# --- Run TLE-based prediction if enabled --------------------------------
if enable_prediction and prediction_mode == "Real TLE (Skyfield)":
    if not tle_file_path or not os.path.exists(tle_file_path):
        st.error(f"TLE file not found: `{tle_file_path}`")
        st.stop()
    if not capture_datetime_str:
        st.error("Enter a capture time in the sidebar (e.g. `2025-11-21 01:51:07`).")
        st.stop()

    with st.spinner(f"Scanning TLEs for visible satellites at {capture_datetime_str} UTC…"):
        raw_preds, pass_data_list, n_tle_total, n_visible, obs_iso = _run_tle_prediction(
            tle_file_path, capture_datetime_str,
            obs_lat, obs_lon, obs_alt,
            pass_duration_min, pass_step_s, max_visible_sats,
        )

    st.success(
        f"Scanned {n_tle_total} TLEs — **{n_visible}** satellites above horizon, "
        f"**{len(raw_preds)}** with usable passes."
    )

    # Map the real predictions into pixel space matching the spectrogram
    n_f, n_t = spectrogram.shape
    predictions = _map_predictions_to_spectrogram(raw_preds, n_f, n_t)

elif enable_prediction and prediction_mode == "Synthetic (demo)" and not predictions:
    # Only generate synthetic if not already done (e.g. by "Synthetic + Predicted overlay")
    predictions = generate_synthetic_prediction(
        n_time=spectrogram.shape[1], n_freq=spectrogram.shape[0],
        n_tracks=n_predicted, seed=int(pred_seed),
    )

# --- Run pipeline -------------------------------------------------------
with st.spinner("Removing leakage…"):
    cleaned, leakage_mask = _run_leakage_removal(
        spectrogram, leakage_percentile, leakage_dilation, removal_method,
    )

with st.spinner("Detecting tracks…"):
    track_labels, track_props, enhanced = _run_detection(
        cleaned, detection_sigma, threshold_multiplier, min_track_length,
        median_bg_size,
    )

# --- Correlate detected tracks against predicted curves ----------------
matches = []
unmatched_det = []
unmatched_pred = []
corr_summary = None

if enable_prediction and predictions:
    with st.spinner("Correlating against predicted Doppler curves…"):
        matches, unmatched_det, unmatched_pred = match_tracks_to_predictions(
            track_labels, track_props, predictions,
            max_distance_px=max_match_dist,
        )
        corr_summary = correlation_summary(
            matches, unmatched_det, unmatched_pred,
            n_predicted=len(predictions), n_detected=len(track_props),
        )

# ── Metric bar ──────────────────────────────────────────────────────────
n_detected = len(track_props)
n_truth = metadata["n_satellites"] if metadata else "—"
leakage_bins = int(np.sum(leakage_mask))
snr_before = float(spectrogram.max() - np.median(spectrogram))
snr_after = float(cleaned.max() - np.median(cleaned))

if corr_summary:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Predicted", corr_summary["n_predicted"])
    m2.metric("Detected", n_detected)
    m3.metric("Matched", corr_summary["n_matched"])
    m4.metric("Recall", f"{corr_summary['recall']*100:.0f}%")
    m5.metric("Precision", f"{corr_summary['precision']*100:.0f}%")
    m6.metric("Avg distance", f"{corr_summary['avg_distance_px']:.1f} px")
else:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tracks detected", n_detected)
    m2.metric("Ground truth", n_truth)
    m3.metric("Leakage bins masked", leakage_bins)
    m4.metric("Dynamic range (raw)", f"{snr_before:.1f} dB")
    m5.metric("Dynamic range (clean)", f"{snr_after:.1f} dB")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VISUALIZATIONS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if enable_prediction:
    tab_compare, tab_tracks, tab_enhanced, tab_predict, tab_correlate, tab_data = st.tabs(
        ["Before / After", "Detected Tracks", "Enhancement Detail",
         "Predicted Doppler", "Measured vs Predicted", "Track Data"]
    )
else:
    tab_compare, tab_tracks, tab_enhanced, tab_data = st.tabs(
        ["Before / After", "Detected Tracks", "Enhancement Detail", "Track Data"]
    )
    tab_predict = None
    tab_correlate = None

# ---------------------------------------------------------------------------
# Downsample large spectrograms for Plotly (which serialises the entire
# 2-D array as JSON — anything over ~2 000 x 2 000 blows up the browser).
# ---------------------------------------------------------------------------
def _downsample_for_plot(arr, max_dim=MAX_PLOT_PIXELS):
    """Block-mean downsample a 2-D array so neither axis exceeds max_dim."""
    n_freq, n_time = arr.shape
    f_stride = max(1, n_freq // max_dim)
    t_stride = max(1, n_time // max_dim)
    if f_stride == 1 and t_stride == 1:
        return arr
    # Trim to exact multiple, then reshape + mean
    nf = (n_freq // f_stride) * f_stride
    nt = (n_time // t_stride) * t_stride
    trimmed = arr[:nf, :nt]
    return trimmed.reshape(nf // f_stride, f_stride,
                           nt // t_stride, t_stride).mean(axis=(1, 3))

spec_plot = _downsample_for_plot(spectrogram)
cln_plot  = _downsample_for_plot(cleaned)
enh_plot  = _downsample_for_plot(enhanced)

# Shared percentile clipping for consistent color range
raw_vmin, raw_vmax = float(np.percentile(spectrogram, 2)), float(np.percentile(spectrogram, 98))
cln_vmin, cln_vmax = float(np.percentile(cleaned, 2)), float(np.percentile(cleaned, 98))

# ── Tab 1: Before / After ──────────────────────────────────────────────
with tab_compare:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Raw Spectrogram (leakage dominates)",
                        "After Leakage Removal"),
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        go.Heatmap(z=spec_plot, colorscale=colorscale,
                    zmin=raw_vmin, zmax=raw_vmax,
                    colorbar=dict(title="Power dB", x=0.46, len=0.9)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=cln_plot, colorscale=colorscale,
                    zmin=cln_vmin, zmax=cln_vmax,
                    colorbar=dict(title="Power dB", x=1.02, len=0.9)),
        row=1, col=2,
    )

    # Draw leakage band boundaries on raw panel
    leak_rows = np.where(leakage_mask)[0]
    if len(leak_rows):
        for yval in [float(leak_rows.min()), float(leak_rows.max())]:
            fig.add_shape(
                type="line", x0=0, x1=spectrogram.shape[1], y0=yval, y1=yval,
                line=dict(color="red", width=1.5, dash="dash"),
                row=1, col=1,
            )

    fig.update_layout(
        height=500, margin=dict(t=40, b=30),
        xaxis_title="Time (samples)", yaxis_title="Frequency (bins)",
        xaxis2_title="Time (samples)", yaxis2_title="Frequency (bins)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Detected Tracks overlay ────────────────────────────────────
with tab_tracks:
    fig2 = go.Figure()

    # Background: cleaned spectrogram in grayscale
    fig2.add_trace(
        go.Heatmap(z=cln_plot, colorscale="Gray", zmin=cln_vmin, zmax=cln_vmax,
                    showscale=False, name="Cleaned"),
    )

    # Overlay each track as a scatter of its pixels
    if track_props:
        # Build a colour list
        import plotly.express as px
        palette = px.colors.qualitative.Set1
        for tp in track_props:
            tid = tp["track_id"]
            mask_t = track_labels == tid
            ys, xs = np.where(mask_t)
            color = palette[(tid - 1) % len(palette)]
            fig2.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=3, color=color, opacity=0.7),
                name=f"Track {tid}  (L={tp['length']:.0f}px)",
                hovertemplate=(
                    f"Track {tid}<br>Time: %{{x}}<br>Freq: %{{y}}<br>"
                    f"Length: {tp['length']:.0f}px<br>Power: {tp['mean_power']:.3f}"
                    "<extra></extra>"
                ),
            ))
            # Label
            fig2.add_annotation(
                x=tp["centroid_time"], y=tp["centroid_freq"],
                text=f"T{tid}", showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor=color, borderpad=2, opacity=0.85,
            )

    # Optionally overlay ground truth curves
    if show_ground_truth and metadata and "tracks" in metadata:
        for gt in metadata["tracks"]:
            fig2.add_trace(go.Scatter(
                x=gt["track_times"], y=gt["track_frequencies"],
                mode="lines", line=dict(color="cyan", width=1.5, dash="dot"),
                name=f"GT {gt['satellite_id']}",
                opacity=0.6,
            ))

    fig2.update_layout(
        height=550, margin=dict(t=30, b=30),
        xaxis_title="Time (samples)", yaxis_title="Frequency (bins)",
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0.5)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Enhancement detail ─────────────────────────────────────────
with tab_enhanced:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Enhanced image** (background subtracted)")
        fig3a = go.Figure(go.Heatmap(
            z=enh_plot, colorscale="Hot",
            zmin=0, zmax=float(np.percentile(enhanced, 99)),
            colorbar=dict(title="Enhanced"),
        ))
        fig3a.update_layout(height=420, margin=dict(t=10, b=30),
                            xaxis_title="Time", yaxis_title="Freq")
        st.plotly_chart(fig3a, use_container_width=True)

    with col_b:
        st.markdown("**Mean power per frequency bin** (leakage detector)")
        mean_pwr = np.mean(spectrogram, axis=1)
        threshold_line = np.percentile(mean_pwr, leakage_percentile)
        fig3b = go.Figure()
        fig3b.add_trace(go.Scatter(
            y=np.arange(len(mean_pwr)), x=mean_pwr,
            mode="lines", name="Mean power",
        ))
        fig3b.add_vline(x=threshold_line, line_dash="dash", line_color="red",
                        annotation_text=f"P{leakage_percentile} threshold")
        # shade leakage region
        leak_idx = np.where(leakage_mask)[0]
        if len(leak_idx):
            fig3b.add_hrect(
                y0=float(leak_idx.min()), y1=float(leak_idx.max()),
                fillcolor="red", opacity=0.15,
                annotation_text="Masked", annotation_position="top left",
            )
        fig3b.update_layout(height=420, margin=dict(t=10, b=30),
                            xaxis_title="Mean power (dB)", yaxis_title="Freq bin")
        st.plotly_chart(fig3b, use_container_width=True)

    # Individual track gallery
    if track_props:
        st.markdown("---")
        st.markdown("#### Individual Track Close-ups")
        n_show = min(len(track_props), 8)
        cols = st.columns(min(n_show, 4))
        for i, tp in enumerate(track_props[:n_show]):
            bbox = tp["bbox"]
            margin = 10
            f0 = max(0, bbox["freq_min"] - margin)
            f1 = min(cleaned.shape[0], bbox["freq_max"] + margin)
            t0 = max(0, bbox["time_min"] - margin)
            t1 = min(cleaned.shape[1], bbox["time_max"] + margin)
            patch = cleaned[f0:f1, t0:t1]
            fig_p = go.Figure(go.Heatmap(z=patch, colorscale=colorscale, showscale=False))
            fig_p.update_layout(
                height=200, width=220,
                margin=dict(l=20, r=10, t=30, b=20),
                title=dict(text=f"Track {tp['track_id']}", font=dict(size=11)),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            )
            cols[i % len(cols)].plotly_chart(fig_p, use_container_width=True)
            cols[i % len(cols)].caption(
                f"Length {tp['length']:.0f}px · Power {tp['mean_power']:.3f}"
            )

# ── Tab: Predicted Doppler curves (Skyfield-style overlay) ────────────
if tab_predict is not None:
    with tab_predict:
        if not predictions:
            st.info("Enable predicted overlay in the sidebar to populate this tab.")
        else:
            is_real_tle = "doppler_hz" in predictions[0]
            import plotly.express as px
            palette = px.colors.qualitative.Set2

            if is_real_tle:
                st.markdown(
                    f"**Real Doppler S-curves** — computed from TLE data via "
                    f"Skyfield SGP4 at **{capture_datetime_str} UTC** for "
                    f"**{len(predictions)}** visible satellites."
                )

                # Doppler shift vs time (the real S-curves in kHz)
                fig_dop = go.Figure()
                for i, pred in enumerate(predictions):
                    color = palette[i % len(palette)]
                    fig_dop.add_trace(go.Scatter(
                        x=pred["time_s"], y=pred["doppler_hz"] / 1e3,
                        mode="lines", line=dict(color=color, width=2),
                        name=pred["label"],
                        hovertemplate=(
                            f"{pred['label']}<br>"
                            "Time: %{x:.0f}s<br>Doppler: %{y:.1f} kHz<extra></extra>"
                        ),
                    ))
                fig_dop.add_hline(y=0, line_dash="dash", line_color="white",
                                  annotation_text="Zero Doppler (closest approach)")
                fig_dop.update_layout(
                    height=480, margin=dict(t=30, b=30),
                    xaxis_title="Time into pass (seconds)",
                    yaxis_title="Doppler shift (kHz)",
                    title="Predicted Doppler S-curves (Skyfield / TLE)",
                    legend=dict(font=dict(size=9)),
                )
                st.plotly_chart(fig_dop, use_container_width=True)

                # Velocity vs time
                st.markdown("**Radial velocity** — the classic Doppler signature")
                fig_vel = go.Figure()
                for i, pred in enumerate(predictions):
                    color = palette[i % len(palette)]
                    fig_vel.add_trace(go.Scatter(
                        x=pred["time_s"],
                        y=pred["velocity_m_s"] / 1000,
                        mode="lines", line=dict(color=color, width=2),
                        name=pred["label"],
                    ))
                fig_vel.add_hline(y=0, line_dash="dash", line_color="white",
                                  annotation_text="Closest approach")
                fig_vel.update_layout(
                    height=360, margin=dict(t=30, b=30),
                    xaxis_title="Time into pass (seconds)",
                    yaxis_title="Radial velocity (km/s)  [- approach, + recede]",
                    legend=dict(font=dict(size=9)),
                )
                st.plotly_chart(fig_vel, use_container_width=True)

            else:
                # Synthetic mode — pixel-space plots
                st.markdown(
                    "**Predicted Doppler S-curves** — synthetic demo curves "
                    "(switch to Real TLE mode in the sidebar for actual orbital data)."
                )
                fig_pred = go.Figure()
                for i, pred in enumerate(predictions):
                    color = palette[i % len(palette)]
                    fig_pred.add_trace(go.Scatter(
                        x=pred["time_bins"], y=pred["freq_bins"],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        name=pred["label"],
                    ))
                fig_pred.update_layout(
                    height=480, margin=dict(t=30, b=30),
                    xaxis_title="Time (samples)",
                    yaxis_title="Frequency (bins)",
                    title="Predicted Doppler S-curves (synthetic)",
                    legend=dict(font=dict(size=9)),
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                fig_vel = go.Figure()
                for i, pred in enumerate(predictions):
                    color = palette[i % len(palette)]
                    fig_vel.add_trace(go.Scatter(
                        x=pred["time_bins"], y=pred["velocity_m_s"],
                        mode="lines", line=dict(color=color, width=2),
                        name=pred["label"],
                    ))
                fig_vel.add_hline(y=0, line_dash="dash", line_color="white",
                                  annotation_text="Closest approach")
                fig_vel.update_layout(
                    height=360, margin=dict(t=30, b=30),
                    xaxis_title="Time (samples)",
                    yaxis_title="Radial velocity (m/s)",
                    legend=dict(font=dict(size=9)),
                )
                st.plotly_chart(fig_vel, use_container_width=True)

# ── Tab: Measured vs Predicted (correlation overlay) ─────────────────
if tab_correlate is not None:
    with tab_correlate:
        if not predictions:
            st.info("Enable predicted overlay in the sidebar to populate this tab.")
        else:
            st.markdown(
                "**Correlation overlay** — detected tracks coloured by their "
                "best-matching predicted satellite. Solid lines = predictions, "
                "scatter = detected pixels."
            )
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Heatmap(
                z=cln_plot, colorscale="Gray",
                zmin=cln_vmin, zmax=cln_vmax, showscale=False,
            ))

            import plotly.express as px
            palette = px.colors.qualitative.Set1

            # Map detected_id -> prediction_label for colour assignment
            det_to_pred = {m["detected_id"]: m["prediction_label"] for m in matches}
            pred_to_color = {p["label"]: palette[i % len(palette)]
                             for i, p in enumerate(predictions)}

            # Predicted curves (lines)
            for pred in predictions:
                matched = pred["label"] in {m["prediction_label"] for m in matches}
                fig_corr.add_trace(go.Scatter(
                    x=pred["time_bins"], y=pred["freq_bins"],
                    mode="lines",
                    line=dict(
                        color=pred_to_color[pred["label"]],
                        width=2.5,
                        dash="solid" if matched else "dot",
                    ),
                    name=f"{'✓' if matched else '✗'} {pred['label']}",
                    opacity=0.9 if matched else 0.4,
                ))

            # Detected pixels coloured by their match
            for tp in track_props:
                det_id = tp["track_id"]
                ys, xs = np.where(track_labels == det_id)
                if det_id in det_to_pred:
                    color = pred_to_color[det_to_pred[det_id]]
                    label = f"D{det_id} ↔ {det_to_pred[det_id]}"
                else:
                    color = "white"
                    label = f"D{det_id} (false alarm)"
                fig_corr.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=4, color=color,
                                line=dict(width=0.5, color="black"),
                                opacity=0.8),
                    name=label, showlegend=False,
                    hovertemplate=f"{label}<br>t=%{{x}}, f=%{{y}}<extra></extra>",
                ))

            fig_corr.update_layout(
                height=560, margin=dict(t=30, b=30),
                xaxis_title="Time (samples)", yaxis_title="Frequency (bins)",
                legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.5)"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Match table
            st.markdown("#### Match results")
            if matches:
                match_rows = [{
                    "Detected": m["detected_id"],
                    "Predicted satellite": m["prediction_label"],
                    "Distance (px)": f"{m['distance_px']:.2f}",
                    "Time overlap": m["n_overlap_pts"],
                    "Confidence": f"{m['confidence']*100:.1f}%",
                } for m in matches]
                st.dataframe(match_rows, width="stretch")
            else:
                st.warning("No matches found at the current distance threshold.")

            cm1, cm2 = st.columns(2)
            with cm1:
                if unmatched_pred:
                    st.error(f"**Missed predictions** ({len(unmatched_pred)}):  "
                             + ", ".join(unmatched_pred))
            with cm2:
                if unmatched_det:
                    st.warning(f"**Unmatched detections (false alarms)** "
                               f"({len(unmatched_det)}):  "
                               + ", ".join(str(d) for d in unmatched_det))

            # Export
            st.download_button(
                "⬇ Download correlation report (JSON)",
                data=json.dumps({
                    "summary": corr_summary,
                    "matches": matches,
                    "unmatched_predictions": unmatched_pred,
                    "unmatched_detections": unmatched_det,
                }, indent=2),
                file_name="correlation_report.json",
                mime="application/json",
            )

# ── Tab 4: Track data table & export ──────────────────────────────────
with tab_data:
    if not track_props:
        st.warning("No tracks detected with current settings. Try lowering the threshold.")
    else:
        # Flatten for display
        rows = []
        for tp in track_props:
            rows.append({
                "Track": tp["track_id"],
                "Length (px)": f"{tp['length']:.0f}",
                "Width (px)": f"{tp['width']:.1f}",
                "Mean power": f"{tp['mean_power']:.4f}",
                "Max power": f"{tp['max_power']:.4f}",
                "Eccentricity": f"{tp['eccentricity']:.3f}",
                "Centroid (freq)": f"{tp['centroid_freq']:.1f}",
                "Centroid (time)": f"{tp['centroid_time']:.1f}",
                "Freq range": f"{tp['freq_range'][0]}–{tp['freq_range'][1]}",
                "Time range": f"{tp['time_range'][0]}–{tp['time_range'][1]}",
            })
        st.dataframe(rows, width="stretch")

        # Export buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇ Download track data (JSON)",
                data=json.dumps({"n_detected": n_detected, "tracks": track_props}, indent=2),
                file_name="detected_tracks.json",
                mime="application/json",
            )
        with col_dl2:
            import io
            buf = io.BytesIO()
            np.save(buf, cleaned)
            st.download_button(
                "⬇ Download cleaned spectrogram (.npy)",
                data=buf.getvalue(),
                file_name="cleaned_spectrogram.npy",
                mime="application/octet-stream",
            )

# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Starlink Spectrogram Processing Tool.  "
    "Prototype v2  ·  Bhagyashree Vaidya  ·  "
    "Prof. Sumit Roy, UW ECE  ·  April 2026  ·  Jesse Chiu"
)
