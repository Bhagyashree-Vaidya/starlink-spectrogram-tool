# Starlink Spectrogram Processing Tool

**Built by:** Bhagyashree Vaidya (MS student, UW ECE)
**Lab:** Prof. Sumit Roy, FunLab
**Doppler predictor:** adapted from Jesse Chiu's [doppler-predictor](https://github.com/jessest94106/doppler-predictor)
**Status:** Prototype v2 (April 2026)
**Live app:** https://starlink-spectrogram-tool.streamlit.app

A Streamlit app that takes a raw Starlink satellite capture, cleans it up, finds the satellite tracks, and checks them against TLE-based Doppler predictions.

---

## Screenshots

**Before / After leakage removal**
![Before and After](screenshots/before_after.png)

**Detected Tracks**
![Detected Tracks](screenshots/detected_tracks.png)

**Predicted Doppler (TLE-based, Skyfield)**
![Predicted Doppler](screenshots/Predicted-doppler.png)

**Measured vs Predicted**
![Measured vs Predicted](screenshots/measured_vs_predicted.png)

**Colour scale reference**
![Colour scale](screenshots/colorscale.png)

---

## What it does

The FunLab receiver captures wide-band RF data as Starlink satellites pass overhead. The raw spectrograms have two problems: a bright signal-leakage band that drowns out everything else, and faint satellite tracks that are hard to see. There was also no easy way to check whether a detected track actually matches a predicted satellite pass.

This tool addresses all three:

1. Removes the leakage band automatically.
2. Detects each faint Doppler S-curve in the cleaned spectrogram.
3. Compares the detections against TLE-based predictions and reports which ones matched, which were missed, and which are likely RFI.

---

## Pipeline

```
Load capture -> Remove leakage -> Detect tracks -> Correlate vs predictions
```

**Stage 1: Load**

Supports four input modes:

| Mode | Source |
|---|---|
| Generate synthetic | Built-in generator, no data needed |
| Synthetic + Predicted overlay | Generator + predicted curves, runs the full pipeline |
| Load .npy spectrogram | A pre-computed NumPy array |
| Load SigMF capture | Raw IQ from the FunLab receiver (.sigmf-meta + .sigmf-data) |

For SigMF files, the app memory-maps the IQ data, runs an STFT (NFFT=1024, noverlap=512), shifts DC to centre, and notches the central 40 bins. These settings match the FunLab `plot_sigmf3.py` and `correlation_preprocessing.py` scripts.

**Stage 2: Leakage removal**

1. Computes mean power per frequency bin.
2. Flags bins above a percentile threshold (default P95) as leakage.
3. Dilates the mask a few pixels to catch sidelobes.
4. Fills the masked rows with linear interpolation.

**Stage 3: Track detection**

1. Subtracts a local background (median filter).
2. Smooths with a Gaussian.
3. Thresholds at mean + k * sigma of the enhanced image.
4. Removes short connected components.
5. Labels each remaining blob and extracts shape stats.

**Stage 3b: S-curve trajectory fitting (optional, on by default)**

Connected-component detection returns noisy pixel blobs, and a single
satellite pass often breaks into several fragments. This stage fits a
parametric Doppler S-curve to each blob and merges the fragments that lie on
the same trajectory:

```
f(t) = f0 + (Δf / 2) * tanh((t - t0) / τ)
```

- `f0`  centre frequency (closest-approach frequency)
- `Δf`  total frequency swing across the pass
- `t0`  time of closest approach (the inflection point)
- `τ`   time constant (how sharp the S is)

Each detection becomes a clean analytic curve with a goodness-of-fit (RMSE,
R²) instead of a pixel blob. Fragments whose fitted curves agree to within a
few pixels are merged into one trajectory, which removes the
"multiple detections on the same trajectory" problem.

**Stage 4: Correlation**

The Doppler prediction logic is adapted from Jesse Chiu's `doppler-predictor`. It uses Skyfield's SGP4 propagator to compute the range rate for each TLE and convert it to a Doppler shift:

```
f_doppler = -f_tx * v_radial / c
v_radial = d(slant_range) / dt  (finite difference, 1 s step)
```

The correlation module then matches detections to predictions. Two matchers
are available:

1. **Hungarian (optimal)** — runs on the fitted S-curves and minimises the
   total assignment cost (`scipy.optimize.linear_sum_assignment`), so a
   locally-closest pairing can't starve a globally better one. This is the
   default.
2. **Greedy** — the original pixel-distance matcher (closest pair first),
   used when S-curve fitting is turned off.

Either way it reports recall, precision, average distance, and per-match
confidence. Pairs beyond a configurable distance cutoff are rejected.

A synthetic prediction generator is also included for demos that do not need a live Skyfield scan.

---

## GUI

**Sidebar**

- Data source picker
- Synthetic data settings (array shape, leakage, noise, seed)
- Doppler prediction settings (miss rate, false alarms, max match distance)
- Trajectory fitting (S-curve fit + optimal matching toggle, fragment merge tolerance)
- Leakage removal settings (percentile, dilation, fill method)
- Track detection settings (threshold, minimum length, background filter)

**Metric bar (when correlation is on)**

```
Predicted | Trajectories | Matched | Recall | Precision | Avg distance
```

**Tabs**

| Tab | Content |
|---|---|
| Before / After | Raw vs cleaned spectrogram, leakage bins marked in red |
| Detected Tracks | Cleaned spectrogram with each track labelled |
| Enhancement Detail | Background-subtracted image and mean power plot |
| S-curve Fits | Fitted Doppler S-curves per trajectory, fit parameters and R², fragment-merge summary |
| Predicted Doppler | Predicted S-curves and radial velocity over time |
| Measured vs Predicted | Correlation overlay (fitted curves coloured by match), match table, JSON export |
| Track Data | Shape stats for every detected track, JSON and .npy export |

---

## Project layout

```
Streamlit Interactive UI/
├── app.py                    # Streamlit app
├── files/
│   ├── starlink_pipeline.py  # Leakage removal and track detection
│   ├── scurve_extractor.py   # S-curve fitting + fragment merging (Phase 1)
│   ├── doppler_predictor.py  # Doppler prediction (adapted from Jesse's repo)
│   ├── capture_loader.py     # SigMF and .npy loaders
│   └── correlation.py        # Greedy + Hungarian matchers (Phase 2)
├── doppler-predictor/        # Jesse's repo, added as a git submodule
├── screenshots/
├── sample_data/
├── requirements.txt
├── Dockerfile
└── .streamlit/config.toml
```

---

## Setup

```bash
cd "Streamlit Interactive UI"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# opens at http://localhost:8501
```

Docker:

```bash
docker build -t starlink-spectrogram .
docker run -p 8501:8501 starlink-spectrogram
```

**Using real captures**

1. Download a `starlink_sigmf_*` folder from the FunLab Drive.
2. Make sure the `.sigmf-meta` and `.sigmf-data` files are in the same folder.
3. In the sidebar, choose Load SigMF capture and paste the path to the `.sigmf-meta` file.
4. Adjust sliders as needed. Defaults match the FunLab capture scripts.

**Using TLE predictions**

```python
from files.doppler_predictor import load_tle_file, DopplerPredictor
from datetime import datetime

entries = load_tle_file("doppler-predictor/starlink.txt")
name, l1, l2 = entries[0]
dp = DopplerPredictor(l1, l2, sat_name=name)
pass_data = dp.compute_pass(datetime.utcnow(), duration_s=600, step_s=1.0,
                            elevation_mask=10.0)
```

---

## Credits

- **Doppler prediction** is adapted from [jessest94106/doppler-predictor](https://github.com/jessest94106/doppler-predictor) by Jesse Chiu (UW ECE). The TLE file `starlink.txt` is from the same repo.
- **SigMF loading and STFT settings** follow the FunLab capture pipeline (`plot_sigmf3.py`, `correlation_preprocessing.py` from the shared Drive folder).
- **Leakage removal, track detection, correlation module, and Streamlit GUI** were built by Bhagyashree Vaidya as part of a FunLab prototype.
