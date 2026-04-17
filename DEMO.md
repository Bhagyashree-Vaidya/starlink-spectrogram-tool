# Demo Guide — Starlink Spectrogram Processing Tool

A step-by-step walkthrough for showing the tool to Prof. Roy, Jesse, or
anyone else in the FunLab group. Plan on **8–10 minutes** total. Each
section lists what to click, what to say, and what the audience should be
looking at.

---

## 0. Before the meeting (one-time setup)

```bash
cd "Streamlit Interactive UI"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Sanity check:

```bash
python -c "from files.starlink_pipeline import generate_synthetic_spectrogram; print('ok')"
```

Have the project folder open in a terminal so you can launch the app
quickly during the demo.

---

## 1. Launch the app  *(30 sec)*

In the terminal:

```bash
source venv/bin/activate
streamlit run app.py
```

A browser tab opens at **http://localhost:8501**. The page loads with
"Synthetic + Predicted overlay" already selected by default — that is the
mode the demo runs in.

> **Say:** *"This is a Streamlit app that runs locally on my laptop. No
> cloud, nothing to install on your side. The whole pipeline is in this one
> page."*

---

## 2. The headline frame  *(30 sec)*

Before touching anything, point at the **metric bar** at the top:

```
Predicted | Detected | Matched | Recall | Precision | Avg distance
   10     |    16    |    7    |  70 %  |    44 %   |   1.4 px
```

> **Say:** *"Out of the 10 satellites the orbital model predicted, the
> detector found 7 of them in the measured spectrogram, with an average
> match accuracy of 1.4 pixels. Let me show you how it got there."*

---

## 3. Tab 1 — *Before / After*  *(1 min)*

Click **Before / After**.

Point at the **left panel (raw)**:

> *"This is what comes off the receiver. The bright horizontal stripe is
> the signal-leakage band you mentioned — it dominates the colour scale and
> visually drowns out everything else. The faint curves you might just
> barely see in the background are real Starlink passes."*

Point at the **right panel (cleaned)**:

> *"After leakage removal, the same data with the stripe interpolated out.
> Now the Doppler S-curves are clearly visible."*

Point at the **dashed red lines** in the left panel:

> *"Those mark the freq bins the detector flagged as leakage — it does this
> automatically using a percentile-of-mean-power test, no hand-tuning."*

---

## 4. Tab 2 — *Detected Tracks*  *(1 min)*

Click **Detected Tracks**.

> *"Same cleaned spectrogram, but now every connected component the detector
> found is coloured and labelled `T1`, `T2`, …  Each track is a candidate
> satellite."*

Hover over a few tracks to show the tooltips (length, mean power).

> *"The detector uses background subtraction → adaptive threshold →
> connected components → length filter. The minimum track length and the
> threshold are sliders in the sidebar — you can tighten them to be more
> conservative."*

---

## 5. Tab 3 — *Enhancement Detail*  *(45 sec)*

Click **Enhancement Detail**.

Left panel:

> *"This is the background-subtracted version. The dark band is gone, the
> tracks pop out clearly — this is the image the detector actually
> thresholds against."*

Right panel:

> *"Mean power per frequency bin. The peak in the middle is where the
> leakage band lives. The red dashed line is the percentile threshold; the
> shaded region is what got masked out."*

Scroll down — close-ups of individual detected tracks appear as a gallery.

---

## 6. Tab 4 — *Predicted Doppler*  ⭐  *(1.5 min)*

Click **Predicted Doppler**. **This is the new bit Jesse will care about.**

> *"These are the predicted Doppler S-curves. They come from the same
> finite-difference range-rate math Jesse's `doppler-predictor` GUI uses —
> I lifted the prediction class straight from his repo. For the demo I'm
> generating synthetic curves so we don't have to wait on Skyfield to scan
> 9 000 TLEs, but the real Skyfield path is wired up and ready to use a
> real capture timestamp."*

Point at the **lower plot**:

> *"And here's the radial velocity over time — the classic Doppler
> signature. Negative when the satellite is approaching, zero at closest
> approach, positive on recession. Each curve is one pass."*

Pause on the zero-crossing line:

> *"The white dashed line is the closest-approach moment. That's where each
> satellite's S-curve crosses zero velocity."*

---

## 7. Tab 5 — *Measured vs Predicted*  ⭐⭐  *(2 min)*

Click **Measured vs Predicted**. **This is the headline.**

> *"And here it is — predicted curves overlaid on the cleaned spectrogram,
> with the detected pixels coloured to match whichever prediction they were
> assigned to."*

Walk through the legend:

- **Solid coloured line** = predicted curve that *was* matched to a detection.
- **Dotted line** = prediction that the receiver missed.
- **Coloured dots** = detected pixels coloured by their best-match prediction.
- **White dots** = detections that weren't matched to any prediction (RFI / false alarm).

> *"Each detected track gets greedily assigned to the closest predicted
> curve — closest in mean per-pixel distance along the time bins they share.
> The matcher is 1-to-1 with a configurable distance cutoff, so a single
> prediction can't claim multiple detections."*

Scroll down to the **match table**:

> *"Here are the matches by row. Distance in pixels, time-bin overlap, and a
> confidence score that decays with distance."*

Below the table:

> *"And here are the missed predictions and the false alarms. This is what
> turns the GUI into a validation tool — for any capture, you immediately
> see which TLEs the receiver caught, which it didn't, and how much extra
> stuff was flagged."*

Hit the **Download correlation report** button:

> *"And the whole report exports as JSON, so you can compare nights or feed
> it into another script."*

---

## 8. Live tweaking — show the sliders  *(1 min)*

Open the sidebar and tweak two things while the audience watches the
metric bar update:

1. **Receiver miss rate** — drag from 0.2 → 0.0
   > *"Now we're saying the receiver caught everything."*  (Recall jumps to ~100 %.)

2. **False alarms** — drag from 2 → 4
   > *"More RFI bursts in the capture."*  (Precision drops, false-alarm
   > count goes up at the bottom of the correlation tab.)

3. **Max match distance** — drag from 12 → 4 px
   > *"Tighten how close a detection has to be to a prediction for us to
   > call it a match."*  (Some matches drop, distance threshold visibly
   > tightens.)

> **Say:** *"Every parameter is exposed. There is no hand-tuning hidden in
> the source code — what you see in the sidebar is the whole pipeline."*

---

## 9. (Optional) Real SigMF capture  *(2 min)*

If you have a `starlink_sigmf_*` folder downloaded from the FunLab Drive:

1. In the sidebar, change **Data Source** → **Load SigMF capture**.
2. Paste the absolute path to a `.sigmf-meta` file.
3. The app memory-maps the IQ, downsamples to ~5 M samples, and runs the
   STFT (NFFT=1024, NOVERLAP=512, DC notched) — same parameters as
   `plot_sigmf3.py` in the Drive folder.
4. The same leakage / detection pipeline runs on the result.

> **Say:** *"This is the path for real captures. The loader matches the
> `plot_sigmf3.py` and `correlation_preprocessing.py` settings, so any
> capture from the AFRL test runs should drop straight in."*

If you don't have a SigMF file ready, skip this step and just mention it.

---

## 10. Tab 6 — *Track Data* + close out  *(30 sec)*

Click **Track Data**.

> *"Every detected track as a flat table — length, width, eccentricity,
> centroid, bounding box. JSON download here, cleaned-spectrogram `.npy`
> download next to it. That's what you'd hand off to a follow-up
> correspondence-matching pipeline."*

Close with:

> *"All the code is in the project folder. The Doppler predictor and the
> SigMF loader are credited to Jesse and the Drive scripts respectively —
> the synthetic generator, the leakage remover, the detector, the
> correlation engine, and the GUI are all in `files/`. Happy to add
> anything you'd like to see next."*

---

## Cheat sheet — what to click in order

1. Launch:  `streamlit run app.py`
2. Glance at **metric bar** (top of page).
3. **Before / After**  → point at red dashes + leakage stripe disappearing.
4. **Detected Tracks**  → hover a few tracks.
5. **Enhancement Detail**  → point at percentile threshold line.
6. **Predicted Doppler**  → both plots, mention Jesse.
7. **Measured vs Predicted**  → walk through colours, match table, JSON export.
8. **Sidebar**  → tweak miss rate / false alarms / match distance live.
9. *(optional)* SigMF capture path.
10. **Track Data**  → quick mention, close out.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: skyfield` | `pip install -r requirements.txt` (skyfield is in there) |
| Streamlit port already in use | `pkill -f "streamlit run"`  then re-launch |
| `NameError: n_time` | Reload the page (hit `R`) — old session state |
| SigMF file fails to load | Make sure `.sigmf-data` is next to the `.sigmf-meta` file with the same basename |
| Recall / precision look terrible | Bump **Max match distance** to 15–20 px and lower **Threshold (× σ above mean)** to 1.5 |

---

## What to leave the audience with

- **One sentence:** *"Local Streamlit GUI that takes a Starlink capture,
  removes the leakage band, finds every Doppler track, and tells you which
  TLE each one matches."*
- **One number:** the recall / precision shown in the metric bar.
- **One next step:** *"Drop me a real capture timestamp + ground-station
  location and I'll switch the predictor over to the real Skyfield path
  for that pass."*
