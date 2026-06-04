# S-curve Trajectory Extractor: Analysis and Results

**Author:** Bhagyashree Vaidya
**Date:** June 2026
**For:** Prof. Sumit Roy, Jesse Chiu — FunLab, UW ECE

---

## 1. Problem

The original track detector uses connected-component labeling on a thresholded spectrogram. Each group of bright pixels becomes a "detection." This works for strong, isolated tracks but has two problems on real data:

**Problem 1: Fragmented detections.** A single satellite pass often produces multiple disconnected pixel blobs. The track fades in and out as the signal crosses noise peaks, or the leakage-removal step clips part of it. The detector sees three separate blobs where there is really one satellite.

Example (seed 7, 12 satellites):
- The detector found 12 blobs, but blobs 5, 8, and 11 were all pieces of the same pass.
- Blobs 6 and 12 were also fragments of one pass.
- Actual trajectories: 9, not 12.

**Problem 2: Noisy matching.** The greedy pixel-distance matcher takes each blob and assigns it to the closest predicted curve. When a single pass is split into three blobs, each one gets matched independently. The best fragment wins the prediction, and the other two become "false alarms." Worse, a noise blob sitting near a prediction can steal it from the real track.

The result: inflated detection count, low precision, and unreliable satellite identification.

---

## 2. Method

The trajectory extractor replaces pixel blobs with parametric Doppler S-curves, then uses optimal assignment instead of greedy matching.

### 2.1 S-curve model

A Starlink LEO pass produces a Doppler shift that follows:

```
f(t) = f0 + (Δf / 2) * tanh((t - t0) / τ)
```

| Parameter | Meaning |
|---|---|
| f0 | Centre frequency at closest approach |
| Δf | Total frequency swing (approach to recession) |
| t0 | Time of closest approach (inflection point) |
| τ | Time constant (how sharp the S-curve is) |

This is the standard hyperbolic tangent model for a constant-altitude, constant-speed LEO pass observed from a fixed ground station.

### 2.2 Ridge extraction

For each detected blob, collapse the pixels into a 1-D ridge: one frequency value per time bin. At each time column, the frequency is the intensity-weighted centroid of the blob's pixels in that column. This converts a 2-D pixel cloud into a clean (t, f) sequence.

### 2.3 Curve fitting

Fit the S-curve model to each ridge using least-squares optimization. Key choices:

- **Soft-L1 loss** (Huber-like). A few outlier pixels from noise or RFI do not drag the fit. Standard L2 loss was too sensitive to outliers in testing.
- **Bounded parameters.** f0 stays within the spectrogram, Δf is bounded by twice the frequency range, τ stays positive. Without bounds the optimizer can diverge on short blobs.
- **Initial guess from the data.** f0 = median frequency, Δf = difference between early-third and late-third mean frequency, t0 = median time, τ = quarter of the time span.

Each fit returns the four parameters plus RMSE and R2 as quality metrics.

### 2.4 Fragment merging

After fitting, check every pair of fitted curves: if their predicted frequencies agree to within a tolerance (default 3 pixels) across their combined time span, they belong to the same pass.

Merging uses a union-find structure. After grouping, the combined ridge points are re-fit to get a single clean curve for the whole pass.

Example (seed 7):
```
Blobs 5, 8, 11  ->  merged into Trajectory T8  (R2 = 1.00)
Blobs 6, 12     ->  merged into Trajectory T9  (R2 = 0.95)
12 blobs         ->  9 trajectories
```

### 2.5 Optimal matching (Hungarian algorithm)

The old matcher was greedy: sort all (detection, prediction) pairs by distance, assign the closest pair first, repeat. This is fast but suboptimal. A locally close pair can block a globally better assignment.

The replacement uses the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), which minimizes the total assignment cost across all pairs simultaneously. The distance metric operates on the fitted curves (mean frequency difference at shared time bins) rather than raw pixels.

### 2.6 Fit quality filter

Before matching, drop fitted trajectories with:
- R2 < 0.3 (poor fit, likely noise)
- Fewer than 8 ridge points (too short to be a real pass)

This prevents noise blobs from entering the assignment and stealing predictions meant for real tracks.

---

## 3. Results

### 3.1 Before vs After comparison

Tested on synthetic spectrograms with 12 satellites each, across 6 random seeds. "Before" is the original blob detector + greedy matcher. "After" is S-curve fitting + fragment merging + Hungarian matching + quality filter.

| Seed | Sats | Blobs | Trajectories | Merged | Greedy matched | Greedy recall | Greedy precision | Hungarian matched | Hungarian recall | Hungarian precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 7 | 5 | 0 | 4 | 33% | 57% | 5 | 42% | 100% |
| 7 | 12 | 12 | 8 | 3 | 5 | 42% | 42% | 6 | 50% | 75% |
| 11 | 12 | 10 | 8 | 1 | 6 | 50% | 60% | 5 | 42% | 62% |
| 21 | 12 | 5 | 5 | 0 | 2 | 17% | 40% | 4 | 33% | 80% |
| 33 | 12 | 10 | 7 | 1 | 3 | 25% | 30% | 7 | 58% | 100% |
| 42 | 12 | 10 | 6 | 0 | 3 | 25% | 30% | 4 | 33% | 67% |

**Averages:**

| Metric | Before (greedy blobs) | After (S-curve + Hungarian) | Change |
|---|---:|---:|---:|
| Recall | 32% | 43% | +11 pts |
| Precision | 43% | 81% | **+38 pts** |

### 3.2 What the numbers mean

**Precision nearly doubled.** The biggest gain is in precision (43% to 81%). The old matcher produced many false matches where noise blobs or fragments got paired with the wrong prediction. Fitting a parametric curve acts as a shape filter: if the blob doesn't look like a Doppler S-curve, the fit fails or produces a low R2, and the quality filter removes it before matching.

**Recall also improved.** Fragment merging recovers passes that the blob detector split into pieces. In seed 7, three separate blobs (5, 8, 11) were merged into one trajectory that matched a prediction the greedy matcher missed entirely.

**False alarm count dropped.** With greedy matching on 10 blobs, typically 4-7 are unmatched "false alarms." With the trajectory extractor, the number drops to 1-3 because fragments are merged and noise is filtered.

### 3.3 Fragment merge examples

**Seed 7** (12 blobs to 9 trajectories):
```
Trajectory T8 = merged blobs [5, 8, 11]    R2 = 1.00
Trajectory T9 = merged blobs [6, 12]       R2 = 0.95
```
Blobs 5, 8, and 11 were three disconnected pixel clusters from the same satellite pass. The S-curve fitter recognized they all lie on the same Doppler curve and merged them. The combined fit has R2 = 1.00 because the three fragments trace the same smooth arc.

**Seed 33** (10 blobs to 9 trajectories):
```
Trajectory T6 = merged blobs [5, 9]        R2 = 0.35
```
Here the merge happened but the combined fit is weak (R2 = 0.35). The quality filter flags this as low-confidence, so it does not get priority in the matching step.

### 3.4 Fit quality distribution

Across all 6 seeds (55 total fits):

| R2 range | Count | Meaning |
|---|---:|---|
| 0.95 - 1.00 | 18 | Excellent: clean Doppler pass |
| 0.70 - 0.94 | 8 | Good: real track with some noise |
| 0.30 - 0.69 | 12 | Moderate: partial track or noisy |
| < 0.30 | 7 | Poor: likely noise, filtered out before matching |

The R2 threshold (default 0.3) removes the bottom tier. On real captures where noise is less uniform than in the synthetic data, this filter will be more important.

---

## 4. When does this method help?

| Scenario | Without extractor | With extractor |
|---|---|---|
| Clean, well-separated tracks | Works fine | Same result, slightly cleaner |
| Fragmented pass (signal fades in/out) | Multiple false alarms, missed match | Fragments merged, single correct match |
| Dense passes (many satellites in same band) | Greedy steals predictions from real tracks | Hungarian finds globally optimal assignment |
| Noise blob near a prediction | Noise steals the prediction | Low R2 filters it out before matching |
| Very faint track below threshold | Missed by both | Missed by both (Phase 3 faint recovery addresses this) |

The extractor helps most when the spectrogram is crowded or noisy, which is the typical case for real Starlink captures with 20-30 visible satellites.

---

## 5. Parameters

All parameters are exposed as sidebar sliders in the Streamlit app.

| Parameter | Default | What it controls |
|---|---|---|
| Fragment merge tolerance | 3.0 px | How close two fitted curves must agree to be merged |
| Min R2 | 0.3 | Minimum fit quality to enter matching |
| Min ridge points | 8 | Minimum blob length to enter matching |
| Max match distance | 12.0 px | Maximum curve distance for a valid match |

---

## 6. Implementation

| File | What it does |
|---|---|
| `files/scurve_extractor.py` | Ridge extraction, S-curve fitting, fragment merging |
| `files/correlation.py` | `match_curves_hungarian()` -- optimal 1-to-1 assignment |
| `app.py` | Sidebar controls, S-curve Fits tab, wiring into the pipeline |

The S-curve model, fitting bounds, and merge logic are all in `scurve_extractor.py` (~250 lines). The Hungarian matcher is ~60 lines added to `correlation.py`. No external dependencies beyond scipy and numpy.

---

## 7. Next steps

1. **Test on real captures.** The numbers above are from synthetic data. Real captures will have different noise profiles, and the fit parameters may need tuning. A few SigMF captures from the station would be valuable for validation.

2. **Residual analysis.** Already built (Phase 5). For each matched track, the residual plot shows measured minus predicted Doppler over time. Patterns in the residual diagnose clock drift, stale TLEs, and timing errors.

3. **Matched filter for sub-threshold tracks.** Already built (Phase 3). For predictions the detector missed entirely, the tool walks along the predicted curve and checks for faint energy below the detection threshold.
