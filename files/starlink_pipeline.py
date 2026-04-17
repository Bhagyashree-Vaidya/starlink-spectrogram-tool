"""
Starlink Spectrogram Processing Pipeline — Prototype v1
=========================================================
For: FunLab, Prof. Sumit Roy & Jesse Chiu, UW ECE
By: Bhagyashree Vaidya
Date: March 2026

This module generates synthetic spectrogram data mimicking Starlink
LEO satellite captures from the Sieg Hall receiver, and implements
the processing pipeline discussed in the March 24 meeting:

1. Generate synthetic spectrograms with leakage + satellite tracks + noise
2. Remove signal leakage band
3. Detect and count satellite tracks
4. Extract track parameters for correspondence matching

Usage:
    python starlink_pipeline.py
"""

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.morphology import remove_small_objects, thin
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os
from datetime import datetime


# =============================================================================
# SECTION 1: Synthetic Data Generation
# =============================================================================

def generate_satellite_track(time_axis, freq_center, freq_amplitude, phase, duration_fraction=0.7):
    """
    Generate a single satellite track as an S-curve in time-frequency space.

    LEO satellites create Doppler-shifted tracks that appear as S-curves
    in spectrograms due to their relative motion overhead.

    Parameters:
        time_axis: 1D array of time bins
        freq_center: Center frequency of the track (Hz offset)
        freq_amplitude: Amplitude of the S-curve (frequency deviation)
        phase: Phase offset (radians) — shifts the curve in time
        duration_fraction: What fraction of the time window the satellite is visible

    Returns:
        track_freq: Frequency values at each time step (NaN where satellite not visible)
        track_power: Signal power at each time step
    """
    n_time = len(time_axis)
    t_norm = np.linspace(-np.pi, np.pi, n_time)

    # S-curve shape from Doppler shift: frequency changes as satellite approaches,
    # passes overhead, and recedes
    track_freq = freq_center + freq_amplitude * np.sin(t_norm + phase)

    # Satellite is only visible for a portion of the time window
    visible_start = int(n_time * (1 - duration_fraction) / 2 * np.random.uniform(0.2, 1.8))
    visible_end = min(n_time, visible_start + int(n_time * duration_fraction * np.random.uniform(0.3, 1.0)))

    # Signal power varies — stronger when satellite is closer (near center of pass)
    t_visible = np.linspace(-1, 1, visible_end - visible_start)
    power_envelope = np.exp(-0.5 * t_visible**2 / 0.4**2)  # Gaussian envelope

    track_power = np.zeros(n_time)
    track_power[visible_start:visible_end] = power_envelope * np.random.uniform(0.3, 1.0)

    # Mask frequencies outside visible window
    track_freq_masked = np.full(n_time, np.nan)
    track_freq_masked[visible_start:visible_end] = track_freq[visible_start:visible_end]

    return track_freq_masked, track_power


def generate_synthetic_spectrogram(
    n_time=512,
    n_freq=256,
    n_satellites=8,
    leakage_freq_bin=128,
    leakage_width=6,
    leakage_power=50.0,
    noise_floor=-20.0,
    noise_std=3.0,
    track_width=2,
    seed=42
):
    """
    Generate a synthetic spectrogram mimicking Starlink captures from Sieg Hall.

    Includes:
    - Background noise floor
    - Bright signal leakage band (the "useless" bright line Prof Roy described)
    - Multiple faint satellite tracks (S-curves from LEO passes)
    - Random interference artifacts

    Returns:
        spectrogram: 2D array (n_freq x n_time) — power spectral density
        metadata: dict with ground truth about tracks and leakage
    """
    rng = np.random.RandomState(seed)

    # Background noise
    spectrogram = noise_floor + noise_std * rng.randn(n_freq, n_time)

    # Signal leakage band — bright horizontal band that obscures everything
    # (This is what Prof Roy said is "useless" and "screwing up the display")
    leakage_profile = leakage_power * np.exp(
        -0.5 * (np.arange(n_freq) - leakage_freq_bin)**2 / (leakage_width/2)**2
    )
    # Add some variation along time axis
    leakage_variation = 1.0 + 0.1 * rng.randn(n_time)
    spectrogram += leakage_profile[:, np.newaxis] * leakage_variation[np.newaxis, :]

    # Add a secondary leakage sidelobe (realistic artifact)
    sidelobe_offset = rng.randint(15, 30)
    sidelobe_power = leakage_power * 0.15
    sidelobe_profile = sidelobe_power * np.exp(
        -0.5 * (np.arange(n_freq) - (leakage_freq_bin + sidelobe_offset))**2 / (3)**2
    )
    spectrogram += sidelobe_profile[:, np.newaxis]

    # Satellite tracks
    time_axis = np.arange(n_time)
    tracks_metadata = []

    for i in range(n_satellites):
        # Each satellite has different orbital parameters
        freq_center = rng.uniform(30, n_freq - 30)
        freq_amplitude = rng.uniform(10, 40)
        phase = rng.uniform(-np.pi, np.pi)
        track_power_scale = rng.uniform(3.0, 12.0)  # Faint! Much weaker than leakage

        track_freq, track_power = generate_satellite_track(
            time_axis, freq_center, freq_amplitude, phase,
            duration_fraction=rng.uniform(0.3, 0.8)
        )

        # Paint the track onto the spectrogram
        for t in range(n_time):
            if not np.isnan(track_freq[t]) and track_power[t] > 0.05:
                f_center = int(np.clip(track_freq[t], 0, n_freq - 1))
                # Track has finite width in frequency
                for df in range(-track_width, track_width + 1):
                    f_idx = f_center + df
                    if 0 <= f_idx < n_freq:
                        weight = np.exp(-0.5 * df**2 / (track_width/2)**2)
                        spectrogram[f_idx, t] += track_power_scale * track_power[t] * weight

        # Store ground truth
        visible_mask = ~np.isnan(track_freq)
        if visible_mask.any():
            tracks_metadata.append({
                'satellite_id': f'STARLINK-{1000 + i}',
                'freq_center': float(freq_center),
                'freq_amplitude': float(freq_amplitude),
                'phase': float(phase),
                'power_scale': float(track_power_scale),
                'visible_time_start': int(np.argmax(visible_mask)),
                'visible_time_end': int(n_time - np.argmax(visible_mask[::-1])),
                'track_frequencies': track_freq[visible_mask].tolist(),
                'track_times': time_axis[visible_mask].tolist()
            })

    # A few random interference blips (real data has these)
    n_blips = rng.randint(3, 8)
    for _ in range(n_blips):
        blip_t = rng.randint(0, n_time)
        blip_f = rng.randint(0, n_freq)
        blip_power = rng.uniform(5, 15)
        blip_size = rng.randint(2, 6)
        f_lo = max(0, blip_f - blip_size)
        f_hi = min(n_freq, blip_f + blip_size)
        t_lo = max(0, blip_t - blip_size)
        t_hi = min(n_time, blip_t + blip_size)
        spectrogram[f_lo:f_hi, t_lo:t_hi] += blip_power

    metadata = {
        'n_time': n_time,
        'n_freq': n_freq,
        'n_satellites': n_satellites,
        'leakage_freq_bin': leakage_freq_bin,
        'leakage_width': leakage_width,
        'noise_floor_db': noise_floor,
        'tracks': tracks_metadata,
        'timestamp': datetime.now().isoformat(),
        'location': 'Sieg Hall, UW Seattle (47.6553 N, 122.3035 W)',
        'note': 'SYNTHETIC DATA — for prototype demonstration only'
    }

    return spectrogram, metadata


# =============================================================================
# SECTION 2: Leakage Removal
# =============================================================================

def detect_leakage_band(spectrogram, percentile_threshold=95):
    """
    Detect the leakage band by finding frequency bins with abnormally high
    mean power across time.

    The leakage band appears as a bright horizontal stripe — its mean power
    across time is far higher than any satellite track (which is localized in time).
    """
    mean_power_per_freq = np.mean(spectrogram, axis=1)
    threshold = np.percentile(mean_power_per_freq, percentile_threshold)
    leakage_mask = mean_power_per_freq > threshold

    # Dilate the mask to catch sidelobes
    leakage_mask = ndimage.binary_dilation(leakage_mask, iterations=3)

    return leakage_mask


def remove_leakage(spectrogram, leakage_mask=None, method='interpolate'):
    """
    Remove the signal leakage band from the spectrogram.

    Methods:
    - 'zero': Set leakage bins to noise floor (simple but leaves a gap)
    - 'interpolate': Replace leakage with interpolated values from adjacent bins
    - 'median': Replace with median of surrounding non-leakage bins
    """
    if leakage_mask is None:
        leakage_mask = detect_leakage_band(spectrogram)

    cleaned = spectrogram.copy()

    if method == 'zero':
        noise_floor = np.median(spectrogram[~leakage_mask, :])
        noise_std = np.std(spectrogram[~leakage_mask, :]) * 0.5
        cleaned[leakage_mask, :] = noise_floor + noise_std * np.random.randn(
            np.sum(leakage_mask), spectrogram.shape[1]
        )

    elif method == 'interpolate':
        leakage_indices = np.where(leakage_mask)[0]
        clean_indices = np.where(~leakage_mask)[0]

        if len(clean_indices) > 0 and len(leakage_indices) > 0:
            for t in range(spectrogram.shape[1]):
                cleaned[leakage_indices, t] = np.interp(
                    leakage_indices, clean_indices, spectrogram[clean_indices, t]
                )

    elif method == 'median':
        noise_median = np.median(spectrogram[~leakage_mask, :], axis=0)
        noise_std = np.std(spectrogram[~leakage_mask, :]) * 0.3
        for idx in np.where(leakage_mask)[0]:
            cleaned[idx, :] = noise_median + noise_std * np.random.randn(spectrogram.shape[1])

    return cleaned, leakage_mask


# =============================================================================
# SECTION 3: Track Detection
# =============================================================================

def enhance_tracks(spectrogram, sigma=1.5):
    """
    Enhance satellite tracks by applying contrast normalization and
    directional filtering. Tracks are curves in the spectrogram, so
    they respond well to ridge/edge enhancement.
    """
    from scipy.ndimage import gaussian_filter, median_filter

    # Subtract local background (makes faint tracks stand out)
    background = median_filter(spectrogram, size=15)
    enhanced = spectrogram - background

    # Clip negative values (we only care about signals above background)
    enhanced = np.clip(enhanced, 0, None)

    # Gentle smoothing to reduce noise while preserving track shapes
    enhanced = gaussian_filter(enhanced, sigma=sigma)

    # Normalize to 0-1
    if enhanced.max() > enhanced.min():
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

    return enhanced


def detect_tracks(spectrogram, min_track_length=20, power_threshold=None):
    """
    Detect satellite tracks in a cleaned spectrogram.

    Pipeline:
    1. Enhance tracks (background subtraction + smoothing)
    2. Threshold to create binary image
    3. Label connected components
    4. Filter by size (real tracks are long, noise blips are small)
    5. Extract track properties

    Returns:
        track_labels: 2D array with integer labels for each detected track
        track_properties: List of dicts with track parameters
    """
    # Step 1: Enhance
    enhanced = enhance_tracks(spectrogram)

    # Step 2: Threshold
    if power_threshold is None:
        # Adaptive threshold: mean + 2*std of the enhanced image
        power_threshold = np.mean(enhanced) + 2.0 * np.std(enhanced)

    binary = enhanced > power_threshold

    # Step 3: Clean up — remove small noise blobs
    binary = remove_small_objects(binary, min_size=min_track_length)

    # Optional: slight dilation to connect nearby track segments
    binary = ndimage.binary_dilation(binary, iterations=1)
    binary = ndimage.binary_erosion(binary, iterations=1)

    # Step 4: Label connected components
    track_labels, n_tracks = label(binary, return_num=True)

    # Step 5: Extract properties
    regions = regionprops(track_labels, intensity_image=enhanced)

    track_properties = []
    for region in regions:
        # Filter: real tracks are elongated (major_axis >> minor_axis)
        if region.axis_major_length < min_track_length:
            continue

        # Get track centroid and bounding box
        coords = region.coords  # (row, col) = (freq, time)

        track_info = {
            'track_id': region.label,
            'centroid_freq': float(region.centroid[0]),
            'centroid_time': float(region.centroid[1]),
            'bbox': {
                'freq_min': int(region.bbox[0]),
                'time_min': int(region.bbox[1]),
                'freq_max': int(region.bbox[2]),
                'time_max': int(region.bbox[3]),
            },
            'area_pixels': int(region.area),
            'length': float(region.axis_major_length),
            'width': float(region.axis_minor_length),
            'orientation_deg': float(np.degrees(region.orientation)),
            'mean_power': float(region.intensity_mean),
            'max_power': float(region.intensity_max),
            'eccentricity': float(region.eccentricity),
            'n_pixels': len(coords),
            'freq_range': [int(coords[:, 0].min()), int(coords[:, 0].max())],
            'time_range': [int(coords[:, 1].min()), int(coords[:, 1].max())],
        }

        track_properties.append(track_info)

    # Sort by length (longest tracks first — more likely to be real satellites)
    track_properties.sort(key=lambda x: x['length'], reverse=True)

    # Re-number
    for i, tp in enumerate(track_properties):
        tp['track_id'] = i + 1

    return track_labels, track_properties, enhanced


# =============================================================================
# SECTION 4: Visualization
# =============================================================================

def plot_pipeline_results(spectrogram, cleaned, leakage_mask, track_labels,
                          track_properties, enhanced, metadata, save_path=None):
    """
    Create a comprehensive visualization of the processing pipeline.
    4-panel plot: Raw → Cleaned → Enhanced → Detected Tracks
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Starlink Spectrogram Processing Pipeline — Prototype v1\n"
        f"{metadata.get('note', '')} | {len(track_properties)} tracks detected",
        fontsize=14, fontweight='bold', y=0.98
    )

    cmap = 'viridis'

    # Panel 1: Raw spectrogram
    ax = axes[0, 0]
    vmin, vmax = np.percentile(spectrogram, [2, 98])
    im1 = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap=cmap,
                     vmin=vmin, vmax=vmax)
    ax.set_title('1. Raw Spectrogram\n(leakage band dominates)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Frequency (bins)')
    plt.colorbar(im1, ax=ax, label='Power (dB)', shrink=0.8)

    leakage_rows = np.where(leakage_mask)[0]
    if len(leakage_rows) > 0:
        ax.axhspan(leakage_rows.min(), leakage_rows.max(),
                   alpha=0.3, color='red', label='Leakage band')
        ax.legend(loc='upper right', fontsize=8)

    # Panel 2: Cleaned spectrogram
    ax = axes[0, 1]
    vmin2, vmax2 = np.percentile(cleaned, [2, 98])
    im2 = ax.imshow(cleaned, aspect='auto', origin='lower', cmap=cmap,
                     vmin=vmin2, vmax=vmax2)
    ax.set_title('2. After Leakage Removal\n(tracks now visible)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Frequency (bins)')
    plt.colorbar(im2, ax=ax, label='Power (dB)', shrink=0.8)

    # Panel 3: Enhanced
    ax = axes[1, 0]
    im3 = ax.imshow(enhanced, aspect='auto', origin='lower', cmap='hot',
                     vmin=0, vmax=np.percentile(enhanced, 99))
    ax.set_title('3. Track Enhancement\n(background subtracted, contrast boosted)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Frequency (bins)')
    plt.colorbar(im3, ax=ax, label='Enhanced power', shrink=0.8)

    # Panel 4: Detected tracks with labels
    ax = axes[1, 1]
    ax.imshow(cleaned, aspect='auto', origin='lower', cmap='gray_r',
              vmin=vmin2, vmax=vmax2, alpha=0.5)

    track_overlay = np.ma.masked_where(track_labels == 0, track_labels)
    n_tracks = len(track_properties)
    if n_tracks > 0:
        colors = plt.cm.Set1(np.linspace(0, 1, max(n_tracks, 1)))
        track_cmap = mcolors.ListedColormap(colors[:n_tracks])
        ax.imshow(track_overlay, aspect='auto', origin='lower',
                  cmap=track_cmap, alpha=0.7)

    for tp in track_properties:
        ax.annotate(
            f"Track {tp['track_id']}",
            xy=(tp['centroid_time'], tp['centroid_freq']),
            fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
            ha='center'
        )

    ax.set_title(f'4. Detected Tracks: {n_tracks} found\n(labeled and counted)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Frequency (bins)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved pipeline visualization to: {save_path}")

    return fig


def plot_track_detail(spectrogram, track_properties, track_labels, save_path=None):
    """
    Detailed view of each detected track with statistics.
    """
    n_tracks = len(track_properties)
    if n_tracks == 0:
        print("No tracks detected.")
        return None

    fig, axes = plt.subplots(1, min(n_tracks, 6), figsize=(4 * min(n_tracks, 6), 4))
    if n_tracks == 1:
        axes = [axes]

    fig.suptitle('Individual Track Details', fontsize=13, fontweight='bold')

    for i, (tp, ax) in enumerate(zip(track_properties[:6], axes)):
        bbox = tp['bbox']
        margin = 10
        f_lo = max(0, bbox['freq_min'] - margin)
        f_hi = min(spectrogram.shape[0], bbox['freq_max'] + margin)
        t_lo = max(0, bbox['time_min'] - margin)
        t_hi = min(spectrogram.shape[1], bbox['time_max'] + margin)

        patch = spectrogram[f_lo:f_hi, t_lo:t_hi]
        ax.imshow(patch, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Track {tp['track_id']}\n"
                     f"Length: {tp['length']:.0f}px\n"
                     f"Power: {tp['mean_power']:.3f}",
                     fontsize=9)
        ax.set_xlabel('Time')
        ax.set_ylabel('Freq')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# SECTION 5: Main Pipeline
# =============================================================================

def run_pipeline(seed=42, n_satellites=10, save_dir='output'):
    """
    Run the complete Starlink spectrogram processing pipeline.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("STARLINK SPECTROGRAM PROCESSING PIPELINE — Prototype v1")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\n[1/4] Generating synthetic spectrogram...")
    spectrogram, metadata = generate_synthetic_spectrogram(
        n_time=512, n_freq=256, n_satellites=n_satellites, seed=seed
    )
    print(f"  Shape: {spectrogram.shape} (freq x time)")
    print(f"  Satellites injected: {metadata['n_satellites']}")
    print(f"  Power range: {spectrogram.min():.1f} to {spectrogram.max():.1f} dB")

    np.save(os.path.join(save_dir, 'raw_spectrogram.npy'), spectrogram)
    with open(os.path.join(save_dir, 'ground_truth.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Step 2: Remove leakage
    print("\n[2/4] Removing signal leakage...")
    cleaned, leakage_mask = remove_leakage(spectrogram, method='interpolate')
    n_leakage_bins = np.sum(leakage_mask)
    print(f"  Leakage band detected: {n_leakage_bins} frequency bins masked")
    print(f"  Cleaned power range: {cleaned.min():.1f} to {cleaned.max():.1f} dB")

    np.save(os.path.join(save_dir, 'cleaned_spectrogram.npy'), cleaned)

    # Step 3: Detect tracks
    print("\n[3/4] Detecting satellite tracks...")
    track_labels, track_properties, enhanced = detect_tracks(cleaned, min_track_length=15)
    print(f"  Tracks detected: {len(track_properties)}")
    print(f"  Ground truth satellites: {metadata['n_satellites']}")

    for tp in track_properties:
        print(f"    Track {tp['track_id']}: length={tp['length']:.0f}px, "
              f"power={tp['mean_power']:.3f}, "
              f"freq=[{tp['freq_range'][0]}-{tp['freq_range'][1]}], "
              f"time=[{tp['time_range'][0]}-{tp['time_range'][1]}]")

    with open(os.path.join(save_dir, 'detected_tracks.json'), 'w') as f:
        json.dump({
            'n_detected': len(track_properties),
            'n_ground_truth': metadata['n_satellites'],
            'tracks': track_properties
        }, f, indent=2)

    # Step 4: Visualize
    print("\n[4/4] Generating visualizations...")
    fig1 = plot_pipeline_results(
        spectrogram, cleaned, leakage_mask, track_labels,
        track_properties, enhanced, metadata,
        save_path=os.path.join(save_dir, 'pipeline_overview.png')
    )

    fig2 = plot_track_detail(
        cleaned, track_properties, track_labels,
        save_path=os.path.join(save_dir, 'track_details.png')
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Input: {spectrogram.shape[0]}x{spectrogram.shape[1]} spectrogram")
    print(f"  Leakage removed: {n_leakage_bins} freq bins")
    print(f"  Tracks detected: {len(track_properties)} / {metadata['n_satellites']} ground truth")
    print(f"  Output files in: {save_dir}/")

    return spectrogram, cleaned, track_labels, track_properties, metadata


if __name__ == '__main__':
    spectrogram, cleaned, track_labels, track_properties, metadata = run_pipeline(
        seed=42, n_satellites=10, save_dir='output'
    )
    plt.show()
