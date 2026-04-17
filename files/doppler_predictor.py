"""
Doppler Predictor — Skyfield-based satellite Doppler shift computation.
Extracted and adapted from Jesse Chiu's doppler-predictor GUI for use in
the Streamlit processing pipeline.

Computes predicted Doppler S-curves from TLE data for correlation with
measured spectrograms captured at Sieg Hall.

Reference:  f_doppler = -f_tx * v_radial / c
            v_radial  = d(slant_range)/dt   (finite difference, 1 s step)
"""

import numpy as np
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_M_S = 299792458.0                   # speed of light  [m/s]
DEFAULT_TX_FREQ_HZ = 10.5e9           # Starlink X-band downlink [Hz]
SIEG_HALL = {                          # default ground station
    "latitude":  47.6553,
    "longitude": -122.3035,
    "altitude":  60.0,                 # metres
}


# ---------------------------------------------------------------------------
# Single-satellite predictor
# ---------------------------------------------------------------------------
class DopplerPredictor:
    """Predict Doppler shift for one satellite from its TLE."""

    def __init__(self, tle_line1, tle_line2, ue_location=None,
                 tx_freq_hz=DEFAULT_TX_FREQ_HZ, sat_name=None):
        from skyfield.api import EarthSatellite, load, wgs84

        self.tle_line1 = tle_line1.strip()
        self.tle_line2 = tle_line2.strip()
        self.tx_freq_hz = tx_freq_hz
        self.wgs84 = wgs84

        if sat_name:
            self.sat_name = sat_name
        else:
            try:
                self.sat_name = f"SAT-{self.tle_line1[2:7].strip()}"
            except Exception:
                self.sat_name = "SAT-UNKNOWN"

        loc = ue_location or SIEG_HALL
        self.ue_lat = loc["latitude"]
        self.ue_lon = loc["longitude"]
        self.ue_alt_m = loc.get("altitude", 0.0)

        self.ts = load.timescale()
        self.satellite = EarthSatellite(self.tle_line1, self.tle_line2, ts=self.ts)
        self.observer = wgs84.latlon(self.ue_lat, self.ue_lon,
                                     elevation_m=self.ue_alt_m)

    # -- core computation ---------------------------------------------------
    def _relative_at(self, t_skyfield):
        return (self.satellite - self.observer).at(t_skyfield)

    def position_at(self, dt_utc):
        """Return (az_deg, el_deg, dist_km) at a UTC datetime."""
        from skyfield.api import utc
        t = self.ts.from_datetime(dt_utc.replace(tzinfo=utc)
                                  if dt_utc.tzinfo is None else dt_utc)
        alt, az, dist = self._relative_at(t).altaz()
        return az.degrees, alt.degrees, dist.km

    def doppler_at(self, dt_utc):
        """Return Doppler shift [Hz] at a UTC datetime (finite diff, 1 s)."""
        from skyfield.api import utc
        dt = dt_utc.replace(tzinfo=utc) if dt_utc.tzinfo is None else dt_utc
        t0 = self.ts.from_datetime(dt)
        t1 = self.ts.from_datetime(dt + timedelta(seconds=1))

        pos0 = self._relative_at(t0).position.km
        pos1 = self._relative_at(t1).position.km

        d0 = np.linalg.norm(pos0)
        d1 = np.linalg.norm(pos1)
        range_rate_km_s = d1 - d0  # positive = receding

        return -self.tx_freq_hz * (range_rate_km_s * 1000.0) / C_M_S

    # -- full pass ----------------------------------------------------------
    def compute_pass(self, start_utc, duration_s=600, step_s=1.0,
                     elevation_mask=0.0):
        """
        Compute az / el / dist / doppler / velocity over a time window.

        Returns dict of 1-D numpy arrays keyed by:
            time_s, az_deg, el_deg, dist_km, doppler_hz, velocity_m_s
        Only time-steps where elevation >= elevation_mask are included.
        """
        from skyfield.api import utc
        n = int(duration_s / step_s)
        t0 = start_utc.replace(tzinfo=utc) if start_utc.tzinfo is None else start_utc

        times, az, el, dist, doppler, vel = [], [], [], [], [], []
        for i in range(n):
            dt = t0 + timedelta(seconds=i * step_s)
            a, e, d = self.position_at(dt)
            if e >= elevation_mask:
                dop = self.doppler_at(dt)
                times.append(i * step_s)
                az.append(a)
                el.append(e)
                dist.append(d)
                doppler.append(dop)
                vel.append(-dop * C_M_S / self.tx_freq_hz)  # m/s

        return {k: np.array(v) for k, v in [
            ("time_s", times), ("az_deg", az), ("el_deg", el),
            ("dist_km", dist), ("doppler_hz", doppler), ("velocity_m_s", vel),
        ]}

    def build_waterfall(self, pass_data, n_vel_bins=300, vel_range_km_s=5.0,
                        sigma_km_s=0.05):
        """
        Build a 2-D velocity waterfall from pass data (matching Jesse's format).

        Returns:
            waterfall  (n_time, n_vel_bins)  — FSPL-coloured signal intensity
            vel_axis   (n_vel_bins,)         — velocity bin centres [km/s]
            time_axis  (n_time,)             — time offsets [s]
        """
        vel_axis = np.linspace(-vel_range_km_s, vel_range_km_s, n_vel_bins)
        t = pass_data["time_s"]
        v_km = pass_data["velocity_m_s"] / 1000.0
        d_km = pass_data["dist_km"]
        n_time = len(t)

        waterfall = np.full((n_time, n_vel_bins), 250.0)
        for i in range(n_time):
            if d_km[i] > 0:
                fspl = (20 * np.log10(d_km[i] * 1000)
                        + 20 * np.log10(self.tx_freq_hz)
                        + 20 * np.log10(4 * np.pi / C_M_S))
                gauss = np.exp(-0.5 * ((vel_axis - v_km[i]) / sigma_km_s) ** 2)
                waterfall[i] = fspl * (1 - 0.9 * gauss) + 250.0 * (1 - gauss)

        return waterfall, vel_axis, t


# ---------------------------------------------------------------------------
# Multi-satellite loader
# ---------------------------------------------------------------------------
def load_tle_file(path):
    """Parse a 3-line TLE file into [(name, line1, line2), ...]."""
    with open(path) as f:
        lines = [l.rstrip() for l in f if l.strip()]
    entries = []
    i = 0
    while i < len(lines) - 2:
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            entries.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return entries


def find_visible_satellites(tle_entries, obs_time, ue_location=None,
                            elevation_mask=10.0, tx_freq_hz=DEFAULT_TX_FREQ_HZ,
                            max_sats=None):
    """
    Scan TLE entries and return DopplerPredictor objects for satellites
    above the elevation mask at *obs_time*.
    """
    visible = []
    loc = ue_location or SIEG_HALL
    for name, l1, l2 in tle_entries:
        try:
            dp = DopplerPredictor(l1, l2, loc, tx_freq_hz=tx_freq_hz,
                                  sat_name=name)
            _, el, _ = dp.position_at(obs_time)
            if el >= elevation_mask:
                visible.append(dp)
                if max_sats and len(visible) >= max_sats:
                    break
        except Exception:
            continue
    return visible


# ---------------------------------------------------------------------------
# Synthetic predicted S-curve (for demo without TLE / Skyfield)
# ---------------------------------------------------------------------------
def generate_synthetic_prediction(n_time=512, n_freq=256, n_tracks=10, seed=42):
    """
    Generate synthetic predicted Doppler S-curves that can be overlaid on
    a measured spectrogram for correlation demo.

    Returns list of dicts with keys: time_bins, freq_bins, velocity_m_s, label
    """
    rng = np.random.RandomState(seed)
    predictions = []
    t = np.arange(n_time)

    for i in range(n_tracks):
        freq_center = rng.uniform(30, n_freq - 30)
        freq_amp = rng.uniform(10, 40)
        phase = rng.uniform(-np.pi, np.pi)
        dur_frac = rng.uniform(0.3, 0.8)

        t_norm = np.linspace(-np.pi, np.pi, n_time)
        freq_curve = freq_center + freq_amp * np.sin(t_norm + phase)

        vis_start = int(n_time * (1 - dur_frac) / 2 * rng.uniform(0.2, 1.8))
        vis_end = min(n_time, vis_start + int(n_time * dur_frac * rng.uniform(0.3, 1.0)))

        t_vis = t[vis_start:vis_end]
        f_vis = freq_curve[vis_start:vis_end]

        # Fake velocity (linear ramp — approaching → receding)
        vel = np.linspace(-6000, 6000, len(t_vis)) + rng.uniform(-500, 500)

        predictions.append({
            "label": f"STARLINK-{1000 + i}",
            "time_bins": t_vis.astype(float),
            "freq_bins": f_vis,
            "velocity_m_s": vel,
        })
    return predictions
