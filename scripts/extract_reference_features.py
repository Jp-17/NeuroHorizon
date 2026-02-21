"""
Reference Feature Extraction for IDEncoder

Extracts per-unit statistical features from spike data in HDF5 files.
These features serve as input to the IDEncoder, which generates unit
embeddings without requiring per-unit learnable parameters.

Features extracted per unit (~33 dimensions):
- Mean firing rate (1d)
- ISI coefficient of variation (1d)
- ISI log-histogram (20d)
- Autocorrelation features (10d)
- Fano factor (1d)

Usage:
    conda run -n poyo python scripts/extract_reference_features.py \
        --data_dir /path/to/ibl_processed \
        --ref_window 60.0
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Feature dimensions
N_ISI_BINS = 20
N_AUTOCORR_BINS = 10
TOTAL_FEATURE_DIM = 1 + 1 + N_ISI_BINS + N_AUTOCORR_BINS + 1  # = 33


def compute_unit_features(
    spike_times: np.ndarray,
    spike_unit_idx: np.ndarray,
    n_units: int,
    ref_start: float,
    ref_end: float,
    bin_size_ms: float = 20.0,
) -> np.ndarray:
    """Compute reference features for all units in a session.

    Args:
        spike_times: All spike timestamps (sorted)
        spike_unit_idx: Unit index for each spike
        n_units: Total number of units
        ref_start: Reference window start time (seconds)
        ref_end: Reference window end time (seconds)
        bin_size_ms: Bin size in ms for Fano factor computation

    Returns:
        features: (n_units, TOTAL_FEATURE_DIM) array of float64
    """
    ref_duration = ref_end - ref_start
    features = np.zeros((n_units, TOTAL_FEATURE_DIM), dtype=np.float64)

    # Filter spikes to reference window
    mask = (spike_times >= ref_start) & (spike_times < ref_end)
    ref_times = spike_times[mask]
    ref_idx = spike_unit_idx[mask]

    for u in range(n_units):
        unit_times = ref_times[ref_idx == u]
        n_spikes = len(unit_times)
        feat = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float64)

        # 1. Mean firing rate (1d)
        firing_rate = n_spikes / ref_duration if ref_duration > 0 else 0.0
        feat[0] = firing_rate

        if n_spikes < 2:
            features[u] = feat
            continue

        # 2. ISI coefficient of variation (1d)
        isis = np.diff(unit_times)
        isis = isis[isis > 0]  # remove zero ISIs
        if len(isis) > 1:
            isi_mean = np.mean(isis)
            isi_std = np.std(isis)
            feat[1] = isi_std / isi_mean if isi_mean > 0 else 0.0
        else:
            feat[1] = 0.0

        # 3. ISI log-histogram (20d)
        if len(isis) > 0:
            log_isis = np.log10(np.clip(isis, 1e-4, 10.0))  # clip to [0.1ms, 10s]
            hist, _ = np.histogram(log_isis, bins=N_ISI_BINS, range=(-4, 1))
            hist = hist.astype(np.float64)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum  # normalize to probability
            feat[2 : 2 + N_ISI_BINS] = hist

        # 4. Autocorrelation features (10d)
        # Compute autocorrelation of binned spike counts
        bin_size_s = 0.01  # 10ms bins for autocorrelation
        n_bins_auto = int(ref_duration / bin_size_s)
        if n_bins_auto > 10 and n_spikes > 10:
            bin_edges = np.linspace(ref_start, ref_end, n_bins_auto + 1)
            counts, _ = np.histogram(unit_times, bins=bin_edges)
            counts = counts.astype(np.float64)
            counts_centered = counts - counts.mean()
            var = np.var(counts)
            if var > 0:
                # Compute autocorrelation for lags 1..N_AUTOCORR_BINS
                autocorr = np.zeros(N_AUTOCORR_BINS)
                for lag in range(1, N_AUTOCORR_BINS + 1):
                    if lag < len(counts_centered):
                        autocorr[lag - 1] = (
                            np.mean(counts_centered[:-lag] * counts_centered[lag:]) / var
                        )
                feat[2 + N_ISI_BINS : 2 + N_ISI_BINS + N_AUTOCORR_BINS] = autocorr

        # 5. Fano factor (1d)
        bin_size_s_fano = bin_size_ms / 1000.0
        n_bins_fano = int(ref_duration / bin_size_s_fano)
        if n_bins_fano > 1:
            bin_edges_fano = np.linspace(ref_start, ref_end, n_bins_fano + 1)
            counts_fano, _ = np.histogram(unit_times, bins=bin_edges_fano)
            counts_fano = counts_fano.astype(np.float64)
            mean_count = np.mean(counts_fano)
            if mean_count > 0:
                feat[-1] = np.var(counts_fano) / mean_count
            else:
                feat[-1] = 0.0

        features[u] = feat

    return features


def process_file(filepath: Path, ref_window: float) -> bool:
    """Extract reference features and add them to an existing HDF5 file.

    Args:
        filepath: Path to HDF5 file
        ref_window: Duration of reference window in seconds (from session start)

    Returns:
        True if successful
    """
    try:
        with h5py.File(filepath, "a") as f:
            # Check if features already exist
            if "reference_features" in f["units"]:
                logger.info(f"  {filepath.name}: reference_features already exist, skipping")
                return True

            # Load spike data
            spike_times = f["spikes"]["timestamps"][:]
            spike_unit_idx = f["spikes"]["unit_index"][:]
            n_units = len(f["units"]["id"][:])

            # Determine reference window
            domain_start = float(f["domain"]["start"][0])
            domain_end = float(f["domain"]["end"][0])
            session_duration = domain_end - domain_start

            ref_end = domain_start + min(ref_window, session_duration * 0.5)

            logger.info(
                f"  Reference window: [{domain_start:.1f}, {ref_end:.1f}]s "
                f"({ref_end - domain_start:.1f}s of {session_duration:.1f}s session)"
            )

            # Compute features
            features = compute_unit_features(
                spike_times, spike_unit_idx, n_units, domain_start, ref_end
            )

            # Store features
            f["units"].create_dataset(
                "reference_features",
                data=features,
                dtype=np.float64,
            )

            # Log statistics
            firing_rates = features[:, 0]
            logger.info(
                f"  {filepath.name}: {n_units} units, "
                f"FR range=[{firing_rates.min():.2f}, {firing_rates.max():.2f}] Hz, "
                f"mean={firing_rates.mean():.2f} Hz"
            )

        return True

    except Exception as e:
        logger.error(f"  Failed to process {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract IDEncoder reference features from HDF5 files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing HDF5 files",
    )
    parser.add_argument(
        "--ref_window",
        type=float,
        default=60.0,
        help="Reference window duration in seconds (default: 60s)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    h5_files = sorted(data_dir.glob("*.h5"))

    if not h5_files:
        logger.error(f"No .h5 files found in {data_dir}")
        return

    logger.info(f"Extracting reference features for {len(h5_files)} files (ref_window={args.ref_window}s)")

    success = 0
    for filepath in h5_files:
        logger.info(f"Processing {filepath.name}...")
        if process_file(filepath, args.ref_window):
            success += 1

    logger.info(f"Done: {success}/{len(h5_files)} files processed")


if __name__ == "__main__":
    main()
