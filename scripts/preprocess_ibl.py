"""
IBL Data Preprocessing Script

Converts downloaded IBL raw data into HDF5 format compatible with
torch_brain.dataset.Dataset (temporaldata.Data).

Each session becomes one .h5 file with the following schema:
- session.id: str
- domain: Interval (start, end)
- spikes.timestamps: float64 array
- spikes.unit_index: int64 array (local indices 0..N-1)
- units.id: str array (globally unique unit IDs)
- units.brain_region: str array (brain region per unit)
- units.quality_label: int array (1=good, 0=other)
- behavior.timestamps: float64 array (wheel timestamps)
- behavior.wheel_velocity: float64 array
- trials.start: float64 array
- trials.end: float64 array
- trials.stim_on: float64 array
- trials.choice: int array
- trials.contrast_left: float64 array
- trials.contrast_right: float64 array
- trials.feedback_type: int array

Usage:
    conda run -n poyo python scripts/preprocess_ibl.py --input_dir /path/to/ibl_raw --output_dir /path/to/output
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_wheel_velocity(wheel_pos, wheel_ts, smooth_window=0.05):
    """Compute wheel velocity from position using finite differences.

    Args:
        wheel_pos: Wheel position array
        wheel_ts: Wheel timestamps array
        smooth_window: Smoothing window in seconds (default: 50ms)

    Returns:
        velocity: Wheel velocity array (same length as input)
        timestamps: Same as wheel_ts
    """
    dt = np.diff(wheel_ts)
    dp = np.diff(wheel_pos)

    # Avoid division by zero
    dt[dt == 0] = 1e-6
    velocity = dp / dt

    # Pad to same length
    velocity = np.concatenate([velocity, [velocity[-1]]])

    # Simple smoothing via uniform filter
    if smooth_window > 0 and len(velocity) > 1:
        median_dt = np.median(dt)
        if median_dt > 0:
            kernel_size = max(1, int(smooth_window / median_dt))
            if kernel_size > 1 and kernel_size < len(velocity):
                kernel = np.ones(kernel_size) / kernel_size
                velocity = np.convolve(velocity, kernel, mode="same")

    return velocity.astype(np.float64)


def process_session(one, eid: str, output_dir: Path, min_good_units: int = 10):
    """Process a single IBL session into HDF5 format.

    Args:
        one: ONE API instance
        eid: Session ID
        output_dir: Output directory for HDF5 files
        min_good_units: Minimum number of good quality units to include session

    Returns:
        True if successful, False otherwise
    """
    output_file = output_dir / f"{eid}.h5"
    if output_file.exists():
        logger.info(f"  {eid}.h5 already exists, skipping")
        return True

    try:
        # Load spike data directly via ONE API
        # Get probe insertions for this session
        pids = one.alyx.rest("insertions", "list", session=eid)
        if not pids:
            logger.warning(f"  No probe insertions for {eid}")
            return False

        all_spike_times = []
        all_spike_unit_idx = []
        all_unit_ids = []
        all_unit_brain_regions = []
        all_unit_quality = []
        unit_offset = 0

        for pid_info in pids:
            probe_name = pid_info.get("name", "probe00")

            try:
                # Load spike sorting data
                spike_times_probe = one.load_dataset(eid, "spikes.times", collection=f"alf/{probe_name}/pykilosort")
                spike_clusters_probe = one.load_dataset(eid, "spikes.clusters", collection=f"alf/{probe_name}/pykilosort")
            except Exception:
                # Try without specifying collection (let ONE find it)
                try:
                    spike_times_probe = one.load_dataset(eid, "spikes.times")
                    spike_clusters_probe = one.load_dataset(eid, "spikes.clusters")
                except Exception as e:
                    logger.warning(f"  Failed to load spikes for {probe_name}: {e}")
                    continue

            if spike_times_probe is None or len(spike_times_probe) == 0:
                continue

            # Try to load cluster quality labels
            try:
                cluster_metrics = one.load_dataset(eid, "clusters.metrics", collection=f"alf/{probe_name}/pykilosort")
                if hasattr(cluster_metrics, 'label'):
                    good_mask = cluster_metrics['label'] == 1
                    good_cluster_ids = np.where(good_mask)[0]
                else:
                    # Use all clusters if no label
                    unique_clusters = np.unique(spike_clusters_probe)
                    good_cluster_ids = unique_clusters
            except Exception:
                # No quality labels, use all clusters
                unique_clusters = np.unique(spike_clusters_probe)
                good_cluster_ids = unique_clusters

            if len(good_cluster_ids) == 0:
                continue

            # Filter spikes to good clusters
            spike_in_good = np.isin(spike_clusters_probe, good_cluster_ids)
            filtered_spike_times = spike_times_probe[spike_in_good]
            filtered_spike_clusters = spike_clusters_probe[spike_in_good]

            # Remap cluster IDs to contiguous local indices
            cluster_to_local = {int(c): i for i, c in enumerate(good_cluster_ids)}
            filtered_spike_local_idx = np.array(
                [cluster_to_local[int(c)] + unit_offset for c in filtered_spike_clusters]
            )

            # Unit IDs: globally unique (eid/probe/cluster_id)
            unit_ids = [f"{eid}/{probe_name}/cluster_{int(c)}" for c in good_cluster_ids]
            brain_regions = ["unknown"] * len(good_cluster_ids)

            all_spike_times.append(filtered_spike_times)
            all_spike_unit_idx.append(filtered_spike_local_idx)
            all_unit_ids.extend(unit_ids)
            all_unit_brain_regions.extend(brain_regions)
            all_unit_quality.extend([1] * len(good_cluster_ids))
            unit_offset += len(good_cluster_ids)

        if unit_offset < min_good_units:
            logger.warning(
                f"  Only {unit_offset} good units for {eid} (min: {min_good_units}), skipping"
            )
            return False

        # Concatenate all probes
        spike_times = np.concatenate(all_spike_times)
        spike_unit_idx = np.concatenate(all_spike_unit_idx)

        # Sort by time
        sort_idx = np.argsort(spike_times)
        spike_times = spike_times[sort_idx]
        spike_unit_idx = spike_unit_idx[sort_idx]

        # Load behavioral data
        try:
            wheel_pos = one.load_dataset(eid, "_ibl_wheel.position")
            wheel_ts = one.load_dataset(eid, "_ibl_wheel.timestamps")
            wheel_velocity = compute_wheel_velocity(wheel_pos, wheel_ts)
        except Exception:
            wheel_ts = np.array([spike_times[0], spike_times[-1]])
            wheel_velocity = np.zeros(2)
            logger.warning(f"  No wheel data for {eid}, using zeros")

        # Load trial data
        trial_data = {}
        trial_keys = {
            "stimOn_times": "_ibl_trials.stimOn_times",
            "response_times": "_ibl_trials.response_times",
            "choice": "_ibl_trials.choice",
            "contrastLeft": "_ibl_trials.contrastLeft",
            "contrastRight": "_ibl_trials.contrastRight",
            "feedbackType": "_ibl_trials.feedbackType",
            "intervals": "_ibl_trials.intervals",
        }
        for key, dataset_name in trial_keys.items():
            try:
                trial_data[key] = one.load_dataset(eid, dataset_name)
            except Exception:
                trial_data[key] = None

        # Determine session domain
        t_start = float(spike_times[0])
        t_end = float(spike_times[-1])

        # Determine train/valid/test split based on time (80/10/10)
        duration = t_end - t_start
        train_end = t_start + duration * 0.8
        valid_end = t_start + duration * 0.9

        # Write HDF5 file (temporaldata-compatible format)
        with h5py.File(output_file, "w") as f:
            # Root attributes for temporaldata.Data
            f.attrs["object"] = "Data"
            f.attrs["absolute_start"] = 0.0

            # Session info
            session_grp = f.create_group("session")
            session_grp.attrs["object"] = "Data"
            session_grp.attrs["absolute_start"] = 0.0
            session_grp.attrs["id"] = eid

            # Helper to create Interval groups
            def create_interval(parent, name, starts, ends):
                grp = parent.create_group(name)
                grp.attrs["object"] = "Interval"
                grp.attrs["timekeys"] = np.array([b"start", b"end"])
                grp.attrs["allow_split_mask_overlap"] = False
                grp.attrs["_unicode_keys"] = np.array([], dtype="S1")
                grp.create_dataset("start", data=np.asarray(starts, dtype=np.float64))
                grp.create_dataset("end", data=np.asarray(ends, dtype=np.float64))
                return grp

            # Domain (full session)
            create_interval(f, "domain", [t_start], [t_end])

            # Train/valid/test domains
            create_interval(f, "train_domain", [t_start], [train_end])
            create_interval(f, "valid_domain", [train_end], [valid_end])
            create_interval(f, "test_domain", [valid_end], [t_end])

            # Spikes - as IrregularTimeSeries
            spikes_grp = f.create_group("spikes")
            spikes_grp.attrs["object"] = "IrregularTimeSeries"
            spikes_grp.attrs["timekeys"] = np.array([b"timestamps"])
            spikes_grp.attrs["_unicode_keys"] = np.array([], dtype="S1")
            spikes_grp.create_dataset("timestamps", data=spike_times.astype(np.float64))
            spikes_grp.create_dataset("unit_index", data=spike_unit_idx.astype(np.int64))
            create_interval(spikes_grp, "domain", [t_start], [t_end])

            # Units - as ArrayDict
            units_grp = f.create_group("units")
            units_grp.attrs["object"] = "ArrayDict"
            units_grp.attrs["_unicode_keys"] = np.array([], dtype="S1")
            units_grp.create_dataset(
                "id",
                data=np.array(all_unit_ids, dtype=h5py.special_dtype(vlen=str)),
            )
            units_grp.create_dataset(
                "brain_region",
                data=np.array(all_unit_brain_regions, dtype=h5py.special_dtype(vlen=str)),
            )
            units_grp.create_dataset(
                "quality_label",
                data=np.array(all_unit_quality, dtype=np.int64),
            )

            # Behavior (wheel) - as IrregularTimeSeries
            behavior_grp = f.create_group("behavior")
            behavior_grp.attrs["object"] = "IrregularTimeSeries"
            behavior_grp.attrs["timekeys"] = np.array([b"timestamps"])
            behavior_grp.attrs["_unicode_keys"] = np.array([], dtype="S1")
            behavior_grp.create_dataset("timestamps", data=wheel_ts.astype(np.float64))
            behavior_grp.create_dataset(
                "wheel_velocity", data=wheel_velocity.astype(np.float64)
            )
            # Behavior domain (required for temporaldata slicing)
            beh_t_start = float(wheel_ts[0]) if len(wheel_ts) > 0 else t_start
            beh_t_end = float(wheel_ts[-1]) if len(wheel_ts) > 0 else t_end
            create_interval(behavior_grp, "domain", [beh_t_start], [beh_t_end])

            # Trials - as Interval
            trials_grp = create_interval(f, "trials", [], [])
            if trial_data.get("intervals") is not None:
                intervals = trial_data["intervals"]
                del trials_grp["start"]
                del trials_grp["end"]
                trials_grp.create_dataset(
                    "start", data=intervals[:, 0].astype(np.float64)
                )
                trials_grp.create_dataset(
                    "end", data=intervals[:, 1].astype(np.float64)
                )
            for key in ["stimOn_times", "response_times", "choice", "contrastLeft",
                        "contrastRight", "feedbackType"]:
                if trial_data.get(key) is not None:
                    data = trial_data[key]
                    if isinstance(data, np.ndarray):
                        trials_grp.create_dataset(key, data=data)

        n_spikes = len(spike_times)
        n_units = unit_offset
        logger.info(
            f"  Saved {eid}.h5: {n_spikes} spikes, {n_units} units, "
            f"duration={duration:.1f}s"
        )
        return True

    except Exception as e:
        logger.error(f"  Failed to process session {eid}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess IBL data to HDF5")
    parser.add_argument(
        "--n_sessions",
        type=int,
        default=10,
        help="Number of sessions to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/datasets/ibl_processed",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/root/autodl-tmp/datasets/ibl_cache",
        help="ONE API cache directory",
    )
    parser.add_argument(
        "--min_good_units",
        type=int,
        default=10,
        help="Minimum good units per session",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to IBL
    logger.info("Connecting to IBL...")
    from one.api import ONE

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(args.cache_dir),
    )

    # Get sessions (simple query to avoid timeout with complex django filters)
    logger.info("Querying BWM sessions...")
    sessions = one.alyx.rest(
        "sessions",
        "list",
        task_protocol="ephys",
        project="brainwide",
    )
    logger.info(f"Found {len(sessions)} BWM sessions")

    n_sessions = min(args.n_sessions, len(sessions))
    selected = sessions[:n_sessions]

    # Process each session
    success = 0
    fail = 0
    for i, session in enumerate(selected):
        eid = session["id"] if isinstance(session, dict) else session
        logger.info(f"[{i + 1}/{n_sessions}] Processing {eid}")
        if process_session(one, eid, output_dir, args.min_good_units):
            success += 1
        else:
            fail += 1

    logger.info(f"Processing complete: {success} succeeded, {fail} failed")
    logger.info(f"HDF5 files saved to {output_dir}")


if __name__ == "__main__":
    main()
