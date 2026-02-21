"""
Allen Brain Observatory Neuropixels Data Download and Preprocessing Script

Downloads Neuropixels Visual Coding data from Allen Brain Observatory
and converts to HDF5 format compatible with torch_brain.dataset.Dataset.

IMPORTANT: This script must be run in the 'allen' conda environment:
    conda run -n allen python scripts/download_allen.py

The output HDF5 files can then be loaded in the main 'poyo' environment.

Usage:
    conda run -n allen python scripts/download_allen.py --n_sessions 5
    conda run -n allen python scripts/download_allen.py --n_sessions 58  # all sessions
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_ecephys_cache(manifest_path: str):
    """Initialize Allen EcephysProjectCache."""
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
        EcephysProjectCache,
    )

    return EcephysProjectCache.from_warehouse(manifest=manifest_path)


def process_allen_session(cache, session_id: int, output_dir: Path, stimulus_name: str = "natural_scenes"):
    """Process a single Allen Neuropixels session into HDF5 format.

    Args:
        cache: EcephysProjectCache instance
        session_id: Allen session ID
        output_dir: Output directory for HDF5 files
        stimulus_name: Which stimulus to extract (default: natural_scenes)
    """
    output_file = output_dir / f"allen_{session_id}.h5"
    if output_file.exists():
        logger.info(f"  allen_{session_id}.h5 already exists, skipping")
        return True

    try:
        session = cache.get_session_data(session_id)
        logger.info(f"  Loaded session {session_id}")

        # Get spike times
        spike_times_dict = session.spike_times  # dict: unit_id -> array of spike times

        # Get units table with quality metrics
        units = session.units
        # Filter for good quality units
        good_units = units[
            (units["presence_ratio"] > 0.95)
            & (units["isi_violations"] < 0.5)
            & (units["amplitude_cutoff"] < 0.1)
        ]

        if len(good_units) < 10:
            logger.warning(f"  Only {len(good_units)} good units, skipping")
            return False

        good_unit_ids = good_units.index.values
        logger.info(f"  {len(good_unit_ids)} good units (of {len(units)} total)")

        # Collect all spikes from good units
        all_spike_times = []
        all_spike_unit_idx = []
        unit_id_list = []
        brain_region_list = []

        for local_idx, unit_id in enumerate(good_unit_ids):
            times = spike_times_dict.get(unit_id, np.array([]))
            if len(times) == 0:
                continue

            all_spike_times.append(times)
            all_spike_unit_idx.append(np.full(len(times), local_idx, dtype=np.int64))
            unit_id_list.append(f"allen_{session_id}/unit_{unit_id}")

            # Brain region
            region = good_units.loc[unit_id, "ecephys_structure_acronym"]
            brain_region_list.append(str(region) if region else "unknown")

        if len(all_spike_times) == 0:
            logger.warning(f"  No spikes from good units, skipping")
            return False

        spike_times = np.concatenate(all_spike_times)
        spike_unit_idx = np.concatenate(all_spike_unit_idx)

        # Sort by time
        sort_idx = np.argsort(spike_times)
        spike_times = spike_times[sort_idx]
        spike_unit_idx = spike_unit_idx[sort_idx]

        # Session domain
        t_start = float(spike_times[0])
        t_end = float(spike_times[-1])
        duration = t_end - t_start

        # Train/valid/test split (80/10/10 by time)
        train_end = t_start + duration * 0.8
        valid_end = t_start + duration * 0.9

        # Get stimulus presentations
        stim_table = session.stimulus_presentations
        stim_data = {}

        # Natural scenes
        if stimulus_name in stim_table["stimulus_name"].values:
            ns_table = stim_table[stim_table["stimulus_name"] == stimulus_name]
            stim_data["natural_scenes"] = {
                "start_time": ns_table["start_time"].values.astype(np.float64),
                "stop_time": ns_table["stop_time"].values.astype(np.float64),
                "frame": ns_table["frame"].values.astype(np.int64)
                if "frame" in ns_table.columns
                else np.arange(len(ns_table), dtype=np.int64),
            }

        # Natural movies
        for movie_name in ["natural_movie_one", "natural_movie_three"]:
            if movie_name in stim_table["stimulus_name"].values:
                nm_table = stim_table[stim_table["stimulus_name"] == movie_name]
                stim_data[movie_name] = {
                    "start_time": nm_table["start_time"].values.astype(np.float64),
                    "stop_time": nm_table["stop_time"].values.astype(np.float64),
                    "frame": nm_table["frame"].values.astype(np.int64)
                    if "frame" in nm_table.columns
                    else np.arange(len(nm_table), dtype=np.int64),
                }

        # Running speed
        try:
            running_speed = session.running_speed
            running_ts = running_speed["start_time"].values.astype(np.float64)
            running_vals = running_speed["velocity"].values.astype(np.float64)
        except Exception:
            running_ts = np.array([t_start, t_end])
            running_vals = np.zeros(2)
            logger.warning(f"  No running speed data")

        # Write HDF5
        with h5py.File(output_file, "w") as f:
            # Session
            session_grp = f.create_group("session")
            session_grp.create_dataset("id", data=f"allen_{session_id}")

            # Domain
            domain_grp = f.create_group("domain")
            domain_grp.create_dataset("start", data=np.array([t_start]))
            domain_grp.create_dataset("end", data=np.array([t_end]))

            # Train/valid/test domains
            for name, (s, e) in [
                ("train_domain", (t_start, train_end)),
                ("valid_domain", (train_end, valid_end)),
                ("test_domain", (valid_end, t_end)),
            ]:
                grp = f.create_group(name)
                grp.create_dataset("start", data=np.array([s]))
                grp.create_dataset("end", data=np.array([e]))

            # Spikes
            spikes_grp = f.create_group("spikes")
            spikes_grp.create_dataset("timestamps", data=spike_times)
            spikes_grp.create_dataset("unit_index", data=spike_unit_idx)
            spike_domain = spikes_grp.create_group("domain")
            spike_domain.create_dataset("start", data=np.array([t_start]))
            spike_domain.create_dataset("end", data=np.array([t_end]))

            # Units
            units_grp = f.create_group("units")
            units_grp.create_dataset(
                "id",
                data=np.array(unit_id_list, dtype=h5py.special_dtype(vlen=str)),
            )
            units_grp.create_dataset(
                "brain_region",
                data=np.array(brain_region_list, dtype=h5py.special_dtype(vlen=str)),
            )

            # Behavior (running speed)
            running_grp = f.create_group("running")
            running_grp.create_dataset("timestamps", data=running_ts)
            running_grp.create_dataset("running_speed", data=running_vals)

            # Stimulus presentations
            for stim_name, stim_info in stim_data.items():
                stim_grp = f.create_group(stim_name)
                for key, val in stim_info.items():
                    stim_grp.create_dataset(key, data=val)

        n_spikes = len(spike_times)
        n_units = len(unit_id_list)
        logger.info(
            f"  Saved allen_{session_id}.h5: {n_spikes} spikes, {n_units} units, "
            f"duration={duration:.1f}s, stimuli={list(stim_data.keys())}"
        )
        return True

    except Exception as e:
        logger.error(f"  Failed to process session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Allen Neuropixels data")
    parser.add_argument(
        "--n_sessions",
        type=int,
        default=5,
        help="Number of sessions to process (default: 5, max: 58)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/datasets/allen_processed",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default="/root/autodl-tmp/datasets/allen_cache",
        help="AllenSDK cache/manifest directory",
    )
    parser.add_argument(
        "--stimulus_set",
        type=str,
        default="brain_observatory_1.1",
        choices=["brain_observatory_1.1", "functional_connectivity"],
        help="Which stimulus set to download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = str(manifest_dir / "manifest.json")

    # Initialize cache
    logger.info("Initializing Allen EcephysProjectCache...")
    cache = get_ecephys_cache(manifest_path)

    # Get sessions table
    sessions_table = cache.get_session_table()

    # Filter for the desired stimulus set
    if args.stimulus_set == "brain_observatory_1.1":
        filtered = sessions_table[
            sessions_table["session_type"] == "brain_observatory_1.1"
        ]
    else:
        filtered = sessions_table[
            sessions_table["session_type"] == "functional_connectivity"
        ]

    session_ids = filtered.index.tolist()
    logger.info(f"Found {len(session_ids)} sessions for stimulus set '{args.stimulus_set}'")

    n_sessions = min(args.n_sessions, len(session_ids))
    selected = session_ids[:n_sessions]

    # Process each session
    success = 0
    fail = 0
    for i, sid in enumerate(selected):
        logger.info(f"[{i + 1}/{n_sessions}] Processing session {sid}")
        if process_allen_session(cache, sid, output_dir):
            success += 1
        else:
            fail += 1

    logger.info(f"Processing complete: {success} succeeded, {fail} failed")
    logger.info(f"HDF5 files saved to {output_dir}")


if __name__ == "__main__":
    main()
