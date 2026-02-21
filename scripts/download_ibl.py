"""
IBL Brain-wide Map Data Download Script

Downloads spike sorting and behavioral data from IBL via ONE API.
Saves raw data for later preprocessing into HDF5 format.

Usage:
    conda run -n poyo python scripts/download_ibl.py --n_sessions 10 --output_dir /path/to/output
    conda run -n poyo python scripts/download_ibl.py --n_sessions 459 --output_dir /path/to/output
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from one.api import ONE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_bwm_sessions(one: ONE):
    """Get sessions from IBL Brain-wide Map that pass quality control.

    Criteria:
    - Have at least one probe insertion with good histology
    - Have spike sorting data available
    - Have behavioral data available
    """
    # Query for sessions with resolved alignment (good histology)
    sessions = one.alyx.rest(
        "sessions",
        "list",
        task_protocol="ephys",
        project="brainwide",
        performance_gte=70,
        django=(
            "extended_qc__behavior,1,"
            "~json__IS_MOCK,True,"
            "n_correct_trials__gte,400"
        ),
    )

    logger.info(f"Found {len(sessions)} BWM sessions meeting quality criteria")
    return sessions


def download_session_data(one: ONE, eid: str, output_dir: Path):
    """Download spike sorting and behavioral data for a single session.

    Downloads:
    - Spike times, clusters, amplitudes, depths
    - Cluster metrics and quality labels
    - Wheel position and timestamps
    - Trial information
    - Channel brain locations
    """
    session_dir = output_dir / eid
    session_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    marker_file = session_dir / ".download_complete"
    if marker_file.exists():
        logger.info(f"Session {eid} already downloaded, skipping")
        return True

    try:
        # --- Spike sorting data ---
        spike_datasets = [
            "spikes.times",
            "spikes.clusters",
            "spikes.amps",
            "spikes.depths",
            "clusters.metrics",
            "clusters.channels",
            "clusters.depths",
            "clusters.waveforms",
            "channels.localCoordinates",
            "channels.brainLocationIds_ccf_2017",
            "channels.mlapdv",
        ]
        logger.info(f"  Downloading spike sorting data for {eid}...")
        spike_files = one.load_datasets(
            eid,
            datasets=spike_datasets,
            download_only=True,
            assert_present=False,
        )

        # --- Behavioral data ---
        behavior_datasets = [
            "_ibl_wheel.position",
            "_ibl_wheel.timestamps",
            "_ibl_wheelMoves.intervals",
            "_ibl_trials.stimOn_times",
            "_ibl_trials.response_times",
            "_ibl_trials.choice",
            "_ibl_trials.contrastLeft",
            "_ibl_trials.contrastRight",
            "_ibl_trials.feedbackType",
            "_ibl_trials.reactionTime",
            "_ibl_trials.goCue_times",
            "_ibl_trials.feedback_times",
            "_ibl_trials.firstMovement_times",
            "_ibl_trials.intervals",
        ]
        logger.info(f"  Downloading behavioral data for {eid}...")
        behavior_files = one.load_datasets(
            eid,
            datasets=behavior_datasets,
            download_only=True,
            assert_present=False,
        )

        # Save session info
        session_info = one.alyx.rest("sessions", "read", id=eid)
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2, default=str)

        # Mark download complete
        marker_file.touch()
        logger.info(f"  Session {eid} download complete")
        return True

    except Exception as e:
        logger.error(f"  Failed to download session {eid}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download IBL Brain-wide Map data")
    parser.add_argument(
        "--n_sessions",
        type=int,
        default=10,
        help="Number of sessions to download (default: 10, use 459 for full dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/datasets/ibl_raw",
        help="Output directory for raw downloaded data",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/root/autodl-tmp/datasets/ibl_cache",
        help="ONE API cache directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Connect to IBL public server
    logger.info("Connecting to IBL public Alyx server...")
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(cache_dir),
    )

    # Get BWM sessions
    logger.info("Querying BWM sessions...")
    sessions = get_bwm_sessions(one)

    if len(sessions) == 0:
        logger.error("No sessions found! Check network connection.")
        return

    # Select sessions
    n_sessions = min(args.n_sessions, len(sessions))
    selected_sessions = sessions[:n_sessions]
    logger.info(f"Selected {n_sessions} sessions for download")

    # Download each session
    success_count = 0
    fail_count = 0
    for i, session in enumerate(selected_sessions):
        eid = session["id"] if isinstance(session, dict) else session
        logger.info(f"[{i + 1}/{n_sessions}] Downloading session {eid}")
        if download_session_data(one, eid, output_dir):
            success_count += 1
        else:
            fail_count += 1

    # Save session list
    session_list = [s["id"] if isinstance(s, dict) else s for s in selected_sessions]
    with open(output_dir / "session_list.json", "w") as f:
        json.dump(session_list, f, indent=2)

    logger.info(f"Download complete: {success_count} succeeded, {fail_count} failed")
    logger.info(f"Session list saved to {output_dir / 'session_list.json'}")


if __name__ == "__main__":
    main()
