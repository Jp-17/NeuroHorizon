# /// brainset-pipeline
# python-version = "3.10"
# dependencies = [
#   "dandi==0.61.2",
#   "scikit-learn==1.5.1",
# ]
# ///

from argparse import ArgumentParser
from typing import NamedTuple
from pynwb import NWBHDF5IO

import datetime
import h5py

import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.model_selection import train_test_split
import pandas as pd

from temporaldata import Data, IrregularTimeSeries, Interval
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.utils.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
    download_file,
    get_nwb_asset_list,
)
from brainsets.taxonomy import RecordingTech, Task
from brainsets import serialize_fn_map

from brainsets.pipeline import BrainsetPipeline

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")

# Selected sessions: 4 small C-CO + 3 J-CO + 3 small M-CO
SELECTED_PATHS = {
    "sub-C/sub-C_ses-CO-20131003_behavior+ecephys.nwb",
    "sub-C/sub-C_ses-CO-20131022_behavior+ecephys.nwb",
    "sub-C/sub-C_ses-CO-20131101_behavior+ecephys.nwb",
    "sub-C/sub-C_ses-CO-20131204_behavior+ecephys.nwb",
    "sub-J/sub-J_ses-CO-20160405_behavior+ecephys.nwb",
    "sub-J/sub-J_ses-CO-20160406_behavior+ecephys.nwb",
    "sub-J/sub-J_ses-CO-20160407_behavior+ecephys.nwb",
    "sub-M/sub-M_ses-CO-20150610_behavior+ecephys.nwb",
    "sub-M/sub-M_ses-CO-20150612_behavior+ecephys.nwb",
    "sub-M/sub-M_ses-CO-20150615_behavior+ecephys.nwb",
}


class Pipeline(BrainsetPipeline):
    brainset_id = "perich_miller_population_2018"
    dandiset_id = "DANDI:000688/draft"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = []
        for x in asset_list:
            if x.path in SELECTED_PATHS:
                manifest_list.append({"path": x.path, "url": x.download_url})

        for m in manifest_list:
            path = m["path"]
            subject_alpha = path.split("/")[0].split("-")[1].lower()
            assert len(subject_alpha) == 1
            task = path.split("/")[1].split("-")[2]
            if task == "CO":
                task = "center_out_reaching"
            elif task == "RT":
                task = "random_target_reaching"
            else:
                raise ValueError(f"Unknown task {task}")
            date = path.split("/")[1].split("-")[3].split("_")[0]
            m["session_id"] = f"{subject_alpha}_{date}_{task}"

        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        fpath = download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=self.args.redownload,
        )
        return fpath

    def process(self, fpath):
        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        brainset_description = BrainsetDescription(
            id="perich_miller_population_2018",
            origin_version="dandi/000688/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000688",
            description="This dataset contains electrophysiology and behavioral data from "
            "three macaques performing either a center-out task or a continuous random "
            "target acquisition task.",
        )

        print(f"Processing file: {fpath}")

        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        self.update_status("Extracting Metadata")
        subject = extract_subject_from_nwb(nwbfile)

        recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
        device_id = f"{subject.id}_{recording_date}"
        task = "center_out_reaching" if "CO" in str(fpath) else "random_target_reaching"
        session_id = f"{device_id}_{task}"

        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
            task=Task.REACHING,
        )

        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        self.update_status("Extracting Spikes")
        spikes, units = extract_spikes_from_nwbfile(
            nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
        )

        self.update_status("Extracting Behavior")
        cursor = extract_behavior(nwbfile)
        cursor_outlier_segments = detect_outliers(cursor)

        if task == "center_out_reaching":
            trials, movement_phases = extract_center_out_reaching_trials(
                nwbfile, cursor
            )
        else:
            trials, movement_phases = extract_random_target_reaching_trials(
                nwbfile, cursor
            )

        for key in movement_phases.keys():
            setattr(
                movement_phases,
                key,
                getattr(movement_phases, key).difference(cursor_outlier_segments),
            )

        io.close()

        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            spikes=spikes,
            units=units,
            trials=trials,
            movement_phases=movement_phases,
            cursor=cursor,
            cursor_outlier_segments=cursor_outlier_segments,
            domain=cursor.domain,
        )

        self.update_status("Creating splits")
        _, valid_trials, test_trials = split_trials(
            trials.select_by_mask(trials.is_valid),
            test_size=0.2,
            valid_size=0.1,
            random_state=42,
        )

        train_sampling_intervals = data.domain.difference(
            (valid_trials | test_trials).dilate(1.0)
        )

        data.set_train_domain(train_sampling_intervals)
        data.set_valid_domain(valid_trials)
        data.set_test_domain(test_trials)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def extract_behavior(nwbfile):
    timestamps = nwbfile.processing["behavior"]["Position"]["cursor_pos"].timestamps[:]
    cursor_pos = nwbfile.processing["behavior"]["Position"]["cursor_pos"].data[:]
    cursor_vel = nwbfile.processing["behavior"]["Velocity"]["cursor_vel"].data[:]
    cursor_acc = nwbfile.processing["behavior"]["Acceleration"]["cursor_acc"].data[:]
    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        acc=cursor_acc,
        domain="auto",
    )
    return cursor


def extract_center_out_reaching_trials(nwbfile, cursor):
    trial_table = nwbfile.trials.to_dataframe()

    trial_grid = np.append(
        trial_table.target_on_time.iloc[:],
        min(trial_table.stop_time.iloc[-1] + 1.0, cursor.domain.end[-1]),
    )
    default_value = np.append(
        trial_table.start_time.iloc[0], trial_table.stop_time.values + 1.0
    )
    default_value[1:-1] = np.minimum(
        default_value[1:-1], trial_table.stop_time.values[1:]
    )
    nan_mask = np.isnan(trial_grid)
    trial_grid[nan_mask] = default_value[nan_mask]
    trial_grid = trial_grid.astype(np.float64)
    trial_table["end"] = trial_grid[1:]
    trial_table["start"] = trial_grid[:-1]

    trials = Interval.from_dataframe(trial_table)
    assert trials.is_disjoint()

    success_mask = trials.result == "R"
    valid_target_mask = ~np.isnan(trials.target_id)
    max_duration_mask = (trials.end - trials.start) < 6.0
    min_duration_mask = (trials.end - trials.start) > 0.5

    trials.is_valid = (
        success_mask & valid_target_mask & max_duration_mask & min_duration_mask
    )

    valid_trials = trials.select_by_mask(trials.is_valid)

    movement_phases = Data(
        hold_period=Interval(
            start=valid_trials.target_on_time, end=valid_trials.go_cue_time
        ),
        reach_period=Interval(
            start=valid_trials.go_cue_time, end=valid_trials.stop_time
        ),
        return_period=Interval(start=valid_trials.stop_time, end=valid_trials.end),
        invalid=trials.select_by_mask(~trials.is_valid),
        domain="auto",
    )
    movement_phases.random_period = cursor.domain.difference(movement_phases.domain)
    return trials, movement_phases


def extract_random_target_reaching_trials(nwbfile, cursor):
    trial_table = nwbfile.trials.to_dataframe()
    trial_table = trial_table.rename(
        columns={"start_time": "start", "stop_time": "end"}
    )
    trials = Interval.from_dataframe(trial_table)

    success_mask = trials.result == "R"
    valid_num_attempts = trials.num_attempted == 4
    max_duration_mask = (trials.end - trials.start) < 10.0
    min_duration_mask = (trials.end - trials.start) > 2.0

    trials.is_valid = (
        success_mask & valid_num_attempts & max_duration_mask & min_duration_mask
    )

    valid_trials = trials.select_by_mask(~np.isnan(trials.go_cue_time_array[:, 0]))

    movement_phases = Data(
        hold_period=Interval(
            start=valid_trials.start, end=valid_trials.go_cue_time_array[:, 0]
        ),
        domain="auto",
    )
    movement_phases.random_period = cursor.domain.difference(movement_phases.domain)
    return trials, movement_phases


def detect_outliers(cursor):
    hand_acc_norm = np.linalg.norm(cursor.acc, axis=1)
    mask_acceleration = hand_acc_norm > 1500.0
    mask_acceleration = binary_dilation(
        mask_acceleration, structure=np.ones(2, dtype=bool)
    )
    mask_position = np.logical_or(cursor.pos[:, 0] < -10, cursor.pos[:, 0] > 10)
    mask_position = np.logical_or(mask_position, cursor.pos[:, 1] < -10)
    mask_position = np.logical_or(mask_position, cursor.pos[:, 1] > 10)
    mask_position = binary_dilation(mask_position, np.ones(400, dtype=bool))
    mask_position = binary_erosion(mask_position, np.ones(100, dtype=bool))

    outlier_mask = np.logical_or(mask_acceleration, mask_position)

    start = cursor.timestamps[np.where(np.diff(outlier_mask.astype(int)) == 1)[0]]
    if outlier_mask[0]:
        start = np.insert(start, 0, cursor.timestamps[0])
    end = cursor.timestamps[np.where(np.diff(outlier_mask.astype(int)) == -1)[0]]
    if outlier_mask[-1]:
        end = np.append(end, cursor.timestamps[-1])

    cursor_outlier_segments = Interval(start=start, end=end)
    assert cursor_outlier_segments.is_disjoint()
    return cursor_outlier_segments


def split_trials(trials, test_size=0.2, valid_size=0.1, random_state=42):
    num_trials = len(trials)
    train_size = 1.0 - test_size - valid_size

    train_valid_ids, test_ids = train_test_split(
        np.arange(num_trials), test_size=test_size, random_state=random_state
    )
    train_ids, valid_ids = train_test_split(
        train_valid_ids,
        test_size=valid_size / (train_size + valid_size),
        random_state=random_state,
    )

    train_trials = trials.select_by_mask(np.isin(np.arange(num_trials), train_ids))
    valid_trials = trials.select_by_mask(np.isin(np.arange(num_trials), valid_ids))
    test_trials = trials.select_by_mask(np.isin(np.arange(num_trials), test_ids))

    return train_trials, valid_trials, test_trials
