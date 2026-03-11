"""Trial-aligned sampler for NeuroHorizon.

Provides TrialAlignedSampler that generates DatasetIndex objects
centered on go_cue_time for each trial.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import Sampler

from torch_brain.data.dataset import DatasetIndex


@dataclass
class TrialDatasetIndex(DatasetIndex):
    """DatasetIndex extended with trial metadata.

    Attributes:
        target_id: Trial direction (0-7 for 8-direction center-out reaching)
        go_cue_time: Go cue timestamp for trial alignment
    """

    target_id: int = -1
    go_cue_time: float = 0.0


class TrialAlignedSampler(Sampler):
    """Trial-aligned sampler with go_cue alignment.

    Each sample = one trial, with window:
        [go_cue_time - obs_window, go_cue_time + pred_window]

    This ensures the model's history window covers pre-movement activity
    and the prediction window covers the reach period.

    Args:
        trial_info: Dict[recording_id -> dict with keys:
            'go_cue_time': array of go_cue times
            'target_id': array of target IDs (0-7)
        ]
        obs_window: Observation window before go_cue (seconds)
        pred_window: Prediction window after go_cue (seconds)
        shuffle: Whether to shuffle trial order
        generator: Optional random generator
    """

    def __init__(
        self,
        *,
        trial_info: Dict[str, Dict],
        obs_window: float = 0.5,
        pred_window: float = 0.25,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        self.trial_info = trial_info
        self.obs_window = obs_window
        self.pred_window = pred_window
        self.shuffle = shuffle
        self.generator = generator

        # Build index list
        self._indices = []
        for recording_id, info in trial_info.items():
            go_cues = info["go_cue_time"]
            target_ids = info["target_id"]

            for i in range(len(go_cues)):
                gc = float(go_cues[i])
                tid = int(target_ids[i])
                start = gc - obs_window
                end = gc + pred_window

                self._indices.append(
                    TrialDatasetIndex(
                        recording_id=recording_id,
                        start=start,
                        end=end,
                        target_id=tid,
                        go_cue_time=gc,
                    )
                )

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self._indices), generator=self.generator)
            for idx in perm:
                yield self._indices[idx]
        else:
            yield from self._indices
