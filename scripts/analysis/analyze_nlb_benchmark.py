#!/usr/bin/env python3
"""
Phase 0.4.1: NLB Benchmark 数据分析与适配性调查

Part A: NLB MC_Maze 数据结构分析与可视化
Part B: 数据适配性调查（split 一致性、held-in/held-out、其他子数据集）
Part C: NLB 指标对齐（co-bps, fp-bps, PSTH R²）

Output:
  - results/figures/data_exploration/06_nlb_data_structure.png
  - results/figures/data_exploration/07_nlb_split_comparison.png
  - results/figures/data_exploration/nlb_analysis_summary.json

Usage:
  conda activate poyo
  cd /root/autodl-tmp/NeuroHorizon
  python scripts/analysis/analyze_nlb_benchmark.py
"""

import os
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/root/autodl-tmp/NeuroHorizon")
NLB_DIR = PROJECT_ROOT / "data/nlb/processed/pei_pandarinath_nlb_2021"
TRAIN_PATH = NLB_DIR / "jenkins_maze_train.h5"
TEST_PATH = NLB_DIR / "jenkins_maze_test.h5"
OUTPUT_DIR = PROJECT_ROOT / "results/figures/data_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_h5_data(path):
    """Load all datasets from HDF5 file into a nested dict."""
    data = {}
    with h5py.File(path, "r") as f:
        def extract(group, d):
            for k in group.keys():
                item = group[k]
                if isinstance(item, h5py.Group):
                    d[k] = {}
                    extract(item, d[k])
                else:
                    d[k] = item[()]
        extract(f, data)
    return data


def part_a_data_structure(train_data, test_data):
    """Part A: NLB MC_Maze 数据结构分析与可视化 -> Figure 06"""
    print("\n" + "="*60)
    print("Part A: NLB MC_Maze 数据结构分析")
    print("="*60)

    # Basic stats
    n_trials = len(train_data["domain"]["start"])
    n_units_train = len(train_data["units"]["id"])
    n_units_test = len(test_data["units"]["id"])
    n_spikes_train = len(train_data["spikes"]["timestamps"])
    n_spikes_test = len(test_data["spikes"]["timestamps"])

    domain_start = train_data["domain"]["start"]
    domain_end = train_data["domain"]["end"]
    trial_durations = domain_end - domain_start

    n_train_domain = len(train_data["train_domain"]["start"])
    n_valid_domain = len(train_data["valid_domain"]["start"])
    n_test_domain = len(train_data["test_domain"]["start"])

    n_hand_samples = len(train_data["hand"]["timestamps"])
    n_eye_samples = len(train_data["eye"]["timestamps"])
    hand_dt = np.median(np.diff(train_data["hand"]["timestamps"]))
    eye_dt = np.median(np.diff(train_data["eye"]["timestamps"]))

    n_eval_intervals = len(train_data["nlb_eval_intervals"]["start"])

    stats = {
        "dataset": "NLB MC_Maze (jenkins_maze, small subset)",
        "train_file": {
            "n_trials": int(n_trials),
            "n_units": int(n_units_train),
            "n_spikes": int(n_spikes_train),
            "train_domain_trials": int(n_train_domain),
            "valid_domain_trials": int(n_valid_domain),
            "test_domain_trials": int(n_test_domain),
            "n_hand_samples": int(n_hand_samples),
            "hand_sampling_rate_hz": round(1.0 / hand_dt, 1),
            "n_eye_samples": int(n_eye_samples),
            "eye_sampling_rate_hz": round(1.0 / eye_dt, 1),
            "n_eval_intervals": int(n_eval_intervals),
            "total_duration_s": round(float(domain_end[-1] - domain_start[0]), 2),
            "trial_duration_mean_s": round(float(np.mean(trial_durations)), 3),
            "trial_duration_std_s": round(float(np.std(trial_durations)), 3),
        },
        "test_file": {
            "n_trials": int(len(test_data["domain"]["start"])),
            "n_units": int(n_units_test),
            "n_spikes": int(n_spikes_test),
        },
        "held_in_units": int(n_units_test),
        "held_out_units": int(n_units_train - n_units_test),
    }

    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"\n  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    # --- Figure 06: NLB Data Structure Visualization ---
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle("NLB MC_Maze (Jenkins) Data Structure Analysis", fontsize=16, fontweight="bold")

    # Use GridSpec: top row (timeline) gets more height
    gs = fig.add_gridspec(4, 2, height_ratios=[2.5, 1, 1, 1], hspace=0.35, wspace=0.3)

    # Extract trial period timing
    trial_starts = train_data["trials"]["start"]
    trial_ends = train_data["trials"]["end"]
    target_on_times = train_data["trials"]["target_on_time"]
    go_cue_times = train_data["trials"]["go_cue_time"]
    move_onset_times = train_data["trials"]["move_onset_time"]

    # (0, 0:1) Full-width timeline spanning both columns
    ax = fig.add_subplot(gs[0, :])

    # --- Domain-level tracks ---
    colors_map = {
        "domain": "#2196F3",
        "train_domain": "#4CAF50",
        "valid_domain": "#FF9800",
        "test_domain": "#F44336",
        "nlb_eval_intervals": "#9C27B0",
    }
    y_labels = []
    y_pos = 0
    for track_name, color in colors_map.items():
        if track_name == "nlb_eval_intervals":
            starts = train_data["nlb_eval_intervals"]["start"]
            ends = train_data["nlb_eval_intervals"]["end"]
        elif track_name == "domain":
            starts = domain_start
            ends = domain_end
        else:
            starts = train_data[track_name]["start"]
            ends = train_data[track_name]["end"]
        for s, e in zip(starts, ends):
            ax.barh(y_pos, e - s, left=s, height=0.6, color=color, alpha=0.7)
        y_labels.append(track_name)
        y_pos += 1

    # --- Trial period track ---
    period_colors = {
        "pre-target": "#90CAF9",   # light blue
        "delay":      "#FFE082",   # light yellow/amber
        "RT":         "#FFAB91",   # light orange/red
        "movement":   "#A5D6A7",   # light green
    }
    for i in range(n_trials):
        ts = trial_starts[i]
        to = target_on_times[i]
        gc = go_cue_times[i]
        mo = move_onset_times[i]
        te = trial_ends[i]
        ax.barh(y_pos, to - ts, left=ts, height=0.6, color=period_colors["pre-target"], alpha=0.85)
        ax.barh(y_pos, gc - to, left=to, height=0.6, color=period_colors["delay"], alpha=0.85)
        ax.barh(y_pos, mo - gc, left=gc, height=0.6, color=period_colors["RT"], alpha=0.85)
        ax.barh(y_pos, te - mo, left=mo, height=0.6, color=period_colors["movement"], alpha=0.85)
    y_labels.append("trial_periods")
    y_pos += 1

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_title("Timeline: All Temporal Structures (full session)")
    ax.invert_yaxis()

    # Add period legend
    period_patches = [
        mpatches.Patch(color=period_colors["pre-target"], label="pre-target"),
        mpatches.Patch(color=period_colors["delay"], label="delay"),
        mpatches.Patch(color=period_colors["RT"], label="RT"),
        mpatches.Patch(color=period_colors["movement"], label="movement"),
    ]
    domain_patches = [
        mpatches.Patch(color="#2196F3", label="domain"),
        mpatches.Patch(color="#4CAF50", label="train_domain"),
        mpatches.Patch(color="#FF9800", label="valid_domain"),
        mpatches.Patch(color="#F44336", label="test_domain"),
        mpatches.Patch(color="#9C27B0", label="nlb_eval_intervals"),
    ]
    leg1 = ax.legend(handles=domain_patches, loc="upper right", fontsize=7, title="Domains", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=period_patches, loc="lower right", fontsize=7, title="Trial Periods", title_fontsize=8)

    # (1,0) Trial duration distribution
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(trial_durations, bins=20, color="#2196F3", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(trial_durations), color="red", linestyle="--",
               label=f"Mean: {np.mean(trial_durations):.3f}s")
    ax.set_xlabel("Trial Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Trial Duration Distribution (n={n_trials})")
    ax.legend()

    # (1,1) Trial period duration distributions (stacked)
    ax = fig.add_subplot(gs[1, 1])
    pre_target_durs = target_on_times - trial_starts
    delay_durs = go_cue_times - target_on_times
    rt_durs = move_onset_times - go_cue_times
    movement_durs = trial_ends - move_onset_times
    period_names = ["pre-target", "delay", "RT", "movement"]
    period_data = [pre_target_durs, delay_durs, rt_durs, movement_durs]
    bp = ax.boxplot(period_data, tick_labels=period_names, patch_artist=True, widths=0.6)
    for patch, pname in zip(bp["boxes"], period_names):
        patch.set_facecolor(period_colors[pname])
        patch.set_alpha(0.8)
    ax.set_ylabel("Duration (s)")
    ax.set_title("Trial Period Durations")
    # Annotate with means
    for i, (pname, pdata) in enumerate(zip(period_names, period_data)):
        ax.text(i + 1, np.max(pdata) + 0.02, f"{np.mean(pdata):.3f}s",
                ha="center", fontsize=8, color="gray")

    # (2,0) Spike count per unit
    ax = fig.add_subplot(gs[2, 0])
    unit_indices = train_data["spikes"]["unit_index"]
    unit_counts = np.bincount(unit_indices, minlength=n_units_train)
    ax.bar(range(n_units_train), unit_counts, color="#4CAF50", alpha=0.7)
    ax.axhline(np.mean(unit_counts), color="red", linestyle="--",
               label=f"Mean: {np.mean(unit_counts):.0f}")
    ax.set_xlabel("Unit Index")
    ax.set_ylabel("Spike Count")
    ax.set_title(f"Spike Count per Unit (n={n_units_train} units, {n_spikes_train} total spikes)")
    ax.legend()

    # (2,1) Firing rate distribution
    ax = fig.add_subplot(gs[2, 1])
    total_dur = float(domain_end[-1] - domain_start[0])
    firing_rates = unit_counts / total_dur
    ax.hist(firing_rates, bins=30, color="#FF9800", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(firing_rates), color="red", linestyle="--",
               label=f"Mean: {np.mean(firing_rates):.1f} Hz")
    ax.axvline(np.median(firing_rates), color="blue", linestyle="--",
               label=f"Median: {np.median(firing_rates):.1f} Hz")
    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Firing Rate Distribution (Train)")
    ax.legend()

    # (3,0) Behavioral data overview: hand velocity
    ax = fig.add_subplot(gs[3, 0])
    hand_ts = train_data["hand"]["timestamps"]
    hand_vel = train_data["hand"]["vel"]
    speed = np.sqrt(hand_vel[:, 0]**2 + hand_vel[:, 1]**2)
    n_show = min(2000, len(hand_ts))
    ax.plot(hand_ts[:n_show], speed[:n_show], color="#2196F3", alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hand Speed (a.u.)")
    ax.set_title(f"Hand Speed (first {n_show} samples, {stats['train_file']['hand_sampling_rate_hz']} Hz)")

    # (3,1) Summary stats table
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    table_data = [
        ["Metric", "Train File", "Test File"],
        ["Trials", str(n_trials), str(len(test_data["domain"]["start"]))],
        ["Units", str(n_units_train), str(n_units_test)],
        ["Spikes", f"{n_spikes_train:,}", f"{n_spikes_test:,}"],
        ["Held-in Units", str(n_units_test), str(n_units_test)],
        ["Held-out Units", str(n_units_train - n_units_test), "N/A"],
        ["Hand Data", "Yes (pos+vel)", "No"],
        ["Eye Data", "Yes (pos)", "No"],
        ["train_domain", str(n_train_domain), "N/A"],
        ["valid_domain", str(n_valid_domain), "N/A"],
        ["test_domain", str(n_test_domain), "N/A"],
        ["nlb_eval_intervals", str(n_eval_intervals), "N/A"],
        ["Duration (s)", f"{total_dur:.1f}", "N/A"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    for j in range(3):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=20)

    fig_path = OUTPUT_DIR / "06_nlb_data_structure.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")

    return stats


def part_b_adaptability(train_data, test_data, stats):
    """Part B: NLB 数据适配性调查 -> Figure 07 + analysis results"""
    print("\n" + "="*60)
    print("Part B: NLB 数据适配性调查")
    print("="*60)

    adaptability = {}

    # --- B1: Split consistency check ---
    print("\n  --- B1: Split 一致性检查 ---")

    trials = train_data["trials"]
    n_trials = len(trials["start"])

    train_mask_nwb = trials["train_mask_nwb"]
    test_mask_nwb = trials["test_mask_nwb"]
    train_mask_bs = trials["train_mask"]
    valid_mask_bs = trials["valid_mask"]
    test_mask_bs = trials["test_mask"]

    n_nwb_train = int(np.sum(train_mask_nwb))
    n_nwb_test = int(np.sum(test_mask_nwb))
    n_bs_train = int(np.sum(train_mask_bs))
    n_bs_valid = int(np.sum(valid_mask_bs))
    n_bs_test = int(np.sum(test_mask_bs))

    print(f"    NLB original split:  train={n_nwb_train}, test={n_nwb_test}, neither={n_trials - n_nwb_train - n_nwb_test}")
    print(f"    Brainsets split:     train={n_bs_train}, valid={n_bs_valid}, test={n_bs_test}")

    nwb_train_indices = set(np.where(train_mask_nwb)[0])
    nwb_test_indices = set(np.where(test_mask_nwb)[0])
    bs_train_indices = set(np.where(train_mask_bs)[0])
    bs_valid_indices = set(np.where(valid_mask_bs)[0])
    bs_test_indices = set(np.where(test_mask_bs)[0])

    bs_train_in_nwb_train = len(bs_train_indices & nwb_train_indices)
    bs_train_in_nwb_test = len(bs_train_indices & nwb_test_indices)
    bs_valid_in_nwb_train = len(bs_valid_indices & nwb_train_indices)
    bs_valid_in_nwb_test = len(bs_valid_indices & nwb_test_indices)
    bs_test_in_nwb_train = len(bs_test_indices & nwb_train_indices)
    bs_test_in_nwb_test = len(bs_test_indices & nwb_test_indices)

    split_comparison = {
        "nlb_original": {"train": n_nwb_train, "test": n_nwb_test},
        "brainsets": {"train": n_bs_train, "valid": n_bs_valid, "test": n_bs_test},
        "cross_mapping": {
            "bs_train_in_nwb_train": bs_train_in_nwb_train,
            "bs_train_in_nwb_test": bs_train_in_nwb_test,
            "bs_valid_in_nwb_train": bs_valid_in_nwb_train,
            "bs_valid_in_nwb_test": bs_valid_in_nwb_test,
            "bs_test_in_nwb_train": bs_test_in_nwb_train,
            "bs_test_in_nwb_test": bs_test_in_nwb_test,
        },
    }

    # Check split_indicator
    split_indicators = trials["split_indicator"]
    unique_splits = set()
    for s in split_indicators:
        if isinstance(s, bytes):
            unique_splits.add(s.decode())
        else:
            unique_splits.add(str(s))
    print(f"    split_indicator unique values: {unique_splits}")
    split_comparison["split_indicator_values"] = list(unique_splits)

    # Check train_domain/valid_domain/test_domain vs trials mapping
    train_domain_starts = set(np.round(train_data["train_domain"]["start"], 6).tolist())
    valid_domain_starts = set(np.round(train_data["valid_domain"]["start"], 6).tolist())
    test_domain_starts = set(np.round(train_data["test_domain"]["start"], 6).tolist())
    trial_starts = np.round(train_data["trials"]["start"], 6).tolist()

    td_from_trials = []
    vd_from_trials = []
    testd_from_trials = []
    for i, ts in enumerate(trial_starts):
        if ts in train_domain_starts:
            td_from_trials.append(i)
        if ts in valid_domain_starts:
            vd_from_trials.append(i)
        if ts in test_domain_starts:
            testd_from_trials.append(i)

    print(f"    train_domain maps to trial indices: {len(td_from_trials)} trials")
    print(f"    valid_domain maps to trial indices: {len(vd_from_trials)} trials")
    print(f"    test_domain maps to trial indices: {len(testd_from_trials)} trials")

    all_bs_assigned = set(td_from_trials) | set(vd_from_trials) | set(testd_from_trials)
    if len(all_bs_assigned) == n_trials:
        consistency_note = "All 100 trials are assigned to exactly one brainsets domain."
    else:
        consistency_note = f"WARNING: Only {len(all_bs_assigned)}/{n_trials} trials assigned to brainsets domains."

    td_in_nwb_train = len(set(td_from_trials) & nwb_train_indices)
    td_in_nwb_test = len(set(td_from_trials) & nwb_test_indices)
    vd_in_nwb_train = len(set(vd_from_trials) & nwb_train_indices)
    testd_in_nwb_train = len(set(testd_from_trials) & nwb_train_indices)
    testd_in_nwb_test = len(set(testd_from_trials) & nwb_test_indices)

    conclusion_lines = [
        consistency_note,
        f"brainsets train_domain ({len(td_from_trials)} trials): {td_in_nwb_train} from NLB-train, {td_in_nwb_test} from NLB-test",
        f"brainsets valid_domain ({len(vd_from_trials)} trials): {vd_in_nwb_train} from NLB-train",
        f"brainsets test_domain ({len(testd_from_trials)} trials): {testd_in_nwb_train} from NLB-train, {testd_in_nwb_test} from NLB-test",
    ]
    if td_in_nwb_test > 0:
        conclusion_lines.append("WARNING: brainsets train_domain contains NLB test trials! Comparability with NLB leaderboard is compromised.")
    else:
        conclusion_lines.append("OK: brainsets train_domain does not contain NLB test trials.")

    split_comparison["conclusion"] = conclusion_lines
    for line in conclusion_lines:
        print(f"    {line}")

    adaptability["split_consistency"] = split_comparison

    # --- B2: Held-in/held-out mechanism ---
    print("\n  --- B2: Held-in/held-out 机制分析 ---")

    # Note: unit IDs (string paths) differ between train/test due to electrode remapping.
    # Use unit_number (integer) for correct matching.
    train_unit_nums = set(train_data["units"]["unit_number"].tolist())
    test_unit_nums = set(test_data["units"]["unit_number"].tolist())

    held_in = train_unit_nums & test_unit_nums
    held_out = train_unit_nums - test_unit_nums

    # Also collect string IDs for reference
    train_unit_ids = set()
    for uid in train_data["units"]["id"]:
        train_unit_ids.add(uid.decode() if isinstance(uid, bytes) else str(uid))
    test_unit_ids = set()
    for uid in test_data["units"]["id"]:
        test_unit_ids.add(uid.decode() if isinstance(uid, bytes) else str(uid))

    held_in_out_analysis = {
        "train_units": len(train_unit_nums),
        "test_units": len(test_unit_nums),
        "held_in_units": len(held_in),
        "held_out_units": len(held_out),
        "explanation": (
            "NLB uses a held-in/held-out neuron design: "
            f"train file has {len(train_unit_ids)} units (all neurons), "
            f"test file has {len(test_unit_ids)} units (held-in only, {len(held_out)} held-out neurons removed). "
            "The primary NLB task (co-smoothing, co-bps) requires predicting held-out neuron activity "
            "given only held-in neuron observations. This is NOT the same as behavior decoding."
        ),
        "impact_on_neurohorizon": (
            "NeuroHorizon currently trains on ALL neurons and evaluates behavior decoding (R-squared). "
            "To properly compare with NLB co-bps, would need to: "
            "(1) train only on held-in neurons, (2) predict held-out neuron firing rates, "
            "(3) compute bits-per-spike metric. This requires significant pipeline changes."
        ),
    }
    print(f"    Train units: {len(train_unit_nums)}")
    print(f"    Test units: {len(test_unit_nums)}")
    print(f"    Held-in (by unit_number): {len(held_in)}")
    print(f"    Held-out (by unit_number): {len(held_out)}")
    print(f"    Unit number overlap verified: {held_in == test_unit_nums}")
    print(f"    Note: string unit IDs differ between train/test (electrode remapping)")
    adaptability["held_in_out"] = held_in_out_analysis

    # --- B3: nlb_eval_intervals analysis ---
    print("\n  --- B3: nlb_eval_intervals 分析 ---")
    eval_starts = train_data["nlb_eval_intervals"]["start"]
    eval_ends = train_data["nlb_eval_intervals"]["end"]
    eval_durations = eval_ends - eval_starts
    eval_train_mask = train_data["nlb_eval_intervals"]["train_mask"]
    eval_test_mask = train_data["nlb_eval_intervals"]["test_mask"]
    eval_valid_mask = train_data["nlb_eval_intervals"]["valid_mask"]

    eval_analysis = {
        "n_intervals": int(len(eval_starts)),
        "duration_mean_s": round(float(np.mean(eval_durations)), 4),
        "duration_std_s": round(float(np.std(eval_durations)), 4),
        "duration_min_s": round(float(np.min(eval_durations)), 4),
        "duration_max_s": round(float(np.max(eval_durations)), 4),
        "n_train_eval": int(np.sum(eval_train_mask)),
        "n_valid_eval": int(np.sum(eval_valid_mask)),
        "n_test_eval": int(np.sum(eval_test_mask)),
        "explanation": (
            "nlb_eval_intervals defines 100 evaluation windows (one per trial). "
            "These correspond to specific time segments within each trial used for NLB metric computation. "
            "The train/valid/test masks on these intervals follow brainsets split."
        ),
    }
    print(f"    n_intervals: {eval_analysis['n_intervals']}")
    print(f"    duration: {eval_analysis['duration_mean_s']}+/-{eval_analysis['duration_std_s']}s "
          f"(range: {eval_analysis['duration_min_s']}-{eval_analysis['duration_max_s']}s)")
    print(f"    train/valid/test: {eval_analysis['n_train_eval']}/{eval_analysis['n_valid_eval']}/{eval_analysis['n_test_eval']}")
    adaptability["nlb_eval_intervals"] = eval_analysis

    # --- B4: Other NLB sub-datasets ---
    print("\n  --- B4: 其他 NLB 子数据集适配性 ---")
    other_datasets = {
        "MC_RTT": {
            "description": "Random Target Task, single session, single area (M1)",
            "brainsets_support": "Unknown - not in current brainsets datasets list",
            "adaptation_difficulty": "Medium - similar spike format, different task structure",
        },
        "Area2_Bump": {
            "description": "Bump perturbation task, somatosensory area 2",
            "brainsets_support": "Unknown - not in current brainsets datasets list",
            "adaptation_difficulty": "Medium-High - different brain area, different task",
        },
        "DMFC_RSG": {
            "description": "Ready-Set-Go timing task, DMFC (dorsomedial frontal cortex)",
            "brainsets_support": "Unknown - not in current brainsets datasets list",
            "adaptation_difficulty": "High - very different task paradigm, timing-based",
        },
        "MC_Cycle": {
            "description": "Cycling movement task, motor cortex",
            "brainsets_support": "Unknown - not in current brainsets datasets list",
            "adaptation_difficulty": "Medium - similar brain area but periodic movement",
        },
    }
    adaptability["other_nlb_datasets"] = other_datasets
    for name, info in other_datasets.items():
        print(f"    {name}: {info['adaptation_difficulty']}")

    # --- Figure 07: Split comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("NLB MC_Maze: Brainsets Split vs NLB Original Split Comparison",
                 fontsize=14, fontweight="bold")

    # (0,0) Trial-by-trial split comparison
    ax = axes[0, 0]
    trial_matrix = np.zeros((n_trials, 2))
    for i in range(n_trials):
        if train_mask_nwb[i]:
            trial_matrix[i, 0] = 1
        elif test_mask_nwb[i]:
            trial_matrix[i, 0] = 2
    for i in range(n_trials):
        if i in set(td_from_trials):
            trial_matrix[i, 1] = 1
        elif i in set(vd_from_trials):
            trial_matrix[i, 1] = 2
        elif i in set(testd_from_trials):
            trial_matrix[i, 1] = 3

    nlb_colors = {0: "#CCCCCC", 1: "#4CAF50", 2: "#F44336"}
    bs_colors = {0: "#CCCCCC", 1: "#4CAF50", 2: "#FF9800", 3: "#F44336"}

    for i in range(n_trials):
        ax.barh(i, 0.4, left=0, color=nlb_colors[int(trial_matrix[i, 0])], alpha=0.8)
        ax.barh(i, 0.4, left=0.5, color=bs_colors[int(trial_matrix[i, 1])], alpha=0.8)

    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0.2, 0.7])
    ax.set_xticklabels(["NLB Split", "Brainsets Split"])
    ax.set_ylabel("Trial Index")
    ax.set_title("Trial-by-Trial Split Assignment")
    ax.invert_yaxis()
    nlb_patches = [mpatches.Patch(color="#4CAF50", label="NLB train"),
                   mpatches.Patch(color="#F44336", label="NLB test"),
                   mpatches.Patch(color="#CCCCCC", label="Neither")]
    ax.legend(handles=nlb_patches, loc="lower right", fontsize=7)

    # (0,1) Cross-mapping matrix
    ax = axes[0, 1]
    cross_matrix = np.array([
        [bs_train_in_nwb_train, bs_train_in_nwb_test,
         n_bs_train - bs_train_in_nwb_train - bs_train_in_nwb_test],
        [bs_valid_in_nwb_train, bs_valid_in_nwb_test,
         n_bs_valid - bs_valid_in_nwb_train - bs_valid_in_nwb_test],
        [bs_test_in_nwb_train, bs_test_in_nwb_test,
         n_bs_test - bs_test_in_nwb_train - bs_test_in_nwb_test],
    ])
    im = ax.imshow(cross_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["NLB Train", "NLB Test", "NLB Neither"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["BS Train", "BS Valid", "BS Test"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(int(cross_matrix[i, j])),
                    ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title("Cross-Mapping: Brainsets vs NLB Splits")
    plt.colorbar(im, ax=ax, label="# Trials")

    # (1,0) nlb_eval_intervals timeline with split coloring
    ax = axes[1, 0]
    for i in range(len(eval_starts)):
        if eval_train_mask[i]:
            c = "#4CAF50"
        elif eval_valid_mask[i]:
            c = "#FF9800"
        elif eval_test_mask[i]:
            c = "#F44336"
        else:
            c = "#CCCCCC"
        ax.barh(0, eval_ends[i] - eval_starts[i], left=eval_starts[i],
                height=0.5, color=c, alpha=0.7)
    ax.set_yticks([0])
    ax.set_yticklabels(["eval_intervals"])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"NLB Eval Intervals (n={len(eval_starts)}, "
                 f"duration: {np.mean(eval_durations):.3f}+/-{np.std(eval_durations):.3f}s)")
    eval_patches = [
        mpatches.Patch(color="#4CAF50", label=f"Train ({int(np.sum(eval_train_mask))})"),
        mpatches.Patch(color="#FF9800", label=f"Valid ({int(np.sum(eval_valid_mask))})"),
        mpatches.Patch(color="#F44336", label=f"Test ({int(np.sum(eval_test_mask))})")
    ]
    ax.legend(handles=eval_patches, loc="upper right")

    # (1,1) Conclusions text panel
    ax = axes[1, 1]
    ax.axis("off")
    conclusion_text = [
        "Held-in/Held-out Mechanism:",
        f"  - Train file: {len(train_unit_ids)} units (all neurons)",
        f"  - Test file: {len(test_unit_ids)} units (held-in only)",
        f"  - Held-out: {len(held_out)} neurons removed from test",
        "",
        "Split Consistency Findings:",
    ]
    for line in conclusion_lines:
        conclusion_text.append(f"  - {line}")
    conclusion_text.extend([
        "",
        "Comparability Assessment:",
        "  - Brainsets behavior decoding (R-sq) uses all neurons",
        "  - NLB co-bps requires held-out neuron prediction",
        "  - Direct metric comparison NOT possible without",
        "    implementing NLB-specific evaluation pipeline",
        "  - nlb_eval_intervals config in POYO enables",
        "    behavior decoding on NLB-matched time windows",
    ])
    ax.text(0.05, 0.95, "\n".join(conclusion_text), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))
    ax.set_title("Analysis Conclusions", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "07_nlb_split_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")

    return adaptability


def part_c_metrics_alignment():
    """Part C: NLB 指标对齐分析"""
    print("\n" + "="*60)
    print("Part C: NLB 指标对齐")
    print("="*60)

    metrics_analysis = {
        "co_bps": {
            "full_name": "Co-smoothing Bits Per Spike",
            "description": (
                "Primary NLB metric. Measures how well a model predicts held-out neuron "
                "firing rates given held-in neuron observations. Computed as: "
                "bps = (log_likelihood_model - log_likelihood_null) / (n_spikes * ln(2)). "
                "Null model uses neuron-specific mean firing rate."
            ),
            "implementation_difficulty": "High",
            "required_changes": [
                "Need held-in/held-out neuron split during training",
                "Need to predict held-out neuron firing rates from held-in observations",
                "Need Poisson log-likelihood computation per neuron",
                "Need null model (mean firing rate per neuron) baseline",
                "Requires nlb_tools.evaluation module for standardized computation",
            ],
        },
        "fp_bps": {
            "full_name": "Forward Prediction Bits Per Spike",
            "description": (
                "Forward prediction task. Given neural activity up to time t, predict "
                "future activity at time t+delta. Uses same bps formula as co-bps but "
                "applied to forward prediction rather than co-smoothing."
            ),
            "implementation_difficulty": "Medium-High",
            "required_changes": [
                "More aligned with NeuroHorizon autoregressive prediction",
                "Need to adapt prediction format to NLB evaluation intervals",
                "Need null model baseline computation",
                "Need nlb_tools for standardized bps computation",
            ],
        },
        "psth_r2": {
            "full_name": "PSTH R-squared (Peri-Stimulus Time Histogram)",
            "description": (
                "Trial-averaged prediction quality. Compute R-squared between trial-averaged "
                "predicted firing rates and trial-averaged true firing rates, averaged "
                "across neurons. Less noisy than single-trial metrics."
            ),
            "implementation_difficulty": "Low-Medium",
            "required_changes": [
                "Already have R-squared computation in NeuroHorizon",
                "Need trial-averaging logic (group by condition/trial type)",
                "Need per-neuron R-squared computation (currently per-behavior-variable)",
                "Most compatible with current NeuroHorizon evaluation",
            ],
        },
    }

    nlb_tools_status = {
        "installed": False,
        "installation_command": "pip install nlb-tools",
        "github": "https://github.com/neurallatents/nlb_tools",
        "key_modules": [
            "nlb_tools.evaluation: standardized metric computation",
            "nlb_tools.nwb_interface: NWB file loading utilities",
            "nlb_tools.make_tensors: data preprocessing for NLB format",
        ],
    }

    overall_assessment = {
        "behavior_decoding_comparability": (
            "NeuroHorizon's current behavior decoding (hand velocity R-squared) on NLB data "
            "CAN be compared with other models' behavior decoding results, "
            "since the POYO NLB config already uses nlb_eval_intervals for evaluation. "
            "However, this is NOT the primary NLB benchmark metric."
        ),
        "co_bps_comparability": (
            "Achieving NLB leaderboard-comparable co-bps requires: "
            "(1) held-in/held-out neuron split, (2) co-smoothing prediction task, "
            "(3) nlb_tools evaluation. This is a fundamentally different task from "
            "NeuroHorizon's current autoregressive spike prediction."
        ),
        "recommended_path": (
            "For paper benchmarking, recommend: "
            "(1) Report behavior decoding R-squared on NLB eval intervals (easy, already supported), "
            "(2) Implement fp-bps as closest metric to NeuroHorizon autoregressive task (medium effort), "
            "(3) co-bps only if competitive advantage is clear (high effort)."
        ),
    }

    for metric_name, info in metrics_analysis.items():
        print(f"\n  {metric_name} ({info['full_name']}):")
        print(f"    Difficulty: {info['implementation_difficulty']}")
        for change in info["required_changes"]:
            print(f"      - {change}")

    print(f"\n  nlb_tools installed: {nlb_tools_status['installed']}")
    print(f"\n  Overall assessment:")
    print(f"    {overall_assessment['recommended_path']}")

    return {
        "metrics": metrics_analysis,
        "nlb_tools": nlb_tools_status,
        "overall_assessment": overall_assessment,
    }


def main():
    print("="*60)
    print("Phase 0.4.1: NLB Benchmark 数据分析与适配性调查")
    print("="*60)

    # Load data
    print("\nLoading NLB data...")
    train_data = load_h5_data(TRAIN_PATH)
    test_data = load_h5_data(TEST_PATH)
    print(f"  Train: {TRAIN_PATH}")
    print(f"  Test: {TEST_PATH}")

    # Part A
    stats = part_a_data_structure(train_data, test_data)

    # Part B
    adaptability = part_b_adaptability(train_data, test_data, stats)

    # Part C
    metrics = part_c_metrics_alignment()

    # Save summary JSON
    summary = {
        "stats": stats,
        "adaptability": adaptability,
        "metrics_alignment": metrics,
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode()
        elif isinstance(obj, set):
            return list(obj)
        return obj

    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(v) for v in obj]
        else:
            return convert(obj)

    summary = recursive_convert(summary)
    json_path = OUTPUT_DIR / "nlb_analysis_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary JSON saved: {json_path}")

    print("\n" + "="*60)
    print("Phase 0.4.1 COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
