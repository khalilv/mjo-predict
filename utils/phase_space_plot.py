import pandas as pd
from rmm_io import load_rmm_indices
from matplotlib import pyplot as plt
import numpy as np
import math

def smooth(df, window=3):
    return df.rolling(window=window, center=True).mean()

def draw_phase_wedges():
    r = math.sqrt(8)
    labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

    for i in range(8):
        angle = (5 + i) * np.pi / 4  
        x = np.cos(angle)
        y = np.sin(angle)
        plt.plot([0, r * x], [0, r * y], color='lightgray', lw=0.8, zorder=0)

    for i, label in enumerate(labels):
        angle = (4.5 + i) * np.pi / 4
        x = 1.7 * np.cos(angle)
        y = 1.7 * np.sin(angle)
        plt.text(x, y, label, ha='center', va='center', fontsize=9, color='gray')

def add_region_labels():
    plt.text(0, -1.85, "Indian\nOcean", ha='center', va='center', fontsize=9)
    plt.text(1.85, 0, "Maritime\nContinent", ha='center', va='center', fontsize=9, rotation=-90)
    plt.text(0, 1.85, "Western\nPacific", ha='center', va='center', fontsize=9)
    plt.text(-1.85, 0, "West. Hem.\n& Africa", ha='center', va='center', fontsize=9, rotation=90)


def plot_phase_space(gt_df, pred_df, start_dates, trajectory_len=14):
    plt.figure(figsize=(8, 8))

    for start_date in start_dates:
        start = pd.to_datetime(start_date)
        end = start + pd.Timedelta(days=trajectory_len - 1)

        gt_traj = gt_df.loc[start:end]
        pred_traj = pred_df.loc[start:end]

        if len(gt_traj) != trajectory_len or len(pred_traj) != trajectory_len:
            continue
        if gt_traj.isna().any().any() or pred_traj.isna().any().any():
            continue

        # ground truth in black
        plt.plot(gt_traj['RMM1'], gt_traj['RMM2'], 'k-', alpha=0.8, label='ABM')
        plt.plot(gt_traj['RMM1'].iloc[0], gt_traj['RMM2'].iloc[0], 'ko')  # start
        plt.plot(gt_traj['RMM1'].iloc[-1], gt_traj['RMM2'].iloc[-1], 'k*')  # end

        # prediction in red
        plt.plot(pred_traj['RMM1'], pred_traj['RMM2'], 'r-', alpha=0.8, label='Ours')
        plt.plot(pred_traj['RMM1'].iloc[0], pred_traj['RMM2'].iloc[0], 'ro')  # start
        plt.plot(pred_traj['RMM1'].iloc[-1], pred_traj['RMM2'].iloc[-1], 'r*')  # end

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    draw_phase_wedges()
    add_region_labels()

    plt.xlabel("RMM1")
    plt.ylabel("RMM2")
    plt.title(f"{trajectory_len}-day trajectory phase space diagram")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal')

    # prevent duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    plt.show()

predict_filepath = '/glade/derecho/scratch/kvirji/DATA/MJO/RMM/reference_period_1979-09-07_to_2001-12-31/rmm.txt'
ground_truth_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ABM/rmm.74toRealtime.txt"

start_dates = ["2005-01-01", "2009-03-01", "2020-02-01"]
trajectory_length = 14

predict_df = load_rmm_indices(predict_filepath, 2000)
ground_truth_df = load_rmm_indices(ground_truth_filepath, 2000)

plot_phase_space(ground_truth_df, predict_df, start_dates, trajectory_length)
