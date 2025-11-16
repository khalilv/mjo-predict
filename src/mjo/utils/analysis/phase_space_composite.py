import numpy as np
import os
import argparse
import pandas as pd
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import load_config
from mjo.utils.plot import phase_space_composite_plot
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Optional, Tuple

PHASE_GROUP_MAP = {
    1: 1, 8: 8,
    2: 2, 3: 3,
    4: 4, 5: 5,
    6: 6, 7: 7
}


def compute_composite_by_phase(
    predict_dir: str,
    gt_df: pd.DataFrame,
    member: Optional[str] = None,
    trajectory_length: int = 42,
    window: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    offset: int = 0
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Compute composite trajectories grouped by initialization phase."""
    pred_phase_groups = defaultdict(list)
    gt_phase_groups = defaultdict(list)

    start_date_dt = None
    end_date_dt = None
    if start_date is not None:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date is not None:
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    for fname in tqdm(sorted(os.listdir(predict_dir)), f'Loading composites for {predict_dir}'):
        if member is not None:
            init_date = datetime.strptime(fname, "%Y-%m-%d")
            init_date_str = str(init_date).split(' ')[0]
            file = os.path.join(predict_dir, init_date_str, f'{member}.txt')
        else:
            init_date = datetime.strptime(fname, "%Y-%m-%d.txt")
            init_date_str = str(init_date).split(' ')[0]
            file = os.path.join(predict_dir, f'{init_date_str}.txt')

        if start_date_dt is not None and init_date < start_date_dt:
            continue

        if end_date_dt is not None and init_date >= end_date_dt:
            continue

        phase = int(gt_df.loc[init_date].phase)
        day_1 = init_date + timedelta(days=1)
        last_day = init_date + timedelta(days=trajectory_length)
        gt_slice = gt_df.loc[day_1:last_day]

        pred_df = load_rmm_indices(file)

        # Apply smoothing if window > 1
        if window > 1:
            pred_df_smoothed = pred_df.rolling(window=window, center=False).mean()
            pred_df_smoothed[:window - 1] = pred_df[:window - 1]
            pred_df = pred_df_smoothed

            gt_slice_smoothed = gt_slice.rolling(window=window, center=False).mean()
            gt_slice_smoothed[:window - 1] = gt_slice[:window - 1]
            gt_slice = gt_slice_smoothed

        if len(pred_df) < trajectory_length:
            raise ValueError(
                f"Prediction dataframe in file: {file} has length: {len(pred_df)}, "
                f"which is less than the required trajectory length: {trajectory_length}."
            )
        pred_df = pred_df.iloc[:trajectory_length]  # clip to trajectory length

        assert not gt_slice.isnull().values.any(), "NaNs found in ground truth dataframe slice"
        assert not pred_df.isnull().values.any(), f"NaNs found in predict file: {file}"
        assert (gt_slice.index == pred_df.index).all(), \
            "Ground truth dataframe slice and predict dataframe slice do not contain the same dates"

        pred_phase_groups[PHASE_GROUP_MAP[phase]].append(pred_df[offset:])
        gt_phase_groups[PHASE_GROUP_MAP[phase]].append(gt_slice[offset:])

    # Compute composite trajectories by averaging
    pred_composite_trajectories = {}
    gt_composite_trajectories = {}
    for phase, dfs in pred_phase_groups.items():
        pred_rmm1 = np.stack([df.RMM1.values for df in dfs])
        pred_rmm2 = np.stack([df.RMM2.values for df in dfs])
        gt_rmm1 = np.stack([df.RMM1.values for df in gt_phase_groups[phase]])
        gt_rmm2 = np.stack([df.RMM2.values for df in gt_phase_groups[phase]])
        pred_composite_trajectories[phase] = (pred_rmm1.mean(axis=0), pred_rmm2.mean(axis=0))
        gt_composite_trajectories[phase] = (gt_rmm1.mean(axis=0), gt_rmm2.mean(axis=0))

    return pred_composite_trajectories, gt_composite_trajectories

def main():
    """Main function to generate phase space composite plot."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate phase space composite plot')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract configuration
    start_date = config['dates']['start']
    end_date = config['dates']['end']
    trajectory_length = config['trajectory']['length']
    smoothing_window = config['trajectory'].get('smoothing_window', 1)
    offset = config['trajectory'].get('offset', 0)
    forecast_dirs = config.get('forecast_dirs', [])
    forecast_labels = config.get('forecast_labels', [])
    ensemble_dirs = config.get('ensemble_dirs', [])
    ground_truth_path = config['paths']['ground_truth']
    bom_path = config['paths'].get('bom')
    output_dir = config['paths']['output_dir']
    output_filename = config['paths'].get('output_filename', 'phase_space_composite.png')
    os.makedirs(output_dir, exist_ok=True)

    # Plot settings
    plot_settings = config.get('plot_settings', {})
    font_settings = plot_settings.get('fonts', {})
    axis_settings = plot_settings.get('axis', {})
    colormap = plot_settings.get('colormap', {})
    title_template = plot_settings.get('title', {}).get('template', '')

    # Build plot kwargs
    plot_kwargs = {
        'fontsize_title': font_settings.get('title', 14),
        'fontsize_axis_label': font_settings.get('axis_label', 12),
        'fontsize_region_label': font_settings.get('region_label', 11),
        'fontsize_phase_label': font_settings.get('phase_label', 11),
        'fontsize_legend': font_settings.get('legend', 11),
        'fontweight_title': font_settings.get('title_weight', 'bold'),
        'fontweight_title_label': font_settings.get('title_label_weight'),
        'font_family': font_settings.get('family'),
        'figsize': tuple(plot_settings.get('figsize', [8, 8])),
        'xlim': tuple(axis_settings.get('xlim', [-1.75, 1.75])),
        'ylim': tuple(axis_settings.get('ylim', [-1.75, 1.75])),
        'colormap': colormap.get('name', 'tab10'),
        'marker_interval': plot_settings.get('marker_interval', 5),
        'legend_loc': plot_settings.get('legend_loc', 'best'),
    }

    # Load ground truth
    gt_df = load_rmm_indices(ground_truth_path)

    # Load BoM composites if configured
    bom_composites = None
    if bom_path:
        bom_composites, _ = compute_composite_by_phase(
            bom_path,
            gt_df,
            member=None,
            trajectory_length=trajectory_length,
            window=smoothing_window,
            start_date=start_date,
            end_date=end_date,
            offset=offset
        )

    # Load forecast composites
    pred_composites_per_dir = []
    gt_composites_per_dir = []

    for forecast_dir in forecast_dirs:
        pred_composites, gt_composites = compute_composite_by_phase(
            forecast_dir,
            gt_df,
            member=None,
            trajectory_length=trajectory_length,
            window=smoothing_window,
            start_date=start_date,
            end_date=end_date,
            offset=offset
        )
        pred_composites_per_dir.append(pred_composites)
        gt_composites_per_dir.append(gt_composites)

    # Load ensemble composites
    ensemble_member_labels = []
    for ensemble_config in ensemble_dirs:
        ensemble_dir = ensemble_config['path']
        ensemble_label = ensemble_config['label']
        ensemble_members = ensemble_config.get('members', ['mean'])

        for member in ensemble_members:
            pred_composites, gt_composites = compute_composite_by_phase(
                ensemble_dir,
                gt_df,
                member=member,
                trajectory_length=trajectory_length,
                window=smoothing_window,
                start_date=start_date,
                end_date=end_date,
                offset=offset
            )
            pred_composites_per_dir.append(pred_composites)
            gt_composites_per_dir.append(gt_composites)
            ensemble_member_labels.append(f'{ensemble_label} {member}' if member != 'mean' else f'{ensemble_label}')

    # Generate title from template
    title = title_template.format(
        trajectory_length=trajectory_length,
        offset=offset,
        start_date=start_date,
        end_date=end_date
    ) if title_template else None

    # Generate plot
    phase_space_composite_plot(
        pred_composites=pred_composites_per_dir,
        gt_composites=gt_composites_per_dir[0] if gt_composites_per_dir else {},
        labels=forecast_labels + ensemble_member_labels,
        gt_label=config.get('gt_label', 'Observed'),
        bom_composites=bom_composites,
        title=title,
        output_filename=os.path.join(output_dir, output_filename),
        **plot_kwargs
    )


if __name__ == "__main__":
    main()
