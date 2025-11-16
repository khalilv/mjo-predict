import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from mjo.utils.analysis.utils import load_forecast, load_config, compute_bcor
from mjo.utils.RMM.io import load_rmm_indices
from typing import List, Callable
from mjo.utils.plot import (
    bivariate_correlation_vs_lead_time_heatmap,
)

def compute_metric_across_leads(
    dataframes: List[pd.DataFrame],
    max_lt: int,
    ground_truth_df: pd.DataFrame,
    metric_fn: Callable,
    filter_low_amp_init: bool = False
) -> np.ndarray:
    """Compute a metric (bcor or bmse) across all lead times for forecasts."""
    results = []
    for lt in tqdm(range(max_lt), 'Computing per-lead metric'):
        preds, truths = [], []
        for i, df in enumerate(dataframes):
            if filter_low_amp_init:
                init_date = df.index[0] - timedelta(days=1)
                init_amplitude = ground_truth_df.loc[init_date].amplitude
                if init_amplitude < 1:
                    continue
            if lt < len(df):
                preds.append(df.iloc[lt])
                truths.append(ground_truth_df.loc[df.index[lt]])
        pred_df = pd.DataFrame(preds)
        truth_df = pd.DataFrame(truths)
        result = metric_fn(pred_df.RMM1.values, truth_df.RMM1.values, pred_df.RMM2.values, truth_df.RMM2.values)
        results.append(result)
    return np.array(results)


def validate_and_flatten_dirs(
    forecast_dirs: List[List[str]],
    model_names: List[str],
    lookback_names: List
) -> List[str]:
    """Validate forecast_dirs shape matches (models × lookbacks) and flatten."""
    num_models = len(model_names)
    num_lookbacks = len(lookback_names)

    if len(forecast_dirs) != num_models:
        raise ValueError(
            f"forecast_dirs has {len(forecast_dirs)} models, "
            f"but model_names has {num_models}. These must match."
        )

    for i, model_dirs in enumerate(forecast_dirs):
        if len(model_dirs) != num_lookbacks:
            raise ValueError(
                f"Model '{model_names[i]}' has {len(model_dirs)} directories, "
                f"but lookback_names has {num_lookbacks}. These must match."
            )

    # Flatten to 1D list (all models concatenated)
    flat_dirs = [dir_path for model_dirs in forecast_dirs for dir_path in model_dirs]
    return flat_dirs

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate lead time heatmaps for MJO predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file (required)'
    )
    parser.add_argument('--start-date', type=str, help='Override start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Override end date (YYYY-MM-DD)')
    parser.add_argument('--ground-truth-path', type=str, help='Override ground truth RMM data path')
    parser.add_argument('--fuxi-path', type=str, help='Override FuXi forecast data path')
    parser.add_argument('--output-dir', type=str, help='Override output directory')

    return parser.parse_args()

def main() -> None:
    """Load config, compute correlations, and generate heatmap plots."""
    args = parse_args()

    # Load config from file (required)
    config = load_config(args.config)

    # Command line args override config
    start_date = args.start_date or config['dates']['start']
    end_date = args.end_date or config['dates']['end']
    ground_truth_path = args.ground_truth_path or config['paths']['ground_truth']
    fuxi_path = args.fuxi_path or config['paths']['fuxi']
    output_dir = args.output_dir or config['paths']['output_dir']

    # Get model and lookback info
    model_names = config['model_names']
    lookback_names = config['lookback_names']
    include_fuxi = config.get('include_fuxi', False)

    # Validate and flatten forecast directories
    forecast_dirs_flat = validate_and_flatten_dirs(
        config['forecast_dirs'],
        model_names,
        lookback_names
    )
    lookback_array = np.array(lookback_names)

    os.makedirs(output_dir, exist_ok=True)

    ground_truth = load_rmm_indices(ground_truth_path)

    correlations = []
    event_correlations = []
    max_lead_times = []


    for predict_dir in forecast_dirs_flat:
        dfs, max_lt, date_strs = load_forecast(predict_dir, start_date, end_date)
        bcor = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcor)
        bcor_events = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcor, filter_low_amp_init=True)
        correlations.append(bcor)
        max_lead_times.append(max_lt)
        event_correlations.append(bcor_events)

    # Validate all forecasts have the same lead time
    assert len(set(max_lead_times)) == 1, f"All forecasts must have the same lead time length, found: {set(max_lead_times)}"
    max_lt = max_lead_times[0]

    fuxi_ds, fuxi_max_lt, _ = load_forecast(fuxi_path, start_date, end_date, member='mean')
    bcor_fuxi = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcor)
    bcor_fuxi_events = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcor, filter_low_amp_init=True)

    # Extract plot settings from config
    plot_settings = config.get('plot_settings', {})
    fonts = plot_settings.get('fonts', {})
    labels = plot_settings.get('labels', {})
    colormap = plot_settings.get('colormap', {})
    threshold = plot_settings.get('threshold', 0.5)
    legend_position = plot_settings.get('legend_position', 'upper right')

    # Prepare common plotting parameters
    num_models = len(model_names)
    num_lookbacks = len(lookback_array)
    plot_kwargs = {
        'threshold': threshold,
        'fontsize_title': fonts.get('title', 14),
        'fontsize_axis_label': fonts.get('axis_label', 12),
        'fontsize_tick_label': fonts.get('tick_label', 11),
        'fontsize_legend': fonts.get('legend', 11),
        'fontsize_colorbar_label': fonts.get('colorbar_label', 12),
        'fontweight_title': fonts.get('title_weight', 'bold'),
        'fontweight_title_label': fonts.get('title_label_weight'),
        'font_family': fonts.get('family'),
        'x_label': labels.get('x_label', 'Lookback window (days)'),
        'y_label': labels.get('y_label', 'Lead time (days)'),
        'colorbar_label': labels.get('colorbar_label', 'Bivariate correlation'),
        'legend_label_template': labels.get('legend_label', '{threshold} skill threshold'),
        'zero_symbol': labels.get('zero_symbol', r'$\varnothing$'),
        'fuxi_label': labels.get('fuxi_label', 'FuXi'),
        'cmap': colormap.get('name', 'YlGnBu'),
        'color_bounds': colormap.get('bounds', None),
        'legend_loc': legend_position,
    }

    # Add FuXi correlations if requested
    if include_fuxi:
        plot_kwargs['fuxi_correlations'] = bcor_fuxi

    bivariate_correlation_vs_lead_time_heatmap(
        lead_times=np.expand_dims(np.arange(1, max_lt + 1), 0).repeat(num_models, axis=0),
        lookbacks=np.expand_dims(lookback_array, 0).repeat(num_models, axis=0),
        correlations=np.array(correlations).reshape(num_models, num_lookbacks, max_lt),
        labels=model_names,
        output_filename=os.path.join(output_dir, 'bcor_heatmap.png'),
        **plot_kwargs
    )

    # Update FuXi correlations for events plot if needed
    if include_fuxi:
        plot_kwargs['fuxi_correlations'] = bcor_fuxi_events

    bivariate_correlation_vs_lead_time_heatmap(
        lead_times=np.expand_dims(np.arange(1, max_lt + 1), 0).repeat(num_models, axis=0),
        lookbacks=np.expand_dims(lookback_array, 0).repeat(num_models, axis=0),
        correlations=np.array(event_correlations).reshape(num_models, num_lookbacks, max_lt),
        labels=model_names,
        output_filename=os.path.join(output_dir, 'bcor_high_amp_events_heatmap.png'),
        **plot_kwargs
    )

if __name__ == "__main__":
    main()