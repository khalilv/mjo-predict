import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import load_forecast, load_config, compute_bcor
from typing import List, Dict, Callable
from mjo.utils.plot import bivariate_correlation_by_month_plot


def compute_metric_across_months(
    dataframes: List[pd.DataFrame],
    max_lt: int,
    ground_truth_df: pd.DataFrame,
    metric_fn: Callable
) -> Dict[int, np.ndarray]:
    """Compute a metric across all lead times, grouped by month of initialization."""
    monthly_groups = defaultdict(list)
    monthly_results = defaultdict()

    # Group forecasts by initialization month
    for df in dataframes:
        init_date = df.index[0] - timedelta(days=1)
        monthly_groups[init_date.month].append(df)

    # Compute metric for each month
    for month, dfs in tqdm(monthly_groups.items(), 'Computing per-month metric'):
        results = []
        for lt in range(max_lt):
            preds, truths = [], []
            for df in dfs:
                if lt < len(df):
                    preds.append(df.iloc[lt])
                    truths.append(ground_truth_df.loc[df.index[lt]])
            pred_df = pd.DataFrame(preds)
            truth_df = pd.DataFrame(truths)
            result = metric_fn(pred_df.RMM1.values, truth_df.RMM1.values, pred_df.RMM2.values, truth_df.RMM2.values)
            results.append(result)
        monthly_results[month] = np.array(results)

    return monthly_results


def main():
    """Main function to compute and plot monthly errors."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate monthly error plots')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract configuration
    start_date = config['dates']['start']
    end_date = config['dates']['end']
    forecast_dirs = config['forecast_dirs']
    plot_labels = config['plot_labels']
    ground_truth_path = config['paths']['ground_truth']
    fuxi_path = config['paths'].get('fuxi')
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Plot settings
    plot_settings = config.get('plot_settings', {})
    font_settings = plot_settings.get('fonts', {})
    colormap = plot_settings.get('colormap', {})
    labels = plot_settings.get('labels', {})

    # Build plot kwargs
    plot_kwargs = {
        'fontsize_title': font_settings.get('title', 14),
        'fontsize_axis_label': font_settings.get('axis_label', 12),
        'fontsize_tick_label': font_settings.get('tick_label', 11),
        'fontsize_colorbar_label': font_settings.get('colorbar_label', 12),
        'fontweight_title': font_settings.get('title_weight', 'bold'),
        'fontweight_title_label': font_settings.get('title_label_weight'),
        'font_family': font_settings.get('family'),
        'cmap': colormap.get('name', 'YlGnBu'),
        'threshold': colormap.get('threshold', 0.5),
        'x_label': labels.get('x_label', 'Month'),
        'y_label': labels.get('y_label', 'Lead time (days)'),
        'colorbar_label': labels.get('colorbar_label', 'Bivariate correlation'),
    }

    # Handle levels if provided
    if 'levels' in colormap:
        levels = colormap['levels']
        if isinstance(levels, dict):
            # If levels is a dict with start, stop, num
            plot_kwargs['levels'] = np.linspace(levels['start'], levels['stop'], levels['num'])
        else:
            plot_kwargs['levels'] = np.array(levels)

    # Extract y-axis label settings
    y_labels = plot_settings.get('y_labels', {})
    if y_labels.get('hide_x', False):
        plot_kwargs['x_label'] = ''
    if y_labels.get('hide_y', False):
        plot_kwargs['y_label'] = ''
    if y_labels.get('hide_colorbar', False):
        plot_kwargs['colorbar_label'] = ''

    # Load ground truth
    ground_truth = load_rmm_indices(ground_truth_path)

    # Process each forecast directory
    for i, forecast_dir in enumerate(forecast_dirs):
        dfs, max_lt, _ = load_forecast(forecast_dir, start_date, end_date)
        bcor_per_month = compute_metric_across_months(dfs, max_lt, ground_truth, compute_bcor)

        # Get title from config or use default
        title_template = plot_settings.get('titles', {}).get('template', '')
        title = title_template.format(model=plot_labels[i]) if title_template else None

        bivariate_correlation_by_month_plot(
            bcor_dict=bcor_per_month,
            title=title,
            output_filename=os.path.join(output_dir, f'bcor_per_month_{plot_labels[i].replace(" ", "_")}.png'),
            **plot_kwargs
        )

    # Process FuXi forecasts if available
    if fuxi_path:
        fuxi_ds, fuxi_max_lt, _ = load_forecast(fuxi_path, start_date, end_date, member='mean')
        fuxi_bcor_per_month = compute_metric_across_months(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcor)

        # Get title from config using same template
        title_template = plot_settings.get('titles', {}).get('template', '')
        title = title_template.format(model='FuXi-S2S') if title_template else None

        bivariate_correlation_by_month_plot(
            bcor_dict=fuxi_bcor_per_month,
            title=title,
            output_filename=os.path.join(output_dir, 'bcor_per_month_FuXi.png'),
            **plot_kwargs
        )


if __name__ == "__main__":
    main()
