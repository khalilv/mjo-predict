import os
import pandas as pd
import numpy as np
import argparse
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import load_forecast, load_config, compute_bcor, compute_bmse
from tqdm import tqdm
from typing import Callable, List, Tuple, Optional
from mjo.utils.plot import (
    lead_time_bcor_bmse_plot,
    lead_time_bcor_bmse_amplitude_grid,
)


def compute_metric_across_leads(
    dataframes: List[pd.DataFrame],
    max_lt: int,
    ground_truth_df: pd.DataFrame,
    metric_fn: Callable,
    fuxi_ds: Optional[List[pd.DataFrame]] = None,
    filter_type: Optional[str] = None
) -> np.ndarray:
    """Compute a metric (bcor or bmse) across all lead times for forecasts."""
    results = []
    for lt in tqdm(range(max_lt), 'Computing per-lead metric'):
        preds, truths = [], []
        for i, df in enumerate(dataframes):
            if fuxi_ds is not None:
                fuxi_df = fuxi_ds[i]
                assert filter_type is not None and filter_type in ['high', 'low'], 'Filter type high or low must be provided for filtering'
                assert (fuxi_df.index == df.index).all(), 'Found mismatch in dates between FuXi forecast and predictions'
                forecast_date = df.index[lt]
                fuxi_amplitude = fuxi_df.loc[forecast_date].amplitude
                if (filter_type == 'low' and fuxi_amplitude < 1) or (filter_type == 'high' and fuxi_amplitude >= 1):
                    continue
            if lt < len(df):
                preds.append(df.iloc[lt])
                truths.append(ground_truth_df.loc[df.index[lt]])
        pred_df = pd.DataFrame(preds)
        truth_df = pd.DataFrame(truths)
        result = metric_fn(pred_df.RMM1.values, truth_df.RMM1.values, pred_df.RMM2.values, truth_df.RMM2.values)
        results.append(result)
    return np.array(results)


def print_relative_improvement(
    metric: List[np.ndarray],
    metric_name: str,
    plot_labels: List[str],
    biweekly_groups: List[Tuple[int, int]],
    higher_is_better: bool = True
) -> None:
    """Print relative improvement table for a metric."""
    print(f"\nRelative Improvement in {metric_name} (compared to '{plot_labels[-1]}')")
    print("=" * 70)
    print("{:<25} {:<15} {:<15} {:<15}".format("Model", "Weeks 1-2", "Weeks 3-4", "Weeks 5-6"))
    print("-" * 70)
    baseline_idx = len(plot_labels) - 1
    for i in range(len(plot_labels) - 1):
        rel_improvements = []
        for start, end in biweekly_groups:
            model_mean = np.mean(metric[i][start:end+1])
            baseline_mean = np.mean(metric[baseline_idx][start:end+1])
            if higher_is_better:
                rel_imp = (model_mean - baseline_mean) / abs(baseline_mean) * 100
            else:
                rel_imp = (baseline_mean - model_mean) / abs(baseline_mean) * 100
            rel_improvements.append(rel_imp)
        print("{:<25} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            plot_labels[i], rel_improvements[0], rel_improvements[1], rel_improvements[2]
        ))
    print("=" * 70)
    if higher_is_better:
        print(f"Values are percent relative improvement in {metric_name} (higher is better).\n")
    else:
        print(f"Values are percent relative improvement in {metric_name} (lower is better).\n")


def main():
    """Main function to compute and plot lead time errors."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate lead time error plots')
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
    titles_settings = plot_settings.get('titles', {})

    # Extract font sizes and line style settings
    plot_kwargs = {
        'fontsize_title': font_settings.get('title', 14),
        'fontsize_axis_label': font_settings.get('axis_label', 12),
        'fontsize_tick_label': font_settings.get('tick_label', 11),
        'fontsize_legend': font_settings.get('legend', 11),
        'fontweight_title': font_settings.get('title_weight', 'bold'),
        'fontweight_title_label': font_settings.get('title_label_weight'),
        'font_family': font_settings.get('family'),
        'colors': plot_settings.get('colors'),
        'linestyles': plot_settings.get('linestyles'),
        'linewidth': plot_settings.get('linewidth', 2),
    }

    # Extract y-axis label settings
    y_labels = plot_settings.get('y_labels', {})
    if y_labels.get('hide_bcor', False):
        plot_kwargs['y_label'] = ''

    # Load ground truth
    ground_truth = load_rmm_indices(ground_truth_path)

    # Initialize result containers
    correlations = []
    phase_errors = []
    amplitude_errors = []
    low_amp_event_correlations = []
    low_amp_event_phase_errors = []
    low_amp_event_amplitude_errors = []
    high_amp_event_correlations = []
    high_amp_event_phase_errors = []
    high_amp_event_amplitude_errors = []
    max_lead_times = []

    # Load FuXi data if configured
    fuxi_ds = None
    fuxi_max_lt = None
    if fuxi_path:
        fuxi_ds, fuxi_max_lt, _ = load_forecast(fuxi_path, start_date, end_date, member='mean')

    # Process each forecast directory
    for predict_dir in forecast_dirs:
        dfs, max_lt, _ = load_forecast(predict_dir, start_date, end_date)

        # Compute metrics
        bcor = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcor)
        bmse = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse).T

        correlations.append(bcor)
        amplitude_errors.append(bmse[0])
        phase_errors.append(bmse[1])
        max_lead_times.append(max_lt)

        # Compute amplitude-filtered metrics if FuXi data is available
        if fuxi_ds is not None:
            bcor_low_amp = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcor, fuxi_ds, 'high')
            bmse_low_amp = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse, fuxi_ds, 'high').T
            bcor_high_amp = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcor, fuxi_ds, 'low')
            bmse_high_amp = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse, fuxi_ds, 'low').T

            low_amp_event_correlations.append(bcor_low_amp)
            low_amp_event_amplitude_errors.append(bmse_low_amp[0])
            low_amp_event_phase_errors.append(bmse_low_amp[1])
            high_amp_event_correlations.append(bcor_high_amp)
            high_amp_event_amplitude_errors.append(bmse_high_amp[0])
            high_amp_event_phase_errors.append(bmse_high_amp[1])

    # Process FuXi forecasts if available
    if fuxi_ds is not None and fuxi_max_lt is not None:
        bcor = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcor)
        bmse = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bmse).T
        bcor_low_amp = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcor, fuxi_ds, 'high')
        bmse_low_amp = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bmse, fuxi_ds, 'high').T
        bcor_high_amp = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcor, fuxi_ds, 'low')
        bmse_high_amp = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bmse, fuxi_ds, 'low').T

        correlations.append(bcor)
        amplitude_errors.append(bmse[0])
        phase_errors.append(bmse[1])
        max_lead_times.append(fuxi_max_lt)
        low_amp_event_correlations.append(bcor_low_amp)
        low_amp_event_amplitude_errors.append(bmse_low_amp[0])
        low_amp_event_phase_errors.append(bmse_low_amp[1])
        high_amp_event_correlations.append(bcor_high_amp)
        high_amp_event_amplitude_errors.append(bmse_high_amp[0])
        high_amp_event_phase_errors.append(bmse_high_amp[1])

    # Prepare lead times
    lead_times = [np.arange(1, max_lt + 1) for max_lt in max_lead_times]

    # Generate skill plot (BCOR and BMSE side by side)
    lead_time_bcor_bmse_plot(
        lead_times=lead_times,
        correlations=correlations,
        bmsea=amplitude_errors,
        bmsep=phase_errors,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'lead_time_bcor_bmse.png'),
        titles=(titles_settings.get('bcor'), titles_settings.get('bmse')),
        combined_bmse=True,
        **plot_kwargs
    )

    # Generate amplitude-filtered plots (2x2 grid) if FuXi data is available
    if fuxi_ds is not None:
        lead_time_bcor_bmse_amplitude_grid(
            lead_times=lead_times,
            low_amp_correlations=low_amp_event_correlations,
            low_amp_bmsea=low_amp_event_amplitude_errors,
            low_amp_bmsep=low_amp_event_phase_errors,
            high_amp_correlations=high_amp_event_correlations,
            high_amp_bmsea=high_amp_event_amplitude_errors,
            high_amp_bmsep=high_amp_event_phase_errors,
            labels=plot_labels,
            output_filename=os.path.join(output_dir, 'lead_time_bcor_bmse_amplitude_grid.png'),
            titles=(
                titles_settings.get('low_amp_bcor'),
                titles_settings.get('low_amp_bmse'),
                titles_settings.get('high_amp_bcor'),
                titles_settings.get('high_amp_bmse')
            ),
            combined_bmse=True,
            **plot_kwargs
        )

    # Print relative improvement tables if enabled
    if config.get('print_tables', True):
        biweekly_groups = [
            (0, 13),   # Days 1-14 (indices 0-13)
            (14, 27),  # Days 15-28 (indices 14-27)
            (28, 41)   # Days 29-42 (indices 28-41)
        ]

        print_relative_improvement(correlations, "Bivariate Correlation", plot_labels, biweekly_groups, higher_is_better=True)

        if fuxi_ds is not None:
            print_relative_improvement(low_amp_event_correlations, "Bivariate Correlation (Low Amplitude Events)", plot_labels, biweekly_groups, higher_is_better=True)
            print_relative_improvement(high_amp_event_correlations, "Bivariate Correlation (High Amplitude Events)", plot_labels, biweekly_groups, higher_is_better=True)

        bmse = [np.array(amplitude_errors[i]) + np.array(phase_errors[i]) for i in range(len(amplitude_errors))]
        print_relative_improvement(bmse, "BMSE (Amplitude + Phase Error)", plot_labels, biweekly_groups, higher_is_better=False)

        if fuxi_ds is not None:
            low_amp_bmse = [np.array(low_amp_event_amplitude_errors[i]) + np.array(low_amp_event_phase_errors[i]) for i in range(len(low_amp_event_amplitude_errors))]
            high_amp_bmse = [np.array(high_amp_event_amplitude_errors[i]) + np.array(high_amp_event_phase_errors[i]) for i in range(len(high_amp_event_amplitude_errors))]
            print_relative_improvement(low_amp_bmse, "BMSE (Low Amplitude Events)", plot_labels, biweekly_groups, higher_is_better=False)
            print_relative_improvement(high_amp_bmse, "BMSE (High Amplitude Events)", plot_labels, biweekly_groups, higher_is_better=False)

if __name__ == "__main__":
    main()