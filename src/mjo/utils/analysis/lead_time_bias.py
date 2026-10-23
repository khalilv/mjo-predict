import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from typing import List, Optional
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import (
    load_forecast, load_config, get_bootstrap_settings, get_line_plot_kwargs,
    collect_lead_pairs, compute_bias_from_frames,
    compute_amplitude_bias, compute_phase_bias,
    block_bootstrap_per_lead_significance,
)
from mjo.utils.plot import lead_time_bias_plot, lead_time_bias_amplitude_grid


def compute_bias_across_leads(
    dataframes: List[pd.DataFrame],
    bias_type: str,
    max_lt: int,
    ground_truth_df: pd.DataFrame,
    fuxi_ds: Optional[List[pd.DataFrame]] = None,
    filter_type: Optional[str] = None
) -> np.ndarray:
    """Compute amplitude or phase bias across all lead times; filter_type ('high'/'low') names the FuXi-amplitude regime to EXCLUDE."""
    results = []
    assert bias_type in ['amplitude', 'phase'], 'Bias type must be either amplitude or phase'

    for lt in tqdm(range(max_lt), 'Computing per-lead bias'):
        _, preds, truths = collect_lead_pairs(dataframes, lt, ground_truth_df, fuxi_ds, filter_type, check_dates=True)
        results.append(compute_bias_from_frames(pd.DataFrame(preds), pd.DataFrame(truths), bias_type))

    return np.array(results)


def main():
    """Compute and plot amplitude and phase bias versus lead time, with optional bootstrap significance."""
    parser = argparse.ArgumentParser(description='Generate lead time bias plots')
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
    titles_settings = plot_settings.get('titles', {})
    ylim_settings = plot_settings.get('ylim', {})
    plot_kwargs = get_line_plot_kwargs(plot_settings)
    plot_kwargs['ylim_amplitude'] = ylim_settings.get('amplitude')
    plot_kwargs['ylim_phase'] = ylim_settings.get('phase')
    y_labels = plot_settings.get('y_labels', {})
    if y_labels.get('hide_amplitude', False):
        plot_kwargs['y_label_amplitude'] = ''
    if y_labels.get('hide_phase', False):
        plot_kwargs['y_label_phase'] = ''

    bootstrap_enabled, bootstrap_kwargs = get_bootstrap_settings(config)

    # Load ground truth
    ground_truth = load_rmm_indices(ground_truth_path)

    # Initialize result containers
    amplitude_biases = []
    phase_biases = []
    low_amp_amplitude_biases = []
    low_amp_phase_biases = []
    high_amp_amplitude_biases = []
    high_amp_phase_biases = []
    max_lead_times = []
    all_model_dfs = []

    # Load FuXi data if configured
    fuxi_ds = None
    fuxi_max_lt = None
    if fuxi_path:
        fuxi_ds, fuxi_max_lt, _ = load_forecast(fuxi_path, start_date, end_date, member='mean')

    def _add_model(dfs, max_lt):
        """Compute and store all-forecast and amplitude-filtered biases for one model."""
        amplitude_biases.append(compute_bias_across_leads(dfs, 'amplitude', max_lt, ground_truth))
        phase_biases.append(compute_bias_across_leads(dfs, 'phase', max_lt, ground_truth))
        max_lead_times.append(max_lt)
        all_model_dfs.append(dfs)

        # Amplitude-filtered biases; filter_type names the FuXi-amplitude regime to exclude
        if fuxi_ds is not None:
            low_amp_amplitude_biases.append(compute_bias_across_leads(dfs, 'amplitude', max_lt, ground_truth, fuxi_ds, 'high'))
            low_amp_phase_biases.append(compute_bias_across_leads(dfs, 'phase', max_lt, ground_truth, fuxi_ds, 'high'))
            high_amp_amplitude_biases.append(compute_bias_across_leads(dfs, 'amplitude', max_lt, ground_truth, fuxi_ds, 'low'))
            high_amp_phase_biases.append(compute_bias_across_leads(dfs, 'phase', max_lt, ground_truth, fuxi_ds, 'low'))

    for forecast_dir in forecast_dirs:
        dfs, max_lt, _ = load_forecast(forecast_dir, start_date=start_date, end_date=end_date)
        _add_model(dfs, max_lt)

    # FuXi is appended last and serves as the baseline
    if fuxi_ds is not None:
        _add_model(fuxi_ds, fuxi_max_lt)

    # Prepare lead times
    lead_times = [np.arange(1, max_lt + 1) for max_lt in max_lead_times]

    # Generate bias plot (1x2) — without CIs
    lead_time_bias_plot(
        lead_times=lead_times,
        amplitude_bias=amplitude_biases,
        phase_bias=phase_biases,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'lead_time_bias.png'),
        titles=(titles_settings.get('amplitude'), titles_settings.get('phase')),
        **plot_kwargs
    )

    # Generate bias plot with significance coloring (no CI bands)
    if bootstrap_enabled and all_model_dfs:
        # higher_is_better is irrelevant here: the significance test is two-sided.
        amplitude_significance = block_bootstrap_per_lead_significance(
            all_model_dfs, max_lead_times, ground_truth, compute_amplitude_bias,
            higher_is_better=True, **bootstrap_kwargs
        )
        phase_significance = block_bootstrap_per_lead_significance(
            all_model_dfs, max_lead_times, ground_truth, compute_phase_bias,
            higher_is_better=True, **bootstrap_kwargs
        )
        lead_time_bias_plot(
            lead_times=lead_times,
            amplitude_bias=amplitude_biases,
            phase_bias=phase_biases,
            labels=plot_labels,
            output_filename=os.path.join(output_dir, 'lead_time_bias_ci.png'),
            titles=(titles_settings.get('amplitude'), titles_settings.get('phase')),
            amplitude_significance=amplitude_significance,
            phase_significance=phase_significance,
            **plot_kwargs
        )

    # Generate amplitude-filtered plots (2x2 grid) if FuXi data is available
    if fuxi_ds is not None:
        lead_time_bias_amplitude_grid(
            lead_times=lead_times,
            low_amp_amplitude_bias=low_amp_amplitude_biases,
            low_amp_phase_bias=low_amp_phase_biases,
            high_amp_amplitude_bias=high_amp_amplitude_biases,
            high_amp_phase_bias=high_amp_phase_biases,
            labels=plot_labels,
            output_filename=os.path.join(output_dir, 'lead_time_bias_amplitude_grid.png'),
            titles=(
                titles_settings.get('low_amp_amplitude'),
                titles_settings.get('low_amp_phase'),
                titles_settings.get('high_amp_amplitude'),
                titles_settings.get('high_amp_phase')
            ),
            **plot_kwargs
        )


if __name__ == "__main__":
    main()
