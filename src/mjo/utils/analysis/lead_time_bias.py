import os
import pandas as pd
import numpy as np
import argparse
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import load_forecast, load_config
from tqdm import tqdm
from typing import List, Optional
from mjo.utils.plot import lead_time_bias_plot, lead_time_bias_amplitude_grid


def compute_bias_across_leads(
    dataframes: List[pd.DataFrame],
    bias_type: str,
    max_lt: int,
    ground_truth_df: pd.DataFrame,
    fuxi_ds: Optional[List[pd.DataFrame]] = None,
    filter_type: Optional[str] = None
) -> np.ndarray:
    """Compute bias (amplitude or phase) across all lead times for forecasts."""
    results = []
    assert bias_type in ['amplitude', 'phase'], 'Bias type must be either amplitude or phase'

    for lt in tqdm(range(max_lt), 'Computing per-lead bias'):
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

        if bias_type == 'amplitude':
            result = (pred_df.amplitude.values - truth_df.amplitude.values).mean()
        else:
            num = (pred_df.RMM1.values * truth_df.RMM2.values - pred_df.RMM2.values * truth_df.RMM1.values)
            den = (pred_df.RMM1.values * truth_df.RMM1.values + pred_df.RMM2.values * truth_df.RMM2.values)
            result = np.arctan2(num, den).mean()

        results.append(result)

    return np.array(results)


def main():
    """Main function to compute and plot lead time bias."""
    # Parse command line arguments
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
    font_settings = plot_settings.get('fonts', {})
    titles_settings = plot_settings.get('titles', {})
    colormap = plot_settings.get('colormap', {})
    ylim_settings = plot_settings.get('ylim', {})

    # Build plot kwargs
    plot_kwargs = {
        'fontsize_title': font_settings.get('title', 14),
        'fontsize_axis_label': font_settings.get('axis_label', 12),
        'fontsize_tick_label': font_settings.get('tick_label', 11),
        'fontsize_legend': font_settings.get('legend', 11),
        'fontweight_title': font_settings.get('title_weight', 'bold'),
        'fontweight_title_label': font_settings.get('title_label_weight'),
        'font_family': font_settings.get('family'),
        'colormap': colormap.get('name', 'viridis'),
        'ylim_amplitude': ylim_settings.get('amplitude'),
        'ylim_phase': ylim_settings.get('phase'),
    }

    # Extract y-axis label settings
    y_labels = plot_settings.get('y_labels', {})
    if y_labels.get('hide_amplitude', False):
        plot_kwargs['y_label_amplitude'] = ''
    if y_labels.get('hide_phase', False):
        plot_kwargs['y_label_phase'] = ''

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

    # Load FuXi data if configured
    fuxi_ds = None
    fuxi_max_lt = None
    if fuxi_path:
        fuxi_ds, fuxi_max_lt, _ = load_forecast(fuxi_path, start_date, end_date, member='mean')

    # Process each forecast directory
    for forecast_dir in forecast_dirs:
        dfs, max_lt, _ = load_forecast(forecast_dir, start_date=start_date, end_date=end_date)

        # Compute biases
        amplitude_bias = compute_bias_across_leads(dfs, 'amplitude', max_lt, ground_truth)
        phase_bias = compute_bias_across_leads(dfs, 'phase', max_lt, ground_truth)

        amplitude_biases.append(amplitude_bias)
        phase_biases.append(phase_bias)
        max_lead_times.append(max_lt)

        # Compute amplitude-filtered biases if FuXi data is available
        if fuxi_ds is not None:
            amplitude_bias_low_amp = compute_bias_across_leads(dfs, 'amplitude', max_lt, ground_truth, fuxi_ds, 'high')
            phase_bias_low_amp = compute_bias_across_leads(dfs, 'phase', max_lt, ground_truth, fuxi_ds, 'high')
            amplitude_bias_high_amp = compute_bias_across_leads(dfs, 'amplitude', max_lt, ground_truth, fuxi_ds, 'low')
            phase_bias_high_amp = compute_bias_across_leads(dfs, 'phase', max_lt, ground_truth, fuxi_ds, 'low')

            low_amp_amplitude_biases.append(amplitude_bias_low_amp)
            low_amp_phase_biases.append(phase_bias_low_amp)
            high_amp_amplitude_biases.append(amplitude_bias_high_amp)
            high_amp_phase_biases.append(phase_bias_high_amp)

    # Process FuXi forecasts if available
    if fuxi_ds is not None and fuxi_max_lt is not None:
        amplitude_bias = compute_bias_across_leads(fuxi_ds, 'amplitude', fuxi_max_lt, ground_truth)
        phase_bias = compute_bias_across_leads(fuxi_ds, 'phase', fuxi_max_lt, ground_truth)
        amplitude_bias_low_amp = compute_bias_across_leads(fuxi_ds, 'amplitude', fuxi_max_lt, ground_truth, fuxi_ds, 'high')
        phase_bias_low_amp = compute_bias_across_leads(fuxi_ds, 'phase', fuxi_max_lt, ground_truth, fuxi_ds, 'high')
        amplitude_bias_high_amp = compute_bias_across_leads(fuxi_ds, 'amplitude', fuxi_max_lt, ground_truth, fuxi_ds, 'low')
        phase_bias_high_amp = compute_bias_across_leads(fuxi_ds, 'phase', fuxi_max_lt, ground_truth, fuxi_ds, 'low')

        amplitude_biases.append(amplitude_bias)
        phase_biases.append(phase_bias)
        max_lead_times.append(fuxi_max_lt)
        low_amp_amplitude_biases.append(amplitude_bias_low_amp)
        low_amp_phase_biases.append(phase_bias_low_amp)
        high_amp_amplitude_biases.append(amplitude_bias_high_amp)
        high_amp_phase_biases.append(phase_bias_high_amp)

    # Prepare lead times
    lead_times = [np.arange(1, max_lt + 1) for max_lt in max_lead_times]

    # Generate bias plot (1x2)
    lead_time_bias_plot(
        lead_times=lead_times,
        amplitude_bias=amplitude_biases,
        phase_bias=phase_biases,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'lead_time_bias.png'),
        titles=(titles_settings.get('amplitude'), titles_settings.get('phase')),
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
