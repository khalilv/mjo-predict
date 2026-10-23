import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from typing import Callable, List, Tuple, Optional
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import (
    load_forecast, load_config, get_bootstrap_settings, get_line_plot_kwargs,
    collect_lead_pairs, compute_bcor, compute_bmse, compute_bmsea, compute_bmsep,
    block_bootstrap_paired_improvement,
    block_bootstrap_per_lead_significance, block_bootstrap_skillful_window,
)
from mjo.utils.plot import (
    lead_time_bcor_bmse_plot,
    lead_time_bmse_decomp_plot,
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
    """Compute a metric (bcor or bmse) across all lead times; filter_type ('high'/'low') names the FuXi-amplitude regime to EXCLUDE."""
    results = []
    for lt in tqdm(range(max_lt), 'Computing per-lead metric'):
        _, preds, truths = collect_lead_pairs(dataframes, lt, ground_truth_df, fuxi_ds, filter_type, check_dates=True)
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
    higher_is_better: bool = True,
    paired_cis: Optional[List[List[Tuple[float, float]]]] = None,
) -> None:
    """Print relative improvement vs the last (baseline) model per biweekly group, optionally with paired bootstrap CIs."""
    has_ci = paired_cis is not None
    print(f"\nRelative Improvement in {metric_name} (compared to '{plot_labels[-1]}')")
    if has_ci:
        print("=" * 100)
        print("{:<25} {:<25} {:<25} {:<25}".format("Model", "Weeks 1-2", "Weeks 3-4", "Weeks 5-6"))
        print("-" * 100)
    else:
        print("=" * 70)
        print("{:<25} {:<15} {:<15} {:<15}".format("Model", "Weeks 1-2", "Weeks 3-4", "Weeks 5-6"))
        print("-" * 70)
    baseline_idx = len(plot_labels) - 1
    for i in range(len(plot_labels) - 1):
        rel_improvements = []
        ci_strings = []
        for g, (start, end) in enumerate(biweekly_groups):
            model_mean = np.mean(metric[i][start:end+1])
            baseline_mean = np.mean(metric[baseline_idx][start:end+1])
            if higher_is_better:
                rel_imp = (model_mean - baseline_mean) / abs(baseline_mean) * 100
            else:
                rel_imp = (baseline_mean - model_mean) / abs(baseline_mean) * 100
            rel_improvements.append(rel_imp)

            if has_ci:
                ci_lo, ci_hi = paired_cis[i][g]
                ci_strings.append(f"{rel_imp:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}]")

        if has_ci:
            print("{:<25} {:<25} {:<25} {:<25}".format(
                plot_labels[i], ci_strings[0], ci_strings[1], ci_strings[2]
            ))
        else:
            print("{:<25} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                plot_labels[i], rel_improvements[0], rel_improvements[1], rel_improvements[2]
            ))
    if has_ci:
        print("=" * 100)
    else:
        print("=" * 70)
    print(f"Values are percent relative improvement in {metric_name}.\n")


def main():
    """Compute and plot BCOR/BMSE versus lead time, with optional bootstrap significance."""
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
    titles_settings = plot_settings.get('titles', {})
    plot_kwargs = get_line_plot_kwargs(plot_settings)
    if plot_settings.get('y_labels', {}).get('hide_bcor', False):
        plot_kwargs['y_label'] = ''

    bootstrap_enabled, bootstrap_kwargs = get_bootstrap_settings(config)

    # Load ground truth
    ground_truth = load_rmm_indices(ground_truth_path)

    # Initialize result containers
    correlations = []
    phase_errors = []
    amplitude_errors = []
    all_model_dfs = []
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

    def _add_model(dfs, max_lt):
        """Compute and store all-forecast and amplitude-filtered metrics for one model."""
        bcor = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcor)
        bmse = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse).T
        correlations.append(bcor)
        amplitude_errors.append(bmse[0])
        phase_errors.append(bmse[1])
        max_lead_times.append(max_lt)
        all_model_dfs.append(dfs)

        # Amplitude-filtered metrics; filter_type names the FuXi-amplitude regime to exclude
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

    for predict_dir in forecast_dirs:
        dfs, max_lt, _ = load_forecast(predict_dir, start_date, end_date)
        _add_model(dfs, max_lt)

    # FuXi is appended last and serves as the baseline
    if fuxi_ds is not None:
        _add_model(fuxi_ds, fuxi_max_lt)

    # Prepare lead times
    lead_times = [np.arange(1, max_lt + 1) for max_lt in max_lead_times]

    # Generate skill plot (BCOR and BMSE side by side) — without CIs
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

    # Generate skill plot with significance coloring
    if bootstrap_enabled and all_model_dfs:
        # Compute per-lead-time significance for BCOR and BMSE
        bcor_significance = block_bootstrap_per_lead_significance(
            all_model_dfs, max_lead_times, ground_truth, compute_bcor,
            higher_is_better=True, **bootstrap_kwargs
        )
        bmse_significance = block_bootstrap_per_lead_significance(
            all_model_dfs, max_lead_times, ground_truth, compute_bmse,
            higher_is_better=False, **bootstrap_kwargs
        )

        lead_time_bcor_bmse_plot(
            lead_times=lead_times,
            correlations=correlations,
            bmsea=amplitude_errors,
            bmsep=phase_errors,
            labels=plot_labels,
            output_filename=os.path.join(output_dir, 'lead_time_bcor_bmse_ci.png'),
            titles=(titles_settings.get('bcor'), titles_settings.get('bmse')),
            combined_bmse=True,
            bcor_significance=bcor_significance,
            bmse_significance=bmse_significance,
            **plot_kwargs
        )

    # Generate decomposed BMSE figure (amplitude-error vs phase-error components)
    decomp_titles = (
        titles_settings.get('bmsea', '(a) BMSE - Amplitude Error Component'),
        titles_settings.get('bmsep', '(b) BMSE - Phase Error Component'),
    )

    lead_time_bmse_decomp_plot(
        lead_times=lead_times,
        bmsea=amplitude_errors,
        bmsep=phase_errors,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'lead_time_bmse_decomposed.png'),
        titles=decomp_titles,
        **plot_kwargs
    )

    if bootstrap_enabled and all_model_dfs:
        # Significance of each error component vs FuXi baseline (lower error is better)
        bmsea_significance = block_bootstrap_per_lead_significance(
            all_model_dfs, max_lead_times, ground_truth, compute_bmsea,
            higher_is_better=False, **bootstrap_kwargs
        )
        bmsep_significance = block_bootstrap_per_lead_significance(
            all_model_dfs, max_lead_times, ground_truth, compute_bmsep,
            higher_is_better=False, **bootstrap_kwargs
        )
        lead_time_bmse_decomp_plot(
            lead_times=lead_times,
            bmsea=amplitude_errors,
            bmsep=phase_errors,
            labels=plot_labels,
            output_filename=os.path.join(output_dir, 'lead_time_bmse_decomposed_ci.png'),
            titles=decomp_titles,
            amplitude_significance=bmsea_significance,
            phase_significance=bmsep_significance,
            **plot_kwargs
        )

    # Generate amplitude-filtered plots (2x2 grid) if FuXi data is available
    if fuxi_ds is not None:
        amp_grid_titles = (
            titles_settings.get('low_amp_bcor'),
            titles_settings.get('low_amp_bmse'),
            titles_settings.get('high_amp_bcor'),
            titles_settings.get('high_amp_bmse')
        )

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
            titles=amp_grid_titles,
            combined_bmse=True,
            **plot_kwargs
        )

        # Amplitude grid with significance coloring
        if bootstrap_enabled and all_model_dfs:
            # Low-amplitude regime uses filter_type='high' (exclude high-amplitude FuXi forecasts);
            # high-amplitude regime uses filter_type='low' (exclude low-amplitude FuXi forecasts).
            low_amp_bcor_sig = block_bootstrap_per_lead_significance(
                all_model_dfs, max_lead_times, ground_truth, compute_bcor,
                higher_is_better=True, fuxi_ds=fuxi_ds, filter_type='high',
                **bootstrap_kwargs
            )
            low_amp_bmse_sig = block_bootstrap_per_lead_significance(
                all_model_dfs, max_lead_times, ground_truth, compute_bmse,
                higher_is_better=False, fuxi_ds=fuxi_ds, filter_type='high',
                **bootstrap_kwargs
            )
            high_amp_bcor_sig = block_bootstrap_per_lead_significance(
                all_model_dfs, max_lead_times, ground_truth, compute_bcor,
                higher_is_better=True, fuxi_ds=fuxi_ds, filter_type='low',
                **bootstrap_kwargs
            )
            high_amp_bmse_sig = block_bootstrap_per_lead_significance(
                all_model_dfs, max_lead_times, ground_truth, compute_bmse,
                higher_is_better=False, fuxi_ds=fuxi_ds, filter_type='low',
                **bootstrap_kwargs
            )
            lead_time_bcor_bmse_amplitude_grid(
                lead_times=lead_times,
                low_amp_correlations=low_amp_event_correlations,
                low_amp_bmsea=low_amp_event_amplitude_errors,
                low_amp_bmsep=low_amp_event_phase_errors,
                high_amp_correlations=high_amp_event_correlations,
                high_amp_bmsea=high_amp_event_amplitude_errors,
                high_amp_bmsep=high_amp_event_phase_errors,
                labels=plot_labels,
                output_filename=os.path.join(output_dir, 'lead_time_bcor_bmse_amplitude_grid_ci.png'),
                titles=amp_grid_titles,
                combined_bmse=True,
                low_amp_bcor_significance=low_amp_bcor_sig,
                low_amp_bmse_significance=low_amp_bmse_sig,
                high_amp_bcor_significance=high_amp_bcor_sig,
                high_amp_bmse_significance=high_amp_bmse_sig,
                **plot_kwargs
            )

    # Print relative improvement tables if enabled
    if config.get('print_tables', True):
        biweekly_groups = [
            (0, 13),   # Days 1-14 (indices 0-13)
            (14, 27),  # Days 15-28 (indices 14-27)
            (28, 41)   # Days 29-42 (indices 28-41)
        ]

        # Compute paired bootstrap CIs for relative improvement
        bcor_paired_cis = None
        bmse_paired_cis = None
        if bootstrap_enabled and all_model_dfs:
            bcor_paired_cis = block_bootstrap_paired_improvement(
                all_model_dfs, max_lead_times, ground_truth, compute_bcor,
                biweekly_groups, higher_is_better=True, **bootstrap_kwargs
            )
            bmse_paired_cis = block_bootstrap_paired_improvement(
                all_model_dfs, max_lead_times, ground_truth, compute_bmse,
                biweekly_groups, higher_is_better=False, **bootstrap_kwargs
            )

        print_relative_improvement(correlations, "Bivariate Correlation", plot_labels, biweekly_groups,
                                  higher_is_better=True, paired_cis=bcor_paired_cis)

        # Print skillful window CIs
        if bootstrap_enabled and all_model_dfs:
            skillful_windows = block_bootstrap_skillful_window(
                all_model_dfs, max_lead_times, ground_truth,
                threshold=0.5, **bootstrap_kwargs
            )
            print("\nSkillful Window (BCOR > 0.5, 95% CI)")
            print("=" * 60)
            print("{:<25} {:<35}".format("Model", "Skillful Window (days)"))
            print("-" * 60)
            for i, label in enumerate(plot_labels):
                point, lo, hi = skillful_windows[i]
                print("{:<25} {:.0f} days (95% CI: {:.0f}–{:.0f})".format(label, point, lo, hi))
            print("=" * 60)

        if fuxi_ds is not None:
            # Compute amplitude-filtered paired CIs if bootstrap enabled
            low_bcor_cis, high_bcor_cis = None, None
            low_bmse_cis, high_bmse_cis = None, None
            if bootstrap_enabled and all_model_dfs:
                low_bcor_cis = block_bootstrap_paired_improvement(
                    all_model_dfs, max_lead_times, ground_truth, compute_bcor,
                    biweekly_groups, higher_is_better=True,
                    fuxi_ds=fuxi_ds, filter_type='high', **bootstrap_kwargs
                )
                high_bcor_cis = block_bootstrap_paired_improvement(
                    all_model_dfs, max_lead_times, ground_truth, compute_bcor,
                    biweekly_groups, higher_is_better=True,
                    fuxi_ds=fuxi_ds, filter_type='low', **bootstrap_kwargs
                )
                low_bmse_cis = block_bootstrap_paired_improvement(
                    all_model_dfs, max_lead_times, ground_truth, compute_bmse,
                    biweekly_groups, higher_is_better=False,
                    fuxi_ds=fuxi_ds, filter_type='high', **bootstrap_kwargs
                )
                high_bmse_cis = block_bootstrap_paired_improvement(
                    all_model_dfs, max_lead_times, ground_truth, compute_bmse,
                    biweekly_groups, higher_is_better=False,
                    fuxi_ds=fuxi_ds, filter_type='low', **bootstrap_kwargs
                )

            print_relative_improvement(low_amp_event_correlations, "Bivariate Correlation (Low Amplitude Events)",
                                      plot_labels, biweekly_groups, higher_is_better=True, paired_cis=low_bcor_cis)
            print_relative_improvement(high_amp_event_correlations, "Bivariate Correlation (High Amplitude Events)",
                                      plot_labels, biweekly_groups, higher_is_better=True, paired_cis=high_bcor_cis)

        bmse = [np.array(amplitude_errors[i]) + np.array(phase_errors[i]) for i in range(len(amplitude_errors))]
        print_relative_improvement(bmse, "BMSE (Amplitude + Phase Error)", plot_labels, biweekly_groups,
                                  higher_is_better=False, paired_cis=bmse_paired_cis)

        if fuxi_ds is not None:
            low_amp_bmse = [np.array(low_amp_event_amplitude_errors[i]) + np.array(low_amp_event_phase_errors[i]) for i in range(len(low_amp_event_amplitude_errors))]
            high_amp_bmse = [np.array(high_amp_event_amplitude_errors[i]) + np.array(high_amp_event_phase_errors[i]) for i in range(len(high_amp_event_amplitude_errors))]
            print_relative_improvement(low_amp_bmse, "BMSE (Low Amplitude Events)", plot_labels, biweekly_groups,
                                      higher_is_better=False, paired_cis=low_bmse_cis)
            print_relative_improvement(high_amp_bmse, "BMSE (High Amplitude Events)", plot_labels, biweekly_groups,
                                      higher_is_better=False, paired_cis=high_bmse_cis)

if __name__ == "__main__":
    main()