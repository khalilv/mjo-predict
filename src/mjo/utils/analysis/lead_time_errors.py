import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import load_forecast
from mjo.utils.plot import (
    bivariate_correlation_vs_lead_time_plot,
    bivariate_mse_vs_lead_time_plot,
)

def compute_bmse(predict_rmm1, ground_truth_rmm1, predict_rmm2, ground_truth_rmm2):
    predict_amplitude = np.sqrt(np.square(predict_rmm1) + np.square(predict_rmm2))
    predict_phase = np.arctan2(predict_rmm2, predict_rmm1)
    ground_truth_amplitude = np.sqrt(np.square(ground_truth_rmm1) + np.square(ground_truth_rmm2))
    ground_truth_phase = np.arctan2(ground_truth_rmm2, ground_truth_rmm1)

    bmse = np.mean(np.square(predict_rmm1 - ground_truth_rmm1) + np.square(predict_rmm2 - ground_truth_rmm2))
    bmsea = np.mean(np.square(predict_amplitude - ground_truth_amplitude))
    bmsep = np.mean(2*predict_amplitude*ground_truth_amplitude*(1-np.cos(predict_phase - ground_truth_phase)))
    assert np.isclose(bmse, bmsea + bmsep), f'Found mismatch between BMSE {bmse} and components BMSEa {bmsea}, BMSEp {bmsep}'
    return [bmsea, bmsep]

def compute_bcorr(predict_rmm1, ground_truth_rmm1, predict_rmm2, ground_truth_rmm2):
    n = np.sum((predict_rmm1 * ground_truth_rmm1) + (predict_rmm2 * ground_truth_rmm2))
    d1 = np.sqrt(np.sum(np.square(predict_rmm1) + np.square(predict_rmm2)))
    d2 = np.sqrt(np.sum(np.square(ground_truth_rmm1) + np.square(ground_truth_rmm2)))
    return n / (d1*d2)

def compute_metric_across_leads(dataframes, max_lt, ground_truth_df, metric_fn, fuxi_ds=None, filter_type=None):
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


def main():

    start_date = '2019-01-01'
    end_date = '2022-01-01'
    deterministic_dirs = [
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/baselines/four_groups/mean_bias',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/baselines/four_groups/MLR_with_doy',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/bias-correction/LSTM/combined',
    ]
    plot_labels = ['Mean bias correction', 'MVLR', 'LSTM', 'FuXi mean']
    ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
    fuxi_path = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi'
    output_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/plots/production/2019-2021/bias-correction'
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = load_rmm_indices(ground_truth_path)
    fuxi_ds, fuxi_max_lt, _ = load_forecast(fuxi_path, start_date, end_date, member='mean')

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

    for predict_dir in deterministic_dirs:
        dfs, max_lt, date_strs = load_forecast(predict_dir, start_date, end_date)
        bcorr = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcorr)
        bmse = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse).T
        bcorr_low_amp_events = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcorr, fuxi_ds, 'high')
        bmse_low_amp_events = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse, fuxi_ds, 'high').T
        bcorr_high_amp_events = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcorr, fuxi_ds, 'low')
        bmse_high_amp_events = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse, fuxi_ds, 'low').T
        
        correlations.append(bcorr)
        amplitude_errors.append(bmse[0])
        phase_errors.append(bmse[1])
        max_lead_times.append(max_lt)
        low_amp_event_correlations.append(bcorr_low_amp_events)
        low_amp_event_amplitude_errors.append(bmse_low_amp_events[0])
        low_amp_event_phase_errors.append(bmse_low_amp_events[1])
        high_amp_event_correlations.append(bcorr_high_amp_events)
        high_amp_event_amplitude_errors.append(bmse_high_amp_events[0])
        high_amp_event_phase_errors.append(bmse_high_amp_events[1])
        

    bcorr = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcorr)
    bmse = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bmse).T
    bcorr_low_amp_events = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcorr, fuxi_ds, 'high')
    bmse_low_amp_events = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bmse, fuxi_ds, 'high').T
    bcorr_high_amp_events = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcorr, fuxi_ds, 'low')
    bmse_high_amp_events = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bmse, fuxi_ds, 'low').T
    correlations.append(bcorr)
    amplitude_errors.append(bmse[0])
    phase_errors.append(bmse[1])
    max_lead_times.append(fuxi_max_lt)
    low_amp_event_correlations.append(bcorr_low_amp_events)
    low_amp_event_amplitude_errors.append(bmse_low_amp_events[0])
    low_amp_event_phase_errors.append(bmse_low_amp_events[1])
    high_amp_event_correlations.append(bcorr_high_amp_events)
    high_amp_event_amplitude_errors.append(bmse_high_amp_events[0])
    high_amp_event_phase_errors.append(bmse_high_amp_events[1])
            

    bivariate_correlation_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        correlations=correlations,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bcorr.png')
    )

    bivariate_correlation_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        correlations=low_amp_event_correlations,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bcorr_forecast_low_amp_events.png')
    )

    bivariate_correlation_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        correlations=high_amp_event_correlations,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bcorr_forecast_high_amp_events.png')
    )

    bivariate_mse_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        bmsea=amplitude_errors,
        bmsep=phase_errors,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bmse.png'),
        combined=True
    )

    bivariate_mse_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        bmsea=low_amp_event_amplitude_errors,
        bmsep=low_amp_event_phase_errors,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bmse_forecast_low_amp_events.png'),
        combined=True
    )

    bivariate_mse_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        bmsea=high_amp_event_amplitude_errors,
        bmsep=high_amp_event_phase_errors,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bmse_forecast_high_amp_events.png'),
        combined=True
    )

    # Group indices for biweekly periods (0-based indexing)
    biweekly_groups = [
        (0, 13),   # Days 1-14 (indices 0-13)
        (14, 27),  # Days 15-28 (indices 14-27)
        (28, 41)   # Days 29-42 (indices 28-41)
    ]

    def print_relative_improvement(metric, metric_name, higher_is_better=True):
        print(f"\nRelative Improvement in {metric_name} (compared to '{plot_labels[3]}')")
        print("=" * 70)
        print("{:<25} {:<15} {:<15} {:<15}".format("Model", "Weeks 1-2", "Weeks 3-4", "Weeks 5-6"))
        print("-" * 70)
        for i in range(3):
            rel_improvements = []
            for start, end in biweekly_groups:
                model_mean = np.mean(metric[i][start:end+1])
                baseline_mean = np.mean(metric[3][start:end+1])
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

    print_relative_improvement(correlations, "Bivariate Correlation", higher_is_better=True)
    print_relative_improvement(low_amp_event_correlations, "Bivariate Correlation (Low Amplitude Events)", higher_is_better=True)
    print_relative_improvement(high_amp_event_correlations, "Bivariate Correlation (High Amplitude Events)", higher_is_better=True)

    bmse = [np.array(amplitude_errors[i]) + np.array(phase_errors[i]) for i in range(len(amplitude_errors))]
    low_amp_bmse = [np.array(low_amp_event_amplitude_errors[i]) + np.array(low_amp_event_phase_errors[i]) for i in range(len(low_amp_event_amplitude_errors))]
    high_amp_bmse = [np.array(high_amp_event_amplitude_errors[i]) + np.array(high_amp_event_phase_errors[i]) for i in range(len(high_amp_event_amplitude_errors))]

    print_relative_improvement(bmse, "BMSE (Amplitude + Phase Error)", higher_is_better=False)
    print_relative_improvement(low_amp_bmse, "BMSE (Low Amplitude Events)", higher_is_better=False)
    print_relative_improvement(high_amp_bmse, "BMSE (High Amplitude Events)", higher_is_better=False)

if __name__ == "__main__":
    main()