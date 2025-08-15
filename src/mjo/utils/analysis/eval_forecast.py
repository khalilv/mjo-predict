import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import (
    bivariate_correlation_vs_lead_time_plot,
    bivariate_mse_vs_lead_time_plot,
    bivariate_mse_vs_init_date_plot, 
    bivariate_correlation_by_month_plot
)

def compute_bcorr(predict_rmm1, ground_truth_rmm1, predict_rmm2, ground_truth_rmm2):
    n = np.sum((predict_rmm1 * ground_truth_rmm1) + (predict_rmm2 * ground_truth_rmm2))
    d1 = np.sqrt(np.sum(np.square(predict_rmm1) + np.square(predict_rmm2)))
    d2 = np.sqrt(np.sum(np.square(ground_truth_rmm1) + np.square(ground_truth_rmm2)))
    return n / (d1*d2)

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

def load_forecast(predict_dir, start_date, member=None):
    dataframes = []
    max_lt = -1
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = sorted([d for d in os.listdir(predict_dir) if datetime.strptime(d, "%Y-%m-%d" if member else "%Y-%m-%d.txt") > start])
    for d in tqdm(dates, f'Loading data from {predict_dir}'):
        filepath = os.path.join(predict_dir, d, f'{member}.txt') if member else os.path.join(predict_dir, d)
        df = load_rmm_indices(filepath)
        max_lt = max(max_lt, len(df))
        dataframes.append(df)
    return dataframes, max_lt, dates


def compute_metric_across_leads(dataframes, max_lt, ground_truth_df, metric_fn):
    results = []
    for lt in tqdm(range(max_lt), 'Computing per-lead metric'):
        preds, truths = [], []
        for df in dataframes:
            if lt < len(df):
                preds.append(df.iloc[lt])
                truths.append(ground_truth_df.loc[df.index[lt]])
        pred_df = pd.DataFrame(preds)
        truth_df = pd.DataFrame(truths)
        result = metric_fn(pred_df.RMM1.values, truth_df.RMM1.values, pred_df.RMM2.values, truth_df.RMM2.values)
        results.append(result)
    return np.array(results)

def compute_metric_across_dfs(dataframes, max_lt, ground_truth_df, metric_fn):
    results = []
    for df in tqdm(dataframes, 'Computing per-dataframe metric'):
        truths = []
        for lt in range(max_lt):
            if lt < len(df):
                truths.append(ground_truth_df.loc[df.index[lt]])
        truth_df = pd.DataFrame(truths)
        result = metric_fn(df.RMM1.values, truth_df.RMM1.values, df.RMM2.values, truth_df.RMM2.values)
        results.append(result)
    return np.array(results)

def compute_metric_across_months(dataframes, max_lt, ground_truth_df, metric_fn):
    monthly_groups = defaultdict(list)
    monthly_results = defaultdict()

    for df in dataframes:
        monthly_groups[df.index[0].month].append(df)
    
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

    start_date = '2020-01-01'
    deterministic_dirs = [
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/TSMixer/FuXi/ensemble_mean/no_hist/logs/version_0/outputs',
    ]
    deterministic_labels = ['TSMixer no hist']
    ensemble_dirs = ['/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi']
    ensemble_labels = ['FuXi']
    ensemble_members = ['mean']
    ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
    output_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/plots/2020-2021'
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = load_rmm_indices(ground_truth_path)

    correlations = []
    correlations_per_month = []
    phase_errors = []
    amplitude_errors = []
    max_lead_times = []

    bmse_per_init_dates = []
    init_dates = []

    for predict_dir in deterministic_dirs:
        dfs, max_lt, date_strs = load_forecast(predict_dir, start_date)
        bcorr = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcorr)
        bmse = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse).T
        bmse_per_init_date = compute_metric_across_dfs(dfs, max_lt, ground_truth, compute_bmse).T
        bcorr_per_month = compute_metric_across_months(dfs, max_lt, ground_truth, compute_bcorr)

        correlations.append(bcorr)
        correlations_per_month.append(bcorr_per_month)
        amplitude_errors.append(bmse[0])
        phase_errors.append(bmse[1])
        max_lead_times.append(max_lt)
        init_dates.append([datetime.strptime(d, "%Y-%m-%d.txt") for d in date_strs])
        bmse_per_init_dates.append(bmse_per_init_date[0] + bmse_per_init_date[1])

    ensemble_member_labels = []
    for label_idx, predict_dir in enumerate(ensemble_dirs):
        for member in ensemble_members:
            dfs, max_lt, date_strs = load_forecast(predict_dir, start_date, member)
            bcorr = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcorr)
            bmse = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse).T
            bmse_per_init_date = compute_metric_across_dfs(dfs, max_lt, ground_truth, compute_bmse).T
            bcorr_per_month = compute_metric_across_months(dfs, max_lt, ground_truth, compute_bcorr)

            correlations.append(bcorr)
            correlations_per_month.append(bcorr_per_month)
            amplitude_errors.append(bmse[0])
            phase_errors.append(bmse[1])
            max_lead_times.append(max_lt)
            ensemble_member_labels.append(f'{ensemble_labels[label_idx]}-{member}')
            init_dates.append([datetime.strptime(d, "%Y-%m-%d") for d in date_strs])
            bmse_per_init_dates.append(bmse_per_init_date[0] + bmse_per_init_date[1])

    plot_labels = deterministic_labels + ensemble_member_labels

    bivariate_correlation_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        correlations=correlations,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bcorr.png')
    )

    bivariate_mse_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        bmsea=amplitude_errors,
        bmsep=phase_errors,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bmse.png')
    )

    bivariate_mse_vs_init_date_plot(
        init_dates=init_dates,
        bmse=bmse_per_init_dates,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bmse_init_date.png')
    )

    bmse_diff_from_last = [b - bmse_per_init_dates[-1][:len(b)] for b in bmse_per_init_dates]
    bivariate_mse_vs_init_date_plot(
        init_dates=init_dates,
        bmse=bmse_diff_from_last,
        labels=plot_labels,
        output_filename=os.path.join(output_dir, 'bmse_init_date_relative.png')
    )

    for l, bcorr_per_month in enumerate(correlations_per_month):
        bivariate_correlation_by_month_plot(
            bcorr_dict=bcorr_per_month, 
            label='Bivariate Correlation',
            title='Bivariate correlation per month',
            output_filename=os.path.join(output_dir, f'bcorr_per_month_{plot_labels[l]}.png')
        )

if __name__ == "__main__":
    main()