import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import (
    bivariate_correlation_vs_lead_time_plot,
    bivariate_mse_vs_lead_time_plot
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
    return dataframes, max_lt


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
    phase_errors = []
    amplitude_errors = []
    max_lead_times = []
    for predict_dir in deterministic_dirs:
        dfs, max_lt = load_forecast(predict_dir, start_date)
        bcorr = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcorr)
        bmse = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse)
        correlations.append(bcorr)
        amplitude_errors.append(bmse.T[0])
        phase_errors.append(bmse.T[1])
        max_lead_times.append(max_lt)
    
    ensemble_member_labels = []
    for label_idx, predict_dir in enumerate(ensemble_dirs):
        for member in ensemble_members:
            dfs, max_lt = load_forecast(predict_dir, start_date, member)
            bcorr = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcorr)
            bmse = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bmse)
            correlations.append(bcorr)
            amplitude_errors.append(bmse.T[0])
            phase_errors.append(bmse.T[1])
            max_lead_times.append(max_lt)
            ensemble_member_labels.append(f'{ensemble_labels[label_idx]} {member}')

    
    bivariate_correlation_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        correlations=correlations,
        labels=deterministic_labels + ensemble_member_labels,
        output_filename=os.path.join(output_dir, 'bcorr.png')
    )

    bivariate_mse_vs_lead_time_plot(
        lead_times=[np.arange(1, max_lead_time + 1) for max_lead_time in max_lead_times],
        bmsea=amplitude_errors,
        bmsep=phase_errors,
        labels=deterministic_labels + ensemble_member_labels,
        output_filename=os.path.join(output_dir, 'bmse.png')
    )



    



if __name__ == "__main__":
    main()