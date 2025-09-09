import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import (
    bivariate_correlation_vs_lead_time_plot,
    bivariate_mse_vs_lead_time_plot,
    bivariate_mse_vs_init_date_plot, 
    bivariate_correlation_by_month_plot,
    bivariate_correlation_vs_phase_plot,
    bivariate_mse_vs_phase_plot
)
from matplotlib import pyplot as plt

def load_forecast(predict_dir, start_date, end_date, member=None):
    dataframes = []
    max_lt = -1
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = sorted([d for d in os.listdir(predict_dir) if start <= datetime.strptime(d, "%Y-%m-%d" if member else "%Y-%m-%d.txt") < end])
    for d in tqdm(dates, f'Loading data from {predict_dir}'):
        filepath = os.path.join(predict_dir, d, f'{member}.txt') if member else os.path.join(predict_dir, d)
        df = load_rmm_indices(filepath)
        max_lt = max(max_lt, len(df))
        dataframes.append(df)
    return dataframes, max_lt, dates

def compute_amplitude_error_across_leads(dataframes, max_lt, ground_truth_df, filter_low_amplitude_samples=False):
    results = []
    for lt in tqdm(range(max_lt), 'Computing per-lead metric'):
        preds, truths = [], []
        for df in dataframes:
            if filter_low_amplitude_samples:
                init_date = df.index[0] - timedelta(days=1)
                init_amplitude = ground_truth_df.loc[init_date].amplitude
                if init_amplitude < 1: 
                    continue
            if lt < len(df):
                preds.append(df.iloc[lt])
                truths.append(ground_truth_df.loc[df.index[lt]])
        pred_df = pd.DataFrame(preds)
        truth_df = pd.DataFrame(truths)
        result = (pred_df.amplitude.values - truth_df.amplitude.values).mean()
        results.append(result)
    return np.array(results)

ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"    
ground_truth_ds = load_rmm_indices(ground_truth_path)
forecast_dirs = [
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/0d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/1d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/10d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/90d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/180d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/360d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/720d_hist/logs/version_0/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/0d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/1d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/10d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/90d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/180d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/360d_hist/logs/version_0/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/720d_hist/logs/version_0/outputs',
    ]
amplitude_errs = []

for forecast_dir in forecast_dirs:
    forecast_ds, max_lt, _ = load_forecast(forecast_dir, start_date='2019-01-01', end_date='2022-01-01')
    amplitude_err = compute_amplitude_error_across_leads(forecast_ds, max_lt, ground_truth_ds)
    amplitude_errs.append(amplitude_err)

amplitude_errs = np.array(amplitude_errs).reshape(2, 6, 42)
amplitude_err_means = amplitude_errs.mean(axis=1)
amplitude_err_stds = amplitude_errs.std(axis=1)

lead_times = np.arange(1, 43)  # lead times 1–42

colors = ["blue", "green"]

plt.figure(figsize=(8, 4))

for i, (mean, std, label, color) in enumerate(zip(amplitude_err_means, amplitude_err_stds, ['TFT', 'TSMixer'], colors)):
    plt.plot(lead_times, mean, color=color, label=label)
    plt.fill_between(
        lead_times,
        mean - std,
        mean + std,
        color=color,
        alpha=0.2
    )

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.ylim(-1, 0.5)
plt.xlim(1, 42)
plt.xticks(np.arange(1, 43, 3))
plt.xlabel("Lead Time (days)")
plt.ylabel("Amplitude Bias")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("/glade/derecho/scratch/kvirji/mjo-predict/plots/production/2019-2021/history_only/amplitude_bias_vs_lead_time.png", dpi=300)
plt.close()




