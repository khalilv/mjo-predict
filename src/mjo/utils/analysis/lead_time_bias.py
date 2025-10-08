import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from mjo.utils.analysis.utils import load_forecast
from mjo.utils.RMM.io import load_rmm_indices
from matplotlib import pyplot as plt

def compute_bias_across_leads(dataframes, bias_type, max_lt, ground_truth_df, fuxi_ds=None, filter_type=None):
    results = []
    assert bias_type in ['amplitude', 'phase'], 'Bias type must be either amplitude or phase'
    for lt in tqdm(range(max_lt), 'Computing per-lead metric'):
        preds, truths = [], []
        for i, df in enumerate(dataframes):
            if fuxi_ds is not None:
                fuxi_df = fuxi_ds[i]
                assert filter_type is not None and filter_type in ['high', 'low'], 'Filter type high or low must be provided'
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
            num = (pred_df.RMM1.values*truth_df.RMM2.values - pred_df.RMM2.values*truth_df.RMM1.values)
            den = (pred_df.RMM1.values*truth_df.RMM1.values + pred_df.RMM2.values*truth_df.RMM2.values)
            result = np.arctan2(num, den).mean()
        results.append(result)
    return np.array(results)

ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
fuxi_path = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi'
ground_truth_ds = load_rmm_indices(ground_truth_path)
forecast_dirs = [
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/1d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/10d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/90d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/180d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/360d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/720d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/1d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/10d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/90d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/180d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/360d_hist/logs/version_1/outputs',
        # '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/720d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/baselines/four_groups/mean_bias',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/baselines/four_groups/MLR_with_doy',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/bias-correction/LSTM/combined',
    ]
amplitude_errs = []
phase_errs = []
start_date = '2019-01-01'
end_date = '2022-01-01'
fuxi_ds, max_lt, _ = load_forecast(fuxi_path, start_date, end_date, member='mean')

for forecast_dir in forecast_dirs:
    forecast_ds, max_lt, _ = load_forecast(forecast_dir, start_date=start_date, end_date=end_date)
    amplitude_err = compute_bias_across_leads(forecast_ds, 'amplitude', max_lt, ground_truth_ds)
    phase_err = compute_bias_across_leads(forecast_ds, 'phase', max_lt, ground_truth_ds)
    amplitude_errs.append(amplitude_err)
    phase_errs.append(phase_err)
    
amplitude_err = compute_bias_across_leads(fuxi_ds, 'amplitude', max_lt, ground_truth_ds)
phase_err = compute_bias_across_leads(fuxi_ds, 'phase', max_lt, ground_truth_ds)
amplitude_errs.append(amplitude_err)
phase_errs.append(phase_err)

lead_times = np.arange(1, 43)  # lead times 1–42
labels = ['Mean bias correction', 'MVLR', 'LSTM', 'FuXi mean']
colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

plt.figure(figsize=(8, 4))

for i in range(len(amplitude_errs)):
    plt.plot(lead_times, amplitude_errs[i], color=colors[i], label=labels[i])

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.ylim(-1, 0.5)
plt.xlim(1, 42)
plt.xticks(np.arange(1, 43, 3))
plt.xlabel("Lead time (days)")
plt.ylabel("Amplitude bias")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("/glade/derecho/scratch/kvirji/mjo-predict/plots/production/2019-2021/bias-correction/amplitude_bias_vs_lead_time.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 4))

for i in range(len(phase_errs)):
    plt.plot(lead_times, phase_errs[i], color=colors[i], label=labels[i])

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.ylim(-1, 0.5)
plt.xlim(1, 42)
plt.xticks(np.arange(1, 43, 3))
plt.xlabel("Lead time (days)")
plt.ylabel("Phase bias")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("/glade/derecho/scratch/kvirji/mjo-predict/plots/production/2019-2021/bias-correction/phase_bias_vs_lead_time.png", dpi=300)
plt.close()





