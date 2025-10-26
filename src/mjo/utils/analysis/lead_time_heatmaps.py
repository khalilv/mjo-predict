import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from mjo.utils.analysis.utils import load_forecast
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import (
    bivariate_correlation_vs_lead_time_heatmap,
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

def compute_metric_across_leads(dataframes, max_lt, ground_truth_df, metric_fn, filter_low_amp_init=False, fuxi_ds=None, filter_type=None):
    results = []
    for lt in tqdm(range(max_lt), 'Computing per-lead metric'):
        preds, truths = [], []
        for i, df in enumerate(dataframes):
            if filter_low_amp_init:
                init_date = df.index[0] - timedelta(days=1)
                init_amplitude = ground_truth_df.loc[init_date].amplitude
                if init_amplitude < 1: 
                    continue
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
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/0d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/1d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/10d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/90d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/180d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/360d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TFT/720d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/0d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/1d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/10d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/90d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/180d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/360d_hist/logs/version_1/outputs',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/TSMixer/720d_hist/logs/version_1/outputs',
    ]
    ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
    fuxi_path = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi'
    output_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/plots/production/2019-2021/history-only'
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = load_rmm_indices(ground_truth_path)

    correlations = []
    event_correlations = []
    max_lead_times = []


    for predict_dir in deterministic_dirs:
        dfs, max_lt, date_strs = load_forecast(predict_dir, start_date, end_date)
        bcorr = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcorr)
        bcorr_events = compute_metric_across_leads(dfs, max_lt, ground_truth, compute_bcorr, filter_low_amp_init=True)
        correlations.append(bcorr)        
        max_lead_times.append(max_lt)
        event_correlations.append(bcorr_events)
    
    fuxi_ds, fuxi_max_lt, _ = load_forecast(fuxi_path, start_date, end_date, member='mean')
    bcorr_fuxi = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcorr)
    bcorr_fuxi_events = compute_metric_across_leads(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcorr, filter_low_amp_init=True)

    bivariate_correlation_vs_lead_time_heatmap(
        lead_times=np.expand_dims(np.arange(1,43), 0).repeat(2, axis=0),
        lookbacks=np.expand_dims(np.array([0,1,10,90,180,360,720]), 0).repeat(2, axis=0), 
        correlations=np.array(correlations).reshape(2, 7, 42),
        labels=['TFT', 'TSMixer'], 
        output_filename=os.path.join(output_dir, 'bcorr_heatmap.png'),
        # fuxi_correlations=bcorr_fuxi
    )

    bivariate_correlation_vs_lead_time_heatmap(
        lead_times=np.expand_dims(np.arange(1,43), 0).repeat(2, axis=0),
        lookbacks=np.expand_dims(np.array([0,1,10,90,180,360,720]), 0).repeat(2, axis=0), 
        correlations=np.array(event_correlations).reshape(2, 7, 42),
        labels=['TFT', 'TSMixer'], 
        output_filename=os.path.join(output_dir, 'bcorr_high_amp_events_heatmap.png'),
        # fuxi_correlations=bcorr_fuxi_events
    )

if __name__ == "__main__":
    main()