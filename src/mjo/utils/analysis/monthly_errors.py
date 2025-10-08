import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import load_forecast
from mjo.utils.plot import (
    bivariate_correlation_by_month_plot,
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

def compute_metric_across_months(dataframes, max_lt, ground_truth_df, metric_fn):
    monthly_groups = defaultdict(list)
    monthly_results = defaultdict()

    for df in dataframes:
        init_date = df.index[0] - timedelta(days=1)
        monthly_groups[init_date.month].append(df)
    
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

    start_date = '2002-01-01'
    end_date = '2018-01-01'
    deterministic_dirs = [
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/baselines/four_groups/mean_bias',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/baselines/four_groups/MLR_with_doy',
        '/glade/derecho/scratch/kvirji/mjo-predict/exps/production/2019-2021/bias-correction/LSTM/combined',
    ]
    plot_labels = ['Mean bias correction', 'MVLR', 'LSTM']
    ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
    fuxi_path = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi'
    output_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/plots/production/2002-2017'
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = load_rmm_indices(ground_truth_path)
    fuxi_ds, fuxi_max_lt, _ = load_forecast(fuxi_path, start_date, end_date, member='mean')


    for i, predict_dir in enumerate(deterministic_dirs):
        dfs, max_lt, date_strs = load_forecast(predict_dir, start_date, end_date)
        bcorr_per_month = compute_metric_across_months(dfs, max_lt, ground_truth, compute_bcorr)
        bivariate_correlation_by_month_plot(
            bcorr_dict=bcorr_per_month, 
            label='Bivariate correlation',
            title='Bivariate correlation per month',
            output_filename=os.path.join(output_dir, f'bcorr_per_month_{plot_labels[i]}.png')
        )
        
    fuxi_bcorr_per_month = compute_metric_across_months(fuxi_ds, fuxi_max_lt, ground_truth, compute_bcorr)

    bivariate_correlation_by_month_plot(
        bcorr_dict=fuxi_bcorr_per_month, 
        label='Bivariate correlation',
        title='Bivariate correlation per month',
        output_filename=os.path.join(output_dir, f'bcorr_per_month_fuxi.png')
    )

if __name__ == "__main__":
    main()