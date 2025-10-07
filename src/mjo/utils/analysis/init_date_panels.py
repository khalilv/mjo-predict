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
    bivariate_correlation_vs_lead_time_heatmap,
    hexbin_skill_vs_amplitude_plot,
    histogram_skill_vs_phase_plot,
    bivariate_correlation_by_month_multi_year_plot,
    scatter_amplitudes_by_init_date_tripanel
)

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

start_date = '2017-01-01'
end_date = '2017-03-01'
ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
fuxi_path = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi'
output_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/plots/production/2019-2021/case_study/2017/'
month = 'JAN-FEB'
data = []
os.makedirs(output_dir, exist_ok=True)
ground_truth = load_rmm_indices(ground_truth_path)
forecasted_dfs, max_lt, _ = load_forecast(fuxi_path, start_date, end_date, 'mean')

for df in forecasted_dfs:
    gt = ground_truth.loc[df.index].amplitude.values
    fc = df.amplitude.values
    score = ((gt > 1).astype(int) & (fc > 1).astype(int)) | ((gt < 1).astype(int) & (fc < 1).astype(int))
    data.append(np.stack([fc, score], axis=1))
data = np.array(data)
dates = [df.index[0] - timedelta(days=1) for df in forecasted_dfs]
scatter_amplitudes_by_init_date_tripanel(data, dates, output_filename=os.path.join(output_dir, f'forecasted_amplitudes_fuxi_{month}'))
