import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from mjo.utils.RMM.io import load_rmm_indices, save_rmm_indices

def load_data(forecast_dir, ground_truth_path, start_date, end_date, member=None):
    forecast_data = defaultdict(list)
    ground_truth_data = defaultdict(list)
    ground_truth = load_rmm_indices(ground_truth_path)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = sorted([d for d in os.listdir(forecast_dir) if start <= datetime.strptime(d, "%Y-%m-%d" if member else "%Y-%m-%d.txt") < end])
    for d in tqdm(dates, f'Loading data from {forecast_dir}'):
        filepath = os.path.join(forecast_dir, d, f'{member}.txt') if member else os.path.join(forecast_dir, d)
        fc_df = load_rmm_indices(filepath)        
        gt_df = ground_truth.loc[fc_df.index] 
        init_date = fc_df.index[0] - timedelta(days=1)
        phase = int(ground_truth.loc[init_date].phase)            
        fc_sample = np.array([fc_df['RMM1'].values, fc_df['RMM2'].values]).T
        gt_sample = np.array([gt_df['RMM1'].values, gt_df['RMM2'].values]).T
        forecast_data[phase].append(fc_sample)
        ground_truth_data[phase].append(gt_sample)
    for phase in forecast_data.keys():
        forecast_data[phase] = np.array(forecast_data[phase])
        ground_truth_data[phase] = np.array(ground_truth_data[phase])
    return forecast_data, ground_truth_data

def inference(forecast_dir, save_dir, ground_truth_path, model, start_date, end_date, member=None):
    os.makedirs(save_dir, exist_ok=True)
    ground_truth = load_rmm_indices(ground_truth_path)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = sorted([d for d in os.listdir(forecast_dir) if start <= datetime.strptime(d, "%Y-%m-%d" if member else "%Y-%m-%d.txt") < end])
    for d in tqdm(dates, f'Loading data from {forecast_dir}'):
        filepath = os.path.join(forecast_dir, d, f'{member}.txt') if member else os.path.join(forecast_dir, d)
        fc_df = load_rmm_indices(filepath)        
        init_date = fc_df.index[0] - timedelta(days=1)
        phase = int(ground_truth.loc[init_date].phase)            
        fc_sample = np.array([fc_df['RMM1'].values, fc_df['RMM2'].values]).T
        corrected = fc_sample - model[phase]
        filename = os.path.join(save_dir, f"{str(init_date).split(' ')[0]}.txt")
        save_rmm_indices(fc_df.index, corrected[:, 0], corrected[:, 1], filename, method_str='Mean Bias Correction')
    return

forecast_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi'
ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
save_dir = '/glade/derecho/scratch/kvirji/mjo-predict/exps/baselines/mean_bias_correction/2020-2021'
train_start = '1979-01-01'
train_end = '2020-01-01' 
test_end = '2022-01-01'
train_forecast, train_target = load_data(forecast_dir=forecast_dir, ground_truth_path=ground_truth_path, start_date=train_start, end_date=train_end, member='mean')

mean_bias = defaultdict(list)
for phase in train_forecast.keys():
    mean_bias[phase] = np.mean(train_forecast[phase] - train_target[phase], axis=0)

inference(forecast_dir=forecast_dir, save_dir=save_dir, ground_truth_path=ground_truth_path, model=mean_bias, start_date=train_end, end_date=test_end, member='mean')




