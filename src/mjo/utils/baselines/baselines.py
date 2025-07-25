import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from mjo.utils.RMM.io import load_rmm_indices, save_rmm_indices


PHASE_GROUP_MAP = {
    1: 1, 8: 1,
    2: 2, 3: 2,
    4: 3, 5: 3,
    6: 4, 7: 4
}

def get_grouped_phase(phase, use_grouping=False):
    return PHASE_GROUP_MAP[phase] if use_grouping else phase


def load_data(forecast_dir, ground_truth_path, start_date, end_date, member=None, use_grouping=False):
    forecast_data = defaultdict(list)
    ground_truth_data = defaultdict(list)
    ground_truth = load_rmm_indices(ground_truth_path)

    date_format = "%Y-%m-%d" if member else "%Y-%m-%d.txt"
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = sorted([d for d in os.listdir(forecast_dir) if start <= datetime.strptime(d, date_format) < end])

    for d in tqdm(dates, f'Loading data from {forecast_dir}'):
        filepath = os.path.join(forecast_dir, d, f'{member}.txt') if member else os.path.join(forecast_dir, d)
        fc_df = load_rmm_indices(filepath)
        gt_df = ground_truth.loc[fc_df.index]
        init_date = fc_df.index[0] - timedelta(days=1)
        raw_phase = int(ground_truth.loc[init_date].phase)
        phase = get_grouped_phase(raw_phase, use_grouping)
        fc_sample = np.array([fc_df['RMM1'].values, fc_df['RMM2'].values]).T
        gt_sample = np.array([gt_df['RMM1'].values, gt_df['RMM2'].values]).T
        forecast_data[phase].append(fc_sample)
        ground_truth_data[phase].append(gt_sample)

    for phase in forecast_data:
        forecast_data[phase] = np.array(forecast_data[phase])
        ground_truth_data[phase] = np.array(ground_truth_data[phase])

    return forecast_data, ground_truth_data

def compute_mean_bias_model(forecast_data, ground_truth_data):
    mean_bias = {}
    for phase in forecast_data:
        mean_bias[phase] = np.mean(forecast_data[phase] - ground_truth_data[phase], axis=0)
    return mean_bias

def fit_multilinear_bias_model(forecast_data, ground_truth_data):
    weights = defaultdict(dict)
    biases = defaultdict(dict)

    for phase in forecast_data:
        fc = forecast_data[phase]  # (B, T, 2)
        gt = ground_truth_data[phase]  # (B, T, 2)
        B, T, _ = fc.shape
        for t in range(T):
            X = fc[:, t, :]
            Y = gt[:, t, :]
            X_aug = np.hstack([X, np.ones((B, 1))])
            W_aug, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
            weights[phase][t] = W_aug[:2, :].T
            biases[phase][t] = W_aug[2, :]
    return {'weights': weights, 'biases': biases}

def correct_forecast(fc_sample, phase, model, method):
    if method == 'mean_bias':
        return fc_sample - model[phase]
    elif method == 'mlr':
        corrected = np.zeros_like(fc_sample)
        for t in range(fc_sample.shape[0]):
            W = model['weights'][phase][t]
            b = model['biases'][phase][t]
            corrected[t] = W @ fc_sample[t] + b
        return corrected
    else:
        raise ValueError(f"Unknown correction method: {method}")

def run_inference(forecast_dir, save_dir, ground_truth_path, model, start_date, end_date, member='mean', method='mean_bias', use_grouping=False):
    os.makedirs(save_dir, exist_ok=True)
    ground_truth = load_rmm_indices(ground_truth_path)
    date_format = "%Y-%m-%d" if member else "%Y-%m-%d.txt"
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = sorted([d for d in os.listdir(forecast_dir) if start <= datetime.strptime(d, date_format) < end])

    for d in tqdm(dates, f'Applying {method} correction'):
        filepath = os.path.join(forecast_dir, d, f'{member}.txt') if member else os.path.join(forecast_dir, d)
        fc_df = load_rmm_indices(filepath)
        init_date = fc_df.index[0] - timedelta(days=1)
        raw_phase = int(ground_truth.loc[init_date].phase)
        phase = get_grouped_phase(raw_phase, use_grouping)
        fc_sample = np.array([fc_df['RMM1'].values, fc_df['RMM2'].values]).T
        corrected = correct_forecast(fc_sample, phase, model, method)
        filename = os.path.join(save_dir, f"{str(init_date.date())}.txt")
        save_rmm_indices(fc_df.index, corrected[:, 0], corrected[:, 1], filename, method_str=method.upper())

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
        corrected = correct_forecast(fc_sample, phase, model['weights'], model['biases'])
        filename = os.path.join(save_dir, f"{str(init_date).split(' ')[0]}.txt")
        save_rmm_indices(fc_df.index, corrected[:, 0], corrected[:, 1], filename, method_str='MLR')
    return

forecast_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi'
ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
train_start = '1979-01-01'
train_end = '2020-01-01'
test_end = '2023-01-01'
member = 'mean'

# Grouped mean bias correction
train_forecast, train_target = load_data(forecast_dir, ground_truth_path, train_start, train_end, member, use_grouping=True)
mean_bias_model = compute_mean_bias_model(train_forecast, train_target)
save_dir = '/glade/derecho/scratch/kvirji/mjo-predict/exps/baselines/2020-2021/phase_groups/mean_bias'
run_inference(forecast_dir, save_dir, ground_truth_path, mean_bias_model, train_end, test_end, member, method='mean_bias', use_grouping=True)

# Grouped multilinear regression
train_forecast, train_target = load_data(forecast_dir, ground_truth_path, train_start, train_end, member, use_grouping=True)
mlr_model = fit_multilinear_bias_model(train_forecast, train_target)
save_dir = '/glade/derecho/scratch/kvirji/mjo-predict/exps/baselines/2020-2021/phase_groups/MLR'
run_inference(forecast_dir, save_dir, ground_truth_path, mlr_model, train_end, test_end, member, method='mlr', use_grouping=True)