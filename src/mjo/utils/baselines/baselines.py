import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from mjo.utils.RMM.io import load_rmm_indices, save_rmm_indices


PHASE_GROUP_MAP = {
    1: 1, 8: 8,
    2: 2, 3: 3,
    4: 4, 5: 5,
    6: 6, 7: 7
}

def get_grouped_phase(phase, use_grouping=False):
    return PHASE_GROUP_MAP[phase] if use_grouping else phase

def compute_phase(rmm1, rmm2):
    return np.arctan2(rmm2, rmm1)

def compute_amplitude(rmm1, rmm2):
    return np.sqrt(rmm1**2 + rmm2**2)

def rescale_to_fc_amplitude(corrected, fc_sample, eps=1e-8):
    """
    corrected: (T,2) vectors you want to keep the *direction* of
    fc_sample: (T,2) whose amplitudes you want to match
    """
    A_in   = np.linalg.norm(fc_sample, axis=1, keepdims=True)      
    A_corr = np.linalg.norm(corrected,  axis=1, keepdims=True)

    # safe ratio; if corrected is ~0, leave that row as the original fc_sample
    ratio = np.divide(A_in, A_corr, out=np.ones_like(A_corr), where=A_corr > eps)
    out = corrected * ratio
    
    # optional: fallback when corrected has ~zero amplitude (direction undefined)
    zero_mask = (A_corr <= eps).squeeze(-1)
    out[zero_mask] = fc_sample[zero_mask]
    return out

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

def fit_mean_bias_model(forecast_data, ground_truth_data):
    mean_bias = {}
    for phase in forecast_data:
        mean_bias[phase] = np.mean(forecast_data[phase] - ground_truth_data[phase], axis=0)
    return mean_bias

def fit_mean_phase_bias_model(forecast_data, ground_truth_data):
    """
    Learn mean Δθ error per (phase group, lead)
    Δθ = θ_g - θ_f; circular mean via atan2(mean(sin), mean(cos)).
    """
    rot = {}
    for phase in forecast_data:
        fc = forecast_data[phase]                       # (B, T, 2)
        gt = ground_truth_data[phase]                   
        dot  = np.sum(fc[...,0]*gt[...,0] + fc[...,1]*gt[...,1], axis=0)   # ~ Σ cos Δθ_i * ‖X‖‖Y‖
        cross= np.sum(fc[...,0]*gt[...,1] - fc[...,1]*gt[...,0], axis=0)   # ~ Σ sin Δθ_i * ‖X‖‖Y‖
        mean_dtheta = np.arctan2(cross, dot)
        rot[phase] = mean_dtheta
    return rot

def fit_procrustes_model(forecast_data, ground_truth_data, phase_only=False):
    """
    Per (phase, lead) Procrustes:
      if phase_only:   Ŷ = X @ R                  (rotation only; amplitude preserved)
      else:            Ŷ = s * (X @ R)           (rotation + uniform scale)
    Returns:
      params['R'][phase] -> (T,2,2)
      params['s'][phase] -> (T,)
    """
    R_out, s_out = defaultdict(dict), defaultdict(dict)
    for phase in forecast_data:
        X = forecast_data[phase]  # (B,T,2)
        Y = ground_truth_data[phase]
        _, T, _ = X.shape
        R_list, s_list = [], []
        for t in range(T):
            Xt = X[:, t, :]                    # (B,2)
            Yt = Y[:, t, :]                    # (B,2)
            C  = Xt.T @ Yt                     # 2x2
            U, _, Vt = np.linalg.svd(C)
            Rt = U @ np.diag([1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            if phase_only:
                st = 1.0
            else:
                st = np.trace(Rt.T @ C) / np.sum(np.sum(Xt**2, axis=1))
            R_list.append(Rt); s_list.append(st)
        R_out[phase] = np.stack(R_list, axis=0)      # (T,2,2)
        s_out[phase] = np.array(s_list, dtype=float) # (T,)
    return {"R": R_out, "s": s_out}

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
    elif method == 'mlr_rescale':
        corrected = np.zeros_like(fc_sample)
        for t in range(fc_sample.shape[0]):
            W = model['weights'][phase][t]
            b = model['biases'][phase][t]
            corrected[t] = W @ fc_sample[t] + b
        corrected = rescale_to_fc_amplitude(corrected, fc_sample)
        return corrected
    elif method == 'mean_phase_bias':
        delta = model[phase]                    # (T,)
        c, s = np.cos(delta), np.sin(delta)     # (T,), (T,)
        rmm1, rmm2 = fc_sample[:, 0], fc_sample[:, 1]
        corrected = np.empty_like(fc_sample)
        corrected[:, 0] =  c * rmm1 - s * rmm2  # elementwise per lead
        corrected[:, 1] =  s * rmm1 + c * rmm2
        return corrected
    elif method == 'procrustes':
        # per-lead: Y_t = s_t * (x_t @ R_t)
        R = model['R'][phase]       # (T,2,2)
        s = model['s'][phase]       # (T,)
        y = np.einsum('td,tdk->tk', fc_sample, R)
        y *= s[:, None]
        return y
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

forecast_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/U250/FuXi'
ground_truth_path = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
train_start = '1979-01-01'
train_end = '2019-01-01'
test_end = '2023-01-01'
member = 'mean'
n_groups = 'eight_groups'

# Grouped mean bias correction
train_forecast, train_target = load_data(forecast_dir, ground_truth_path, train_start, train_end, member, use_grouping=True)
mean_bias_model = fit_mean_bias_model(train_forecast, train_target)
save_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/exps/baselines/2019-2021/{n_groups}/mean_bias/'
run_inference(forecast_dir, save_dir, ground_truth_path, mean_bias_model, train_end, test_end, member, method='mean_bias', use_grouping=True)

# Grouped multilinear regression
train_forecast, train_target = load_data(forecast_dir, ground_truth_path, train_start, train_end, member, use_grouping=True)
mlr_model = fit_multilinear_bias_model(train_forecast, train_target)
save_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/exps/baselines/2019-2021/{n_groups}/MLR/'
run_inference(forecast_dir, save_dir, ground_truth_path, mlr_model, train_end, test_end, member, method='mlr', use_grouping=True)

# Grouped multilinear regression
train_forecast, train_target = load_data(forecast_dir, ground_truth_path, train_start, train_end, member, use_grouping=True)
mlr_model = fit_multilinear_bias_model(train_forecast, train_target)
save_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/exps/baselines/2019-2021/{n_groups}/MLR_with_phase_rescale/'
run_inference(forecast_dir, save_dir, ground_truth_path, mlr_model, train_end, test_end, member, method='mlr_rescale', use_grouping=True)

# Phase-only mean bias model
train_forecast, train_target = load_data(forecast_dir, ground_truth_path, train_start, train_end, member, use_grouping=True)
mean_phase_model = fit_mean_phase_bias_model(train_forecast, train_target)
save_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/exps/baselines/2019-2021/{n_groups}/mean_phase_bias/'
run_inference(forecast_dir, save_dir, ground_truth_path, mean_phase_model, train_end, test_end, member, method='mean_phase_bias', use_grouping=True)

#Procrustes transform
train_forecast, train_target = load_data(forecast_dir, ground_truth_path, train_start, train_end, member, use_grouping=True)
procrustes_model = fit_procrustes_model(train_forecast, train_target, phase_only=False)
save_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/exps/baselines/2019-2021/{n_groups}/procrustes/'
run_inference(forecast_dir, save_dir, ground_truth_path, procrustes_model, train_end, test_end, member, method='procrustes', use_grouping=True)

#Phase-only procrustes transform 
train_forecast, train_target = load_data(forecast_dir, ground_truth_path, train_start, train_end, member, use_grouping=True)
procrustes_phase_model = fit_procrustes_model(train_forecast, train_target, phase_only=True)
save_dir = f'/glade/derecho/scratch/kvirji/mjo-predict/exps/baselines/2019-2021/{n_groups}/procrustes_phase/'
run_inference(forecast_dir, save_dir, ground_truth_path, procrustes_phase_model, train_end, test_end, member, method='procrustes', use_grouping=True)