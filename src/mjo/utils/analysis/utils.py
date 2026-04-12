import os
import yaml
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Optional, Tuple
import pandas as pd
from mjo.utils.RMM.io import load_rmm_indices


# =============================================================================
# Data loading
# =============================================================================

def load_forecast(predict_dir, start_date, end_date, member=None):
    """Load forecast data from directory for a date range."""
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Metrics
# =============================================================================

def compute_bcor(
    predict_rmm1: np.ndarray,
    ground_truth_rmm1: np.ndarray,
    predict_rmm2: np.ndarray,
    ground_truth_rmm2: np.ndarray
) -> float:
    """Compute bivariate correlation between predicted and ground truth RMM indices."""
    n = np.sum((predict_rmm1 * ground_truth_rmm1) + (predict_rmm2 * ground_truth_rmm2))
    d1 = np.sqrt(np.sum(np.square(predict_rmm1) + np.square(predict_rmm2)))
    d2 = np.sqrt(np.sum(np.square(ground_truth_rmm1) + np.square(ground_truth_rmm2)))
    return n / (d1*d2)


def compute_bmse(
    predict_rmm1: np.ndarray,
    ground_truth_rmm1: np.ndarray,
    predict_rmm2: np.ndarray,
    ground_truth_rmm2: np.ndarray
) -> List[float]:
    """Compute bivariate MSE decomposed into amplitude and phase components."""
    predict_amplitude = np.sqrt(np.square(predict_rmm1) + np.square(predict_rmm2))
    predict_phase = np.arctan2(predict_rmm2, predict_rmm1)
    ground_truth_amplitude = np.sqrt(np.square(ground_truth_rmm1) + np.square(ground_truth_rmm2))
    ground_truth_phase = np.arctan2(ground_truth_rmm2, ground_truth_rmm1)

    bmse = np.mean(np.square(predict_rmm1 - ground_truth_rmm1) + np.square(predict_rmm2 - ground_truth_rmm2))
    bmsea = np.mean(np.square(predict_amplitude - ground_truth_amplitude))
    bmsep = np.mean(2*predict_amplitude*ground_truth_amplitude*(1-np.cos(predict_phase - ground_truth_phase)))
    assert np.isclose(bmse, bmsea + bmsep), f'Found mismatch between BMSE {bmse} and components BMSEa {bmsea}, BMSEp {bmsep}'
    return [bmsea, bmsep]


# =============================================================================
# Bootstrap utils
# =============================================================================

def _generate_block_bootstrap_indices(
    n_samples: int, n_obs: int, block_size: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """Generate resampled index arrays for block bootstrap.

    Draws ceil(n_obs / block_size) contiguous blocks with replacement,
    each of length block_size, and truncates to n_obs.
    """
    n_blocks = int(np.ceil(n_obs / block_size))
    indices = []
    for _ in range(n_samples):
        block_starts = rng.integers(0, n_obs, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, n_obs)) for s in block_starts])
        indices.append(idx[:n_obs])
    return indices


def _resolve_bootstrap_positions(lt_indices: np.ndarray, boot_idx: np.ndarray) -> np.ndarray:
    """Map bootstrap init-date indices to row positions in per-lead-time arrays."""
    idx_to_pos = {}
    for pos, idx in enumerate(lt_indices):
        idx_to_pos.setdefault(idx, []).append(pos)
    positions = []
    for idx in boot_idx:
        if idx in idx_to_pos:
            positions.extend(idx_to_pos[idx])
    return np.array(positions) if positions else np.array([], dtype=int)


def _resample_metric_at_lt(boot_idx, lt_indices, lt_preds, lt_truths, metric_fn):
    """Compute a metric on bootstrap-resampled data at one lead time."""
    positions = _resolve_bootstrap_positions(lt_indices, boot_idx)
    if len(positions) == 0:
        return np.nan
    pred_df = lt_preds.iloc[positions]
    truth_df = lt_truths.iloc[positions]
    return metric_fn(
        pred_df.RMM1.values, truth_df.RMM1.values,
        pred_df.RMM2.values, truth_df.RMM2.values
    )


def _collect_per_lt_data(
    dataframes: List[pd.DataFrame],
    max_lt: int,
    ground_truth_df: pd.DataFrame,
    fuxi_ds: Optional[List[pd.DataFrame]] = None,
    filter_type: Optional[str] = None
) -> Tuple[List[np.ndarray], List[pd.DataFrame], List[pd.DataFrame]]:
    """Pre-collect (pred, truth) pairs per lead time for bootstrap resampling.

    Returns parallel lists of (df_indices, pred_DataFrames, truth_DataFrames),
    one entry per lead time. Optional amplitude filtering via fuxi_ds.
    """
    per_lt_indices, per_lt_preds, per_lt_truths = [], [], []
    for lt in range(max_lt):
        df_indices, preds, truths = [], [], []
        for i, df in enumerate(dataframes):
            if fuxi_ds is not None:
                assert filter_type in ('high', 'low')
                fuxi_amplitude = fuxi_ds[i].loc[df.index[lt]].amplitude
                if (filter_type == 'low' and fuxi_amplitude < 1) or (filter_type == 'high' and fuxi_amplitude >= 1):
                    continue
            if lt < len(df):
                df_indices.append(i)
                preds.append(df.iloc[lt])
                truths.append(ground_truth_df.loc[df.index[lt]])
        per_lt_indices.append(np.array(df_indices))
        per_lt_preds.append(pd.DataFrame(preds))
        per_lt_truths.append(pd.DataFrame(truths))
    return per_lt_indices, per_lt_preds, per_lt_truths


def _first_crossing(vals: np.ndarray, threshold: float) -> float:
    """Find first index where values drop below threshold (with linear interpolation)."""
    v = np.asarray(vals, dtype=float)
    if v[0] < threshold:
        return 0.0
    for j in range(len(v) - 1):
        v0, v1 = v[j], v[j + 1]
        if np.isnan(v0) or np.isnan(v1):
            continue
        if (v0 >= threshold) and (v1 < threshold):
            if v1 == v0:
                return float(j)
            frac = (threshold - v0) / (v1 - v0)
            return float(j) + float(frac)
    return float("nan")


def _bootstrap_model_metrics(boot_idx, all_per_lt, max_lts, metric_fn, n_models):
    """Compute a metric at all lead times for all models on one bootstrap replicate.

    Returns list of arrays, one per model. For list-valued metrics (BMSE),
    sums the components (amplitude + phase).
    """
    model_metrics = []
    for m in range(n_models):
        per_lt_indices, per_lt_preds, per_lt_truths = all_per_lt[m]
        vals = np.zeros(max_lts[m])
        for lt in range(max_lts[m]):
            result = _resample_metric_at_lt(
                boot_idx, per_lt_indices[lt], per_lt_preds[lt], per_lt_truths[lt], metric_fn
            )
            vals[lt] = sum(result) if isinstance(result, list) else result
        model_metrics.append(vals)
    return model_metrics

def block_bootstrap_bias(
    dataframes: List[pd.DataFrame],
    bias_type: str,
    max_lt: int,
    ground_truth_df: pd.DataFrame,
    n_samples: int = 1000,
    block_size: int = 1,
    ci_level: float = 0.95,
    seed: int = 42,
    fuxi_ds: Optional[List[pd.DataFrame]] = None,
    filter_type: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap CIs for amplitude or phase bias across lead times."""
    assert bias_type in ('amplitude', 'phase')
    rng = np.random.default_rng(seed)
    alpha = (1 - ci_level) / 2

    per_lt_indices, per_lt_preds, per_lt_truths = _collect_per_lt_data(
        dataframes, max_lt, ground_truth_df, fuxi_ds, filter_type
    )
    bootstrap_indices = _generate_block_bootstrap_indices(n_samples, len(dataframes), block_size, rng)
    boot_results = np.zeros((n_samples, max_lt))

    for b in range(n_samples):
        for lt in range(max_lt):
            positions = _resolve_bootstrap_positions(per_lt_indices[lt], bootstrap_indices[b])
            if len(positions) == 0:
                boot_results[b, lt] = np.nan
                continue
            pred_df = per_lt_preds[lt].iloc[positions]
            truth_df = per_lt_truths[lt].iloc[positions]
            if bias_type == 'amplitude':
                boot_results[b, lt] = (pred_df.amplitude.values - truth_df.amplitude.values).mean()
            else:
                num = pred_df.RMM1.values * truth_df.RMM2.values - pred_df.RMM2.values * truth_df.RMM1.values
                den = pred_df.RMM1.values * truth_df.RMM1.values + pred_df.RMM2.values * truth_df.RMM2.values
                boot_results[b, lt] = np.arctan2(num, den).mean()

    return (np.nanpercentile(boot_results, alpha * 100, axis=0),
            np.nanpercentile(boot_results, (1 - alpha) * 100, axis=0))


def block_bootstrap_crossing(
    dataframes: List[pd.DataFrame],
    max_lt: int,
    ground_truth_df: pd.DataFrame,
    metric_fn: Callable,
    threshold: float = 0.5,
    n_samples: int = 1000,
    block_size: int = 1,
    ci_level: float = 0.95,
    seed: int = 42,
    filter_low_amp_init: bool = False
) -> Tuple[float, float]:
    """Bootstrap CI for the lead time where a metric first crosses a threshold.

    Used for heatmap threshold bar tails.
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - ci_level) / 2

    # Custom collection for filter_low_amp_init (filters by init-date amplitude)
    per_lt_indices, per_lt_preds, per_lt_truths = [], [], []
    for lt in range(max_lt):
        df_indices, preds, truths = [], [], []
        for i, df in enumerate(dataframes):
            if filter_low_amp_init:
                init_date = df.index[0] - timedelta(days=1)
                if ground_truth_df.loc[init_date].amplitude < 1:
                    continue
            if lt < len(df):
                df_indices.append(i)
                preds.append(df.iloc[lt])
                truths.append(ground_truth_df.loc[df.index[lt]])
        per_lt_indices.append(np.array(df_indices))
        per_lt_preds.append(pd.DataFrame(preds))
        per_lt_truths.append(pd.DataFrame(truths))

    bootstrap_indices = _generate_block_bootstrap_indices(n_samples, len(dataframes), block_size, rng)
    crossings = np.zeros(n_samples)
    for b in range(n_samples):
        metric_vals = np.array([
            _resample_metric_at_lt(bootstrap_indices[b], per_lt_indices[lt], per_lt_preds[lt], per_lt_truths[lt], metric_fn)
            for lt in range(max_lt)
        ])
        c = _first_crossing(metric_vals, threshold)
        # Treat "never crosses" as crossing at max_lt (skillful for full window)
        crossings[b] = c if not np.isnan(c) else max_lt

    return (np.percentile(crossings, alpha * 100),
            np.percentile(crossings, (1 - alpha) * 100))


def block_bootstrap_paired_improvement(
    all_model_dataframes: List[List[pd.DataFrame]],
    max_lts: List[int],
    ground_truth_df: pd.DataFrame,
    metric_fn: Callable,
    biweekly_groups: List[Tuple[int, int]],
    higher_is_better: bool = True,
    n_samples: int = 1000,
    block_size: int = 1,
    ci_level: float = 0.95,
    seed: int = 42,
    fuxi_ds: Optional[List[pd.DataFrame]] = None,
    filter_type: Optional[str] = None
) -> List[List[Tuple[float, float]]]:
    """Paired bootstrap CIs for relative improvement of each model vs baseline.

    The last entry in all_model_dataframes is treated as the baseline.

    Returns list of [(ci_lo, ci_hi)] per model (excluding baseline), per biweekly group.
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - ci_level) / 2
    n_models = len(all_model_dataframes)
    baseline_idx = n_models - 1

    all_per_lt = [_collect_per_lt_data(all_model_dataframes[m], max_lts[m], ground_truth_df, fuxi_ds, filter_type)
                  for m in range(n_models)]

    n_obs = max(len(dfs) for dfs in all_model_dataframes)
    bootstrap_indices = _generate_block_bootstrap_indices(n_samples, n_obs, block_size, rng)

    # For each replicate, compute improvement of each model vs baseline
    boot_improvements = np.zeros((n_samples, n_models - 1, len(biweekly_groups)))
    for b in range(n_samples):
        model_metrics = _bootstrap_model_metrics(bootstrap_indices[b], all_per_lt, max_lts, metric_fn, n_models)
        baseline = model_metrics[baseline_idx]
        for m in range(n_models - 1):
            for g, (start, end) in enumerate(biweekly_groups):
                m_mean = np.nanmean(model_metrics[m][start:end+1])
                b_mean = np.nanmean(baseline[start:end+1])
                if abs(b_mean) < 1e-10:
                    boot_improvements[b, m, g] = np.nan
                elif higher_is_better:
                    boot_improvements[b, m, g] = (m_mean - b_mean) / abs(b_mean) * 100
                else:
                    boot_improvements[b, m, g] = (b_mean - m_mean) / abs(b_mean) * 100

    return [[
        (np.nanpercentile(boot_improvements[:, m, g], alpha * 100),
         np.nanpercentile(boot_improvements[:, m, g], (1 - alpha) * 100))
        for g in range(len(biweekly_groups))
    ] for m in range(n_models - 1)]


def block_bootstrap_per_lead_significance(
    all_model_dataframes: List[List[pd.DataFrame]],
    max_lts: List[int],
    ground_truth_df: pd.DataFrame,
    metric_fn: Callable,
    higher_is_better: bool = True,
    n_samples: int = 1000,
    block_size: int = 1,
    ci_level: float = 0.95,
    seed: int = 42,
    fuxi_ds: Optional[List[pd.DataFrame]] = None,
    filter_type: Optional[str] = None,
) -> List[np.ndarray]:
    """Per-lead-time statistical significance of each model vs the baseline.

    For each bootstrap replicate, computes the paired difference
    (model - baseline) at each lead time using the same resampled init dates.
    A lead time is significant if its 95% CI excludes zero.

    Returns list of boolean arrays (one per model excluding baseline).
    True = CI excludes zero (significant improvement or degradation).
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - ci_level) / 2
    n_models = len(all_model_dataframes)
    baseline_idx = n_models - 1

    all_per_lt = [_collect_per_lt_data(all_model_dataframes[m], max_lts[m], ground_truth_df, fuxi_ds, filter_type)
                  for m in range(n_models)]

    n_obs = max(len(dfs) for dfs in all_model_dataframes)
    bootstrap_indices = _generate_block_bootstrap_indices(n_samples, n_obs, block_size, rng)

    # Accumulate paired differences: model minus baseline at each lead time
    boot_diffs = [np.zeros((n_samples, max_lts[m])) for m in range(n_models - 1)]
    for b in range(n_samples):
        model_metrics = _bootstrap_model_metrics(bootstrap_indices[b], all_per_lt, max_lts, metric_fn, n_models)
        baseline = model_metrics[baseline_idx]
        for m in range(n_models - 1):
            if higher_is_better:
                boot_diffs[m][b] = model_metrics[m] - baseline[:max_lts[m]]
            else:
                boot_diffs[m][b] = baseline[:max_lts[m]] - model_metrics[m]

    # Significant where the entire CI is above or below zero
    significance = []
    for m in range(n_models - 1):
        ci_lo = np.nanpercentile(boot_diffs[m], alpha * 100, axis=0)
        ci_hi = np.nanpercentile(boot_diffs[m], (1 - alpha) * 100, axis=0)
        significance.append((ci_lo > 0) | (ci_hi < 0))
    return significance


def block_bootstrap_skillful_window(
    all_model_dataframes: List[List[pd.DataFrame]],
    max_lts: List[int],
    ground_truth_df: pd.DataFrame,
    threshold: float = 0.5,
    n_samples: int = 1000,
    block_size: int = 1,
    ci_level: float = 0.95,
    seed: int = 42,
) -> List[Tuple[float, float, float]]:
    """Bootstrap CI on the skillful window (lead time where BCOR drops below threshold).

    Returns (point_estimate, ci_lower, ci_upper) in days for each model.
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - ci_level) / 2
    n_models = len(all_model_dataframes)

    all_per_lt = [_collect_per_lt_data(all_model_dataframes[m], max_lts[m], ground_truth_df)
                  for m in range(n_models)]

    n_obs = max(len(dfs) for dfs in all_model_dataframes)
    bootstrap_indices = _generate_block_bootstrap_indices(n_samples, n_obs, block_size, rng)

    boot_windows = [np.zeros(n_samples) for _ in range(n_models)]
    for b in range(n_samples):
        for m in range(n_models):
            per_lt_indices, per_lt_preds, per_lt_truths = all_per_lt[m]
            bcor_vals = np.array([
                _resample_metric_at_lt(bootstrap_indices[b], per_lt_indices[lt], per_lt_preds[lt], per_lt_truths[lt], compute_bcor)
                for lt in range(max_lts[m])
            ])
            crossing = _first_crossing(bcor_vals, threshold)
            boot_windows[m][b] = crossing + 1 if not np.isnan(crossing) else max_lts[m]

    results = []
    for m in range(n_models):
        # Point estimate from full (non-resampled) data
        per_lt_indices, per_lt_preds, per_lt_truths = all_per_lt[m]
        all_idx = np.arange(len(all_model_dataframes[m]))
        full_bcor = np.array([
            _resample_metric_at_lt(all_idx, per_lt_indices[lt], per_lt_preds[lt], per_lt_truths[lt], compute_bcor)
            for lt in range(max_lts[m])
        ])
        crossing = _first_crossing(full_bcor, threshold)
        point_est = crossing + 1 if not np.isnan(crossing) else max_lts[m]
        ci_lo = np.nanpercentile(boot_windows[m], alpha * 100)
        ci_hi = np.nanpercentile(boot_windows[m], (1 - alpha) * 100)
        results.append((point_est, ci_lo, ci_hi))
    return results
