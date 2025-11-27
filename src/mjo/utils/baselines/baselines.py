import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from mjo.utils.RMM.io import load_rmm_indices, save_rmm_indices
from mjo.utils.analysis.utils import load_config


def create_phase_group_map(num_groups):
    """
    Create phase grouping map based on number of groups.

    Args:
        num_groups: Number of phase groups (1, 4, or 8)

    Returns:
        Dictionary mapping original phases (1-8) to group numbers
    """
    if num_groups == 1:
        # All phases map to single group
        return {i: 1 for i in range(1, 9)}
    elif num_groups == 4:
        # Combine adjacent phases into 4 groups
        return {
            1: 1, 8: 1,  
            2: 2, 3: 2,
            4: 3, 5: 3,
            6: 4, 7: 4
        }
    elif num_groups == 8:
        # Each phase is its own group
        return {i: i for i in range(1, 9)}
    else:
        raise ValueError(f"num_groups must be 1, 4, or 8, got {num_groups}")


def get_grouped_phase(phase, phase_group_map):
    """Map individual MJO phase to grouped phase using provided mapping."""
    return phase_group_map[phase]

def load_data(forecast_dir, ground_truth_path, start_date, end_date, member=None, phase_group_map=None, filter_low_amplitude_samples=False, include_doy=False):
    """Load and organize forecast and ground truth data by MJO phase."""
    forecast_data = defaultdict(list)
    ground_truth_data = defaultdict(list)
    ground_truth = load_rmm_indices(ground_truth_path)

    # Determine date format based on whether we're loading ensemble members
    date_format = "%Y-%m-%d" if member else "%Y-%m-%d.txt"
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = sorted([d for d in os.listdir(forecast_dir) if start <= datetime.strptime(d, date_format) < end])

    for d in tqdm(dates, f'Loading data from {forecast_dir}'):
        filepath = os.path.join(forecast_dir, d, f'{member}.txt') if member else os.path.join(forecast_dir, d)
        fc_df = load_rmm_indices(filepath)
        gt_df = ground_truth.loc[fc_df.index]

        # Get initialization date and phase
        init_date = fc_df.index[0] - timedelta(days=1)

        # Skip low amplitude events if requested
        if filter_low_amplitude_samples and ground_truth.loc[init_date].amplitude < 1:
            continue

        raw_phase = int(ground_truth.loc[init_date].phase)
        phase = get_grouped_phase(raw_phase, phase_group_map) if phase_group_map else raw_phase

        # Include day-of-year as features if requested
        if include_doy:
            doy = fc_df.index.day_of_year.values
            doy_sin = np.sin(2*np.pi*doy/366)
            doy_cos = np.sin(2*np.pi*doy/366)
            fc_sample = np.array([fc_df['RMM1'].values, fc_df['RMM2'].values, doy_sin, doy_cos]).T
        else:
            fc_sample = np.array([fc_df['RMM1'].values, fc_df['RMM2'].values]).T

        gt_sample = np.array([gt_df['RMM1'].values, gt_df['RMM2'].values]).T
        forecast_data[phase].append(fc_sample)
        ground_truth_data[phase].append(gt_sample)

    # Convert lists to numpy arrays
    for phase in forecast_data:
        forecast_data[phase] = np.array(forecast_data[phase])
        ground_truth_data[phase] = np.array(ground_truth_data[phase])

    return forecast_data, ground_truth_data

def fit_mean_bias_model(forecast_data, ground_truth_data):
    """Fit mean bias correction model for each phase."""
    mean_bias = {}
    for phase in forecast_data:
        # Compute average bias across all samples for this phase
        mean_bias[phase] = np.mean(forecast_data[phase] - ground_truth_data[phase], axis=0)
    return mean_bias


def fit_mvlr_model(forecast_data, ground_truth_data):
    """Fit multivariate linear regression model for each phase and lead time."""
    weights = defaultdict(dict)
    biases = defaultdict(dict)

    for phase in forecast_data:
        fc = forecast_data[phase]  # (B, T, M) - batch, time, features
        gt = ground_truth_data[phase]  # (B, T, 2) - batch, time, RMM1/RMM2
        B, T, M = fc.shape

        # Fit separate linear model for each lead time
        for t in range(T):
            X = fc[:, t, :]  # Features at lead time t
            Y = gt[:, t, :]  # Ground truth at lead time t

            # Augment with bias term and solve least squares
            X_aug = np.hstack([X, np.ones((B, 1))])
            W_aug, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)

            # Split into weights and biases
            weights[phase][t] = W_aug[:M, :].T
            biases[phase][t] = W_aug[M, :]

    return {'weights': weights, 'biases': biases}

def correct_forecast(fc_sample, phase, model, method):
    """Apply bias correction to a single forecast sample."""
    if method == 'mean_bias':
        # Subtract phase-specific mean bias
        return fc_sample - model[phase]
    elif method == 'mvlr':
        # Apply multivariate linear regression correction for each lead time
        corrected = np.zeros((fc_sample.shape[0], 2))
        for t in range(fc_sample.shape[0]):
            W = model['weights'][phase][t]
            b = model['biases'][phase][t]
            corrected[t] = W @ fc_sample[t] + b
        return corrected
    else:
        raise ValueError(f"Unknown correction method: {method}")

def run_inference(forecast_dir, save_dir, ground_truth_path, model, start_date, end_date, member='mean', method='mean_bias', phase_group_map=None, include_doy=False):
    """Apply trained bias correction model to forecasts and save results."""
    os.makedirs(save_dir, exist_ok=True)
    ground_truth = load_rmm_indices(ground_truth_path)

    # Determine date format based on whether we're loading ensemble members
    date_format = "%Y-%m-%d" if member else "%Y-%m-%d.txt"
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = sorted([d for d in os.listdir(forecast_dir) if start <= datetime.strptime(d, date_format) < end])

    for d in tqdm(dates, f'Applying {method} correction'):
        filepath = os.path.join(forecast_dir, d, f'{member}.txt') if member else os.path.join(forecast_dir, d)
        fc_df = load_rmm_indices(filepath)

        # Get initialization phase
        init_date = fc_df.index[0] - timedelta(days=1)
        raw_phase = int(ground_truth.loc[init_date].phase)
        phase = get_grouped_phase(raw_phase, phase_group_map) if phase_group_map else raw_phase

        # Include day-of-year features if requested
        if include_doy:
            doy = fc_df.index.day_of_year.values
            doy_sin = np.sin(2*np.pi*doy/366)
            doy_cos = np.sin(2*np.pi*doy/366)
            fc_sample = np.array([fc_df['RMM1'].values, fc_df['RMM2'].values, doy_sin, doy_cos]).T
        else:
            fc_sample = np.array([fc_df['RMM1'].values, fc_df['RMM2'].values]).T

        # Apply correction and save
        corrected = correct_forecast(fc_sample, phase, model, method)
        filename = os.path.join(save_dir, f"{str(init_date.date())}.txt")
        save_rmm_indices(fc_df.index, corrected[:, 0], corrected[:, 1], filename, method_str=method.upper())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate bias-corrected MJO forecasts using statistical baseline methods',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file (required)'
    )
    return parser.parse_args()


def main():
    """Load config, train bias correction models, and generate corrected forecasts."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract configuration
    forecast_dir = config['paths']['forecast_dir']
    ground_truth_path = config['paths']['ground_truth']
    output_dir = config['paths']['output_dir']

    train_start = config['dates']['train_start']
    train_end = config['dates']['train_end']
    test_end = config['dates']['test_end']

    member = config.get('member', 'mean')
    num_phase_groups = config.get('num_phase_groups', 4)
    filter_low_amplitude = config.get('filter_low_amplitude_samples', False)

    # Create phase grouping map
    phase_group_map = create_phase_group_map(num_phase_groups)
    print(f"\nUsing {num_phase_groups} phase groups")

    # Get methods to run
    methods = config.get('methods', ['mean_bias'])

    # Run each requested method
    for method_name in methods:
        if method_name == 'mean_bias':
            print(f"\n{'='*60}")
            print(f"Training mean bias correction model")
            print(f"{'='*60}")

            # Load training data
            train_forecast, train_target = load_data(
                forecast_dir, ground_truth_path, train_start, train_end,
                member, phase_group_map=phase_group_map,
                filter_low_amplitude_samples=filter_low_amplitude
            )

            # Fit model
            mean_bias_model = fit_mean_bias_model(train_forecast, train_target)

            # Run inference
            save_dir = os.path.join(output_dir, 'mean_bias')
            run_inference(
                forecast_dir, save_dir, ground_truth_path, mean_bias_model,
                train_end, test_end, member, method='mean_bias',
                phase_group_map=phase_group_map
            )

        elif method_name == 'mvlr':
            include_doy = config.get('mvlr_include_doy', False)
            print(f"\n{'='*60}")
            print(f"Training MVLR model (include_doy={include_doy})")
            print(f"{'='*60}")

            # Load training data
            train_forecast, train_target = load_data(
                forecast_dir, ground_truth_path, train_start, train_end,
                member, phase_group_map=phase_group_map,
                filter_low_amplitude_samples=filter_low_amplitude,
                include_doy=include_doy
            )

            # Fit model
            mvlr_model = fit_mvlr_model(train_forecast, train_target)

            # Run inference
            suffix = '_with_doy' if include_doy else ''
            save_dir = os.path.join(output_dir, f'MVLR{suffix}')
            run_inference(
                forecast_dir, save_dir, ground_truth_path, mvlr_model,
                train_end, test_end, member, method='mvlr',
                phase_group_map=phase_group_map, include_doy=include_doy
            )

        else:
            print(f"Warning: Unknown method '{method_name}', skipping...")

    print(f"\n{'='*60}")
    print(f"All methods completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()