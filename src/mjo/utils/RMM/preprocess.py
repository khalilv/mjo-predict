import os
import argparse
import pandas as pd
import numpy as np
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.ROMI.io import load_romi_indices
from mjo.utils.VPM.io import load_vpm_indices
from mjo.utils.plot import correlation_scatter_plot
from mjo.utils.analysis.utils import compute_bcor

# Train/val/test splits (end dates exclusive)
TRAIN_START_DATE = '2002-01-01'
VAL_START_DATE = '2018-01-01'
TEST_START_DATE = '2019-01-01'
TEST_END_DATE = '2022-02-12'


def subset_rmm_data(df, start_date, end_date, compute_statistics):
    """Subset RMM data by date range and add derived features."""
    assert isinstance(df.index, pd.DatetimeIndex), "DataFrame must have datetime index"

    subset_df = df[(df.index >= start_date) & (df.index < end_date)]

    # Add phase angle as sin/cos features
    phase = np.arctan2(subset_df['RMM2'], subset_df['RMM1'])
    subset_df['phase_sin'] = np.sin(phase)
    subset_df['phase_cos'] = np.cos(phase)

    # Add day-of-year as sin/cos features for seasonality
    doy = subset_df.index.day_of_year
    angle = 2 * np.pi * doy / 366
    subset_df['doy_sin'] = np.sin(angle)
    subset_df['doy_cos'] = np.cos(angle)

    subset_df['year'] = subset_df.index.year

    # Compute normalization statistics on training set only
    if compute_statistics:
        mean = subset_df[['RMM1', 'RMM2', 'amplitude', 'year']].mean()
        std = subset_df[['RMM1', 'RMM2', 'amplitude', 'year']].std()
        return subset_df, mean, std
    else:
        return subset_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Split observed RMM indices into train/val/test npz files and validate against BoM RMM.')
    parser.add_argument('--rmm', default='data/observed/rmm.txt', help='Observed RMM text file (default: %(default)s)')
    parser.add_argument('--bom', default='data/observed/bom_rmm.74toRealtime.txt', help='BoM RMM text file used for validation (default: %(default)s)')
    parser.add_argument('--romi', default=None, help='Optional ROMI text file (e.g. NOAA ROMI) to add ROMI1/ROMI2 as auxiliary inputs')
    parser.add_argument('--vpm', default=None, help='Optional VPM text file (e.g. NOAA VPM) to add VPM1/VPM2 as auxiliary inputs')
    parser.add_argument('--output-dir', default='data/preprocessed', help='Output directory for {train,val,test,statistics}.npz (default: %(default)s)')
    return parser.parse_args()


def main():
    """Split RMM data into train/val/test sets, compute statistics, and validate against BoM indices."""
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Optionally merge ROMI and/or VPM as auxiliary historical context.
    # Inner-join on date so any days missing in any source are dropped.
    rmm_df = load_rmm_indices(args.rmm)
    input_df = rmm_df
    aux_variables = []
    if args.romi:
        input_df = input_df.join(load_romi_indices(args.romi)[['ROMI1', 'ROMI2']], how='inner')
        aux_variables += ['ROMI1', 'ROMI2']
    if args.vpm:
        input_df = input_df.join(load_vpm_indices(args.vpm)[['VPM1', 'VPM2']], how='inner')
        aux_variables += ['VPM1', 'VPM2']

    # Warn if the inner-join dropped any RMM days within the training/test window
    if aux_variables:
        window_start = pd.Timestamp(TRAIN_START_DATE)
        window_end = pd.Timestamp(TEST_END_DATE)
        rmm_in_window = rmm_df.loc[window_start:window_end]
        joined_in_window = input_df.loc[window_start:window_end]
        dropped = rmm_in_window.index.difference(joined_in_window.index)
        if len(dropped) > 0:
            print(
                f"WARNING: inner-join dropped {len(dropped)} RMM day(s) within "
                f"[{TRAIN_START_DATE}, {TEST_END_DATE}] due to missing {'/'.join(aux_variables)} coverage. "
                f"First few: {list(dropped[:10])}",
                flush=True,
            )

    # Create train/val/test splits
    train_df, mean, std = subset_rmm_data(input_df, TRAIN_START_DATE, VAL_START_DATE, True)
    val_df = subset_rmm_data(input_df, VAL_START_DATE, TEST_START_DATE, False)
    test_df = subset_rmm_data(input_df, TEST_START_DATE, TEST_END_DATE, False)

    # Save normalization statistics (PCs, amplitude and periodic features are not normalized; only year is)
    unnormalized = ['RMM1', 'RMM2'] + aux_variables + ['amplitude', 'doy_sin', 'doy_cos', 'phase_sin', 'phase_cos']
    statistics = {f'{v}_mean': 0.0 for v in unnormalized}
    statistics.update({f'{v}_std': 1.0 for v in unnormalized})
    statistics.update(year_mean=mean['year'], year_std=std['year'])
    np.savez(os.path.join(output_dir, 'statistics.npz'), **statistics)

    columns = ['RMM1', 'RMM2'] + aux_variables + ['phase', 'amplitude', 'doy_sin', 'doy_cos', 'phase_sin', 'phase_cos', 'year']
    for name, df in (('train', train_df), ('val', val_df), ('test', test_df)):
        np.savez(os.path.join(output_dir, f'{name}.npz'),
                 **{c: df[c].values for c in columns},
                 dates=df.index.values)
        print(f'Saved {name}.npz ({len(df)} rows)')

    # Validate against Australian Bureau of Meteorology (BoM) RMM indices
    bom_df = load_rmm_indices(args.bom)
    aligned_df = pd.merge(input_df, bom_df, left_index=True, right_index=True, how='inner', suffixes=('_ours', '_bom')).dropna()

    bcorr = compute_bcor(
        predict_rmm1=aligned_df['RMM1_ours'].values,
        ground_truth_rmm1=aligned_df['RMM1_bom'].values,
        predict_rmm2=aligned_df['RMM2_ours'].values,
        ground_truth_rmm2=aligned_df['RMM2_bom'].values
    )
    print(f"\nBivariate Correlation between computed and BoM RMM indices: {bcorr:.4f}")
    print(f"(Based on {len(aligned_df)} aligned time points)\n")

    correlation_scatter_plot(
        pred_rmm1=aligned_df['RMM1_ours'].values,
        gt_rmm1=aligned_df['RMM1_bom'].values,
        pred_rmm2=aligned_df['RMM2_ours'].values,
        gt_rmm2=aligned_df['RMM2_bom'].values,
        pred_amplitude=aligned_df['amplitude_ours'].values,
        gt_amplitude=aligned_df['amplitude_bom'].values,
        pred_label='Ours',
        gt_label='BoM',
        output_filename=os.path.join(output_dir, "rmm_scatter.png")
    )


if __name__ == "__main__":
    main()
