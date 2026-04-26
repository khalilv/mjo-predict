import os
import pandas as pd
import numpy as np
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.ROMI.io import load_romi_indices
from mjo.utils.plot import correlation_scatter_plot
from mjo.utils.analysis.utils import compute_bcor


def subset_rmm_data(df, start_date, end_date, compute_statistics):
    """Subset RMM data by date range and add derived features."""
    # Ensure datetime index
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
    
def main():
    """Split RMM data into train/val/test sets, compute statistics, and validate against BOM indices."""
    # Date ranges for splits
    train_start_date = '2002-01-01'
    val_start_date = '2018-01-01'
    test_start_date = '2019-01-01'
    test_end_date = '2022-02-12'

    # File paths
    input_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
    romi_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ROMI/observed/romi.txt"
    output_dir = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/preprocessed"
    bom_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/BoM/rmm.74toRealtime.txt"

    os.makedirs(output_dir, exist_ok=True)

    # Load RMM indices and merge observed ROMI as auxiliary historical context.
    # Inner-join on date so any days missing in either source are dropped.
    rmm_df = load_rmm_indices(input_filepath)
    romi_df = load_romi_indices(romi_filepath)[['ROMI1', 'ROMI2']]
    input_df = rmm_df.join(romi_df, how='inner')

    # Warn if the inner-join dropped any RMM days within the training/test window.
    # The window is what we actually use; gaps outside it are irrelevant.
    window_start = pd.Timestamp(train_start_date)
    window_end = pd.Timestamp(test_end_date)
    rmm_in_window = rmm_df.loc[window_start:window_end]
    joined_in_window = input_df.loc[window_start:window_end]
    dropped = rmm_in_window.index.difference(joined_in_window.index)
    if len(dropped) > 0:
        print(
            f"WARNING: inner-join dropped {len(dropped)} RMM day(s) within "
            f"[{train_start_date}, {test_end_date}] due to missing ROMI coverage. "
            f"First few: {list(dropped[:10])}",
            flush=True,
        )

    # Create train/val/test splits
    train_df, mean, std = subset_rmm_data(input_df, train_start_date, val_start_date, True)
    val_df = subset_rmm_data(input_df, val_start_date, test_start_date, False)
    test_df = subset_rmm_data(input_df, test_start_date, test_end_date, False)

    # Save normalization statistics (currently not normalizing RMM1/RMM2/ROMI1/ROMI2/amplitude)
    np.savez(os.path.join(output_dir, 'statistics.npz'),
            RMM1_mean=0.0, #mean['RMM1'],
            RMM2_mean=0.0, #mean['RMM2'],
            ROMI1_mean=0.0,
            ROMI2_mean=0.0,
            amplitude_mean=0.0, #mean['amplitude'],
            doy_sin_mean=0.0,
            doy_cos_mean=0.0,
            phase_sin_mean=0.0,
            phase_cos_mean=0.0,
            RMM1_std=1.0, #std['RMM1'],
            RMM2_std=1.0, #std['RMM2'],
            ROMI1_std=1.0,
            ROMI2_std=1.0,
            amplitude_std=1.0, #std['amplitude'],
            doy_sin_std=1.0,
            doy_cos_std=1.0,
            phase_sin_std=1.0,
            phase_cos_std=1.0,
            year_mean=mean['year'],
            year_std=std['year'])

    for name, df in (('train', train_df), ('val', val_df), ('test', test_df)):
        np.savez(os.path.join(output_dir, f'{name}.npz'),
                RMM1=df['RMM1'].values,
                RMM2=df['RMM2'].values,
                ROMI1=df['ROMI1'].values,
                ROMI2=df['ROMI2'].values,
                phase=df['phase'].values,
                amplitude=df['amplitude'].values,
                doy_sin=df['doy_sin'].values,
                doy_cos=df['doy_cos'].values,
                phase_sin=df['phase_sin'].values,
                phase_cos=df['phase_cos'].values,
                year=df['year'].values,
                dates=df.index.values)
        print(f'Saved {name}.npz ({len(df)} rows)')

    # Validate against Australian Bureau of Meteorology (BoM) RMM indices
    bom_df = load_rmm_indices(bom_filepath)
    aligned_df = pd.merge(input_df, bom_df, left_index=True, right_index=True, how='inner', suffixes=('_ours', '_bom')).dropna()

    # Compute bivariate correlation between our computed indices and BoM indices
    bcorr = compute_bcor(
        predict_rmm1=aligned_df['RMM1_ours'].values,
        ground_truth_rmm1=aligned_df['RMM1_bom'].values,
        predict_rmm2=aligned_df['RMM2_ours'].values,
        ground_truth_rmm2=aligned_df['RMM2_bom'].values
    )
    print(f"\nBivariate Correlation between computed and BoM RMM indices: {bcorr:.4f}")
    print(f"(Based on {len(aligned_df)} aligned time points)\n")

    # Generate scatter plot comparing our indices with BoM indices
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
