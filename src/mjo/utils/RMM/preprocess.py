import os
import pandas as pd
import numpy as np
from mjo.utils.RMM.io import load_rmm_indices
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
    output_dir = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/preprocessed"
    abm_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ABM/rmm.74toRealtime.txt"

    os.makedirs(output_dir, exist_ok=True)

    # Load RMM indices
    input_df = load_rmm_indices(input_filepath)

    # Create train/val/test splits
    train_df, mean, std = subset_rmm_data(input_df, train_start_date, val_start_date, True)
    val_df = subset_rmm_data(input_df, val_start_date, test_start_date, False)
    test_df = subset_rmm_data(input_df, test_start_date, test_end_date, False)

    # Save normalization statistics (currently not normalizing RMM1/RMM2/amplitude)
    np.savez(os.path.join(output_dir, 'statistics.npz'),
            RMM1_mean=0.0, #mean['RMM1'],
            RMM2_mean=0.0, #mean['RMM2'],
            amplitude_mean=0.0, #mean['amplitude'],
            doy_sin_mean=0.0,
            doy_cos_mean=0.0,
            phase_sin_mean=0.0,
            phase_cos_mean=0.0,
            RMM1_std=1.0, #std['RMM1'],
            RMM2_std=1.0, #std['RMM2'],
            amplitude_std=1.0, #std['amplitude'],
            doy_sin_std=1.0,
            doy_cos_std=1.0,
            phase_sin_std=1.0,
            phase_cos_std=1.0,
            year_mean=mean['year'], 
            year_std=std['year'])

    np.savez(os.path.join(output_dir, 'train.npz'),
            RMM1=train_df['RMM1'].values,
            RMM2=train_df['RMM2'].values,
            phase=train_df['phase'].values,
            amplitude=train_df['amplitude'].values,
            doy_sin=train_df['doy_sin'].values,
            doy_cos=train_df['doy_cos'].values,
            phase_sin=train_df['phase_sin'].values,
            phase_cos=train_df['phase_cos'].values,
            year=train_df['year'].values,
            dates=train_df.index.values)

    np.savez(os.path.join(output_dir, 'val.npz'),
            RMM1=val_df['RMM1'].values,
            RMM2=val_df['RMM2'].values,
            phase=val_df['phase'].values,
            amplitude=val_df['amplitude'].values,
            doy_sin=val_df['doy_sin'].values,
            doy_cos=val_df['doy_cos'].values,
            phase_sin=val_df['phase_sin'].values,
            phase_cos=val_df['phase_cos'].values,
            year=val_df['year'].values,
            dates=val_df.index.values)

    np.savez(os.path.join(output_dir, 'test.npz'),
            RMM1=test_df['RMM1'].values,
            RMM2=test_df['RMM2'].values,
            phase=test_df['phase'].values,
            amplitude=test_df['amplitude'].values,
            doy_sin=test_df['doy_sin'].values,
            doy_cos=test_df['doy_cos'].values,
            phase_sin=test_df['phase_sin'].values,
            phase_cos=test_df['phase_cos'].values,
            year=test_df['year'].values,
            dates=test_df.index.values)

    # Validate against Australian Bureau of Meteorology (ABM/BOM) RMM indices
    abm_df = load_rmm_indices(abm_filepath)
    aligned_df = pd.merge(input_df, abm_df, left_index=True, right_index=True, how='inner', suffixes=('_ours', '_abm')).dropna()

    # Compute bivariate correlation between our computed indices and BOM indices
    bcorr = compute_bcor(
        predict_rmm1=aligned_df['RMM1_ours'].values,
        ground_truth_rmm1=aligned_df['RMM1_abm'].values,
        predict_rmm2=aligned_df['RMM2_ours'].values,
        ground_truth_rmm2=aligned_df['RMM2_abm'].values
    )
    print(f"\nBivariate Correlation between computed and BOM RMM indices: {bcorr:.4f}")
    print(f"(Based on {len(aligned_df)} aligned time points)\n")

    # Generate scatter plot comparing our indices with BOM indices
    correlation_scatter_plot(
        pred_rmm1=aligned_df['RMM1_ours'].values,
        gt_rmm1=aligned_df['RMM1_abm'].values,
        pred_rmm2=aligned_df['RMM2_ours'].values,
        gt_rmm2=aligned_df['RMM2_abm'].values,
        pred_amplitude=aligned_df['amplitude_ours'].values,
        gt_amplitude=aligned_df['amplitude_abm'].values,
        pred_label='Ours',
        gt_label='ABM',
        output_filename=os.path.join(output_dir, "rmm_scatter.png")
    )


if __name__ == "__main__":
    main()
