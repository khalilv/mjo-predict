import os
import pandas as pd
import numpy as np
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import correlation_scatter_plot
from mjo.utils.analysis.utils import compute_bcor


def subset_romi_data(df, start_date, end_date, compute_statistics):
    """Subset ROMI data by date range and add derived features."""
    assert isinstance(df.index, pd.DatetimeIndex), "DataFrame must have datetime index"

    subset_df = df[(df.index >= start_date) & (df.index < end_date)]

    # Phase angle as sin/cos features
    phase = np.arctan2(subset_df['RMM2'], subset_df['RMM1'])
    subset_df['phase_sin'] = np.sin(phase)
    subset_df['phase_cos'] = np.cos(phase)

    # Day-of-year as sin/cos features for seasonality
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
    """Split ROMI data into train/val/test sets, compute statistics, and validate against NOAA ROMI."""
    train_start_date = '2002-01-01'
    val_start_date = '2018-01-01'
    test_start_date = '2019-01-01'
    test_end_date = '2022-02-12'

    input_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ROMI/observed/romi.txt"
    noaa_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ROMI/observed/noaa_romi.txt"
    output_dir = "/glade/derecho/scratch/kvirji/DATA/MJO/ROMI/preprocessed"

    os.makedirs(output_dir, exist_ok=True)

    input_df = load_rmm_indices(input_filepath)

    train_df, mean, std = subset_romi_data(input_df, train_start_date, val_start_date, True)
    val_df = subset_romi_data(input_df, val_start_date, test_start_date, False)
    test_df = subset_romi_data(input_df, test_start_date, test_end_date, False)

    # Save normalization statistics (currently not normalizing RMM1/RMM2/amplitude)
    np.savez(os.path.join(output_dir, 'statistics.npz'),
            RMM1_mean=0.0,
            RMM2_mean=0.0,
            amplitude_mean=0.0,
            doy_sin_mean=0.0,
            doy_cos_mean=0.0,
            phase_sin_mean=0.0,
            phase_cos_mean=0.0,
            RMM1_std=1.0,
            RMM2_std=1.0,
            amplitude_std=1.0,
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
                phase=df['phase'].values,
                amplitude=df['amplitude'].values,
                doy_sin=df['doy_sin'].values,
                doy_cos=df['doy_cos'].values,
                phase_sin=df['phase_sin'].values,
                phase_cos=df['phase_cos'].values,
                year=df['year'].values,
                dates=df.index.values)
        print(f'Saved {name}.npz ({len(df)} rows)')

    # Validate against NOAA ROMI (analog of BoM-vs-ours RMM check)
    noaa_df = load_rmm_indices(noaa_filepath)
    aligned_df = pd.merge(input_df, noaa_df, left_index=True, right_index=True,
                          how='inner', suffixes=('_ours', '_noaa')).dropna()

    bcorr = compute_bcor(
        predict_rmm1=aligned_df['RMM1_ours'].values,
        ground_truth_rmm1=aligned_df['RMM1_noaa'].values,
        predict_rmm2=aligned_df['RMM2_ours'].values,
        ground_truth_rmm2=aligned_df['RMM2_noaa'].values
    )
    print(f"\nBivariate correlation between computed and NOAA ROMI: {bcorr:.4f}")
    print(f"(Based on {len(aligned_df)} aligned time points)\n")

    correlation_scatter_plot(
        pred_rmm1=aligned_df['RMM1_ours'].values,
        gt_rmm1=aligned_df['RMM1_noaa'].values,
        pred_rmm2=aligned_df['RMM2_ours'].values,
        gt_rmm2=aligned_df['RMM2_noaa'].values,
        pred_amplitude=aligned_df['amplitude_ours'].values,
        gt_amplitude=aligned_df['amplitude_noaa'].values,
        pred_label='Ours',
        gt_label='NOAA',
        output_filename=os.path.join(output_dir, "romi_scatter.png")
    )


if __name__ == "__main__":
    main()
