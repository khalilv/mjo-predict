import os
import pandas as pd
import numpy as np
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.plot import correlation_scatter_plot

def subset_rmm_data(df, start_date, end_date, compute_statistics):
    
    # ensure datetime index
    assert isinstance(df.index, pd.DatetimeIndex), "DataFrame must have datetime index"

    subset_df = df[(df.index >= start_date) & (df.index < end_date)]
    phase = np.arctan2(subset_df['RMM2'],subset_df['RMM1'])
    subset_df['phase_sin'] = np.sin(phase)
    subset_df['phase_cos'] = np.cos(phase)
    
    doy = subset_df.index.day_of_year
    angle = 2 * np.pi * doy / 366
    subset_df['doy_sin'] = np.sin(angle)
    subset_df['doy_cos'] = np.cos(angle)

    subset_df['year'] = subset_df.index.year

    # Compute normalization statistics on train only
    if compute_statistics:
        mean = subset_df[['RMM1', 'RMM2', 'amplitude', 'year']].mean()
        std = subset_df[['RMM1', 'RMM2', 'amplitude', 'year']].std()
        return subset_df, mean, std
    else:
        return subset_df
    
def main():
        
    train_start_date = '2001-01-01'
    val_start_date = '2018-01-01'
    test_start_date = '2019-01-01'
    test_end_date = '2022-02-12'
    input_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/RMM/rmm.txt"
    output_dir = "/glade/derecho/scratch/kvirji/DATA/MJO/U250/preprocessed"
    abm_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ABM/rmm.74toRealtime.txt"

    os.makedirs(output_dir, exist_ok=True)

    input_df = load_rmm_indices(input_filepath)

    train_df, mean, std = subset_rmm_data(input_df, train_start_date, val_start_date, True)
    val_df = subset_rmm_data(input_df, val_start_date, test_start_date, False)
    test_df = subset_rmm_data(input_df, test_start_date, test_end_date, False)

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

    # plot correlation with ABM indices
    abm_df = load_rmm_indices(abm_filepath)
    aligned_df = pd.merge(input_df, abm_df, left_index=True, right_index=True, how='inner', suffixes=('_ours', '_abm')).dropna()
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
