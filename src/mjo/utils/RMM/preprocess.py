import os
import pandas as pd
import numpy as np
from src.mjo.utils.RMM.io import load_rmm_indices
from src.mjo.utils.plot import correlation_scatter_plot

def subset_rmm_data(df, start_year, end_year, compute_statistics):
    
    # ensure datetime index
    assert isinstance(df.index, pd.DatetimeIndex), "DataFrame must have datetime index"

    subset_df = df[(df.index.year >= start_year) & (df.index.year < end_year)]

    # Compute normalization statistics on train only
    if compute_statistics:
        mean = subset_df[['RMM1', 'RMM2']].mean()
        std = subset_df[['RMM1', 'RMM2']].std()
        return subset_df, mean, std
    else:
        return subset_df

def main():
        
    train_start_year = 1979
    val_start_year = 2020
    test_start_year = 2021
    test_end_year = 2023
    input_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/RMM/reference_period_1979-09-07_to_2001-12-31/rmm.txt"
    output_dir = "/glade/derecho/scratch/kvirji/DATA/preprocessed/MJO/reference_period_1979-09-07_to_2001-12-31"
    abm_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ABM/rmm.74toRealtime.txt"

    os.makedirs(output_dir, exist_ok=True)

    input_df = load_rmm_indices(input_filepath, train_start_year, test_end_year)

    train_df, mean, std = subset_rmm_data(input_df, train_start_year, val_start_year, True)
    val_df = subset_rmm_data(input_df, val_start_year, test_start_year, False)
    test_df = subset_rmm_data(input_df, test_start_year, test_end_year, False)

    np.savez(os.path.join(output_dir, 'statistics.npz'),
            mean=mean.values,
            std=std.values)

    np.savez(os.path.join(output_dir, 'train.npz'),
            RMM1=train_df['RMM1'].values,
            RMM2=train_df['RMM2'].values,
            phase=train_df['phase'].values,
            amplitude=train_df['amplitude'].values,
            dates=train_df.index.values)

    np.savez(os.path.join(output_dir, 'val.npz'),
            RMM1=val_df['RMM1'].values,
            RMM2=val_df['RMM2'].values,
            phase=val_df['phase'].values,
            amplitude=val_df['amplitude'].values,
            dates=val_df.index.values)

    np.savez(os.path.join(output_dir, 'test.npz'),
            RMM1=test_df['RMM1'].values,
            RMM2=test_df['RMM2'].values,
            phase=test_df['phase'].values,
            amplitude=test_df['amplitude'].values,
            dates=test_df.index.values)

    # plot correlation with ABM indices
    abm_df = load_rmm_indices(abm_filepath, train_start_year, test_end_year)
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
