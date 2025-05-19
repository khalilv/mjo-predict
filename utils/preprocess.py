import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from rmm_io import load_rmm_indices

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
        
    train_start_year = 1981
    val_start_year = 2020
    test_start_year = 2021
    test_end_year = 2023
    abm_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/ABM/rmm.74toRealtime.txt"
    input_filepath = "/glade/derecho/scratch/kvirji/DATA/MJO/RMM/reference_period_1979-09-07_to_2001-12-31/rmm.txt"
    output_dir = "/glade/derecho/scratch/kvirji/DATA/preprocessed/MJO/reference_period_1979-09-07_to_2001-12-31"

    os.makedirs(output_dir, exist_ok=True)

    input_df = load_rmm_indices(input_filepath, train_start_year)

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
    abm_df = load_rmm_indices(abm_filepath, train_start_year)
    aligned_df = pd.merge(input_df, abm_df, left_index=True, right_index=True, how='inner', suffixes=('_ours', '_abm'))

    # plot correlation scatter
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for i, component in enumerate(['RMM1', 'RMM2', 'amplitude']):
        valid = aligned_df[[f"{component}_ours", f"{component}_abm"]].dropna()
        x = valid[f"{component}_ours"]
        y = valid[f"{component}_abm"]
        corr = np.corrcoef(x, y)[0, 1]

        axes[i].scatter(x, y, alpha=0.5, s=5)
        axes[i].set_title(f"{component}: r = {corr:.3f}")
        axes[i].set_xlabel("Ours")
        axes[i].set_ylabel("ABM")
        axes[i].plot([x.min(), x.max()], [x.min(), x.max()], 'k--')
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmm_scatter.png"), dpi=300)
    plt.close()

    aligned_df['doy'] = aligned_df.index.dayofyear

    # drop rows with nans
    daily_df = aligned_df.dropna(subset=['RMM1_ours', 'RMM1_abm', 'RMM2_ours', 'RMM2_abm'])

    # group by doy and compute correlation
    corrs_rmm1 = daily_df.groupby('doy').apply(lambda g: np.corrcoef(g['RMM1_ours'], g['RMM1_abm'])[0, 1])
    corrs_rmm2 = daily_df.groupby('doy').apply(lambda g: np.corrcoef(g['RMM2_ours'], g['RMM2_abm'])[0, 1])

    # plot correlation by doy
    plt.figure(figsize=(12, 5))
    plt.plot(corrs_rmm1.index, corrs_rmm1, label='RMM1')
    plt.plot(corrs_rmm2.index, corrs_rmm2, label='RMM2')
    plt.ylim(0, 1.0)
    plt.xlabel("Day of year")
    plt.ylabel("Pearson correlation coeff")
    plt.title(f"ABM vs Ours")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "doy_correlation.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
