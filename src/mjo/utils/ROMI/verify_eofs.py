"""Spatial correlation of computed ROMI EOFs vs NOAA reference EOFs."""
import os
import argparse
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Correlate computed per-DOY ROMI EOFs with NOAA reference EOFs.')
    parser.add_argument('--eof', required=True, help='Computed eof.nc (from ROMI/eof.py)')
    parser.add_argument('--noaa-dir', required=True, help='NOAA ROMI EOF directory with eof1/eofDDD.txt and eof2/eofDDD.txt')
    parser.add_argument('--plot', required=True, help='Output plot path (png)')
    return parser.parse_args()


def main():
    """Print and plot |Pearson r| between computed and NOAA ROMI EOFs for each day of year."""
    args = parse_args()
    noaa_dir, plot_path = args.noaa_dir, args.plot

    ds = xr.open_dataset(args.eof)
    eofs_flat = ds['EOF'].sel(mode=[1, 2]).values.reshape(366, 2, -1)

    # Compute |Pearson r| for each DOY
    corrs1, corrs2 = np.zeros(366), np.zeros(366)
    for d in range(1, 367):
        noaa1 = np.loadtxt(os.path.join(noaa_dir, f'eof1/eof{d:03d}.txt'))
        noaa2 = np.loadtxt(os.path.join(noaa_dir, f'eof2/eof{d:03d}.txt'))
        corrs1[d - 1] = abs(pearsonr(eofs_flat[d - 1, 0, :], noaa1)[0])
        corrs2[d - 1] = abs(pearsonr(eofs_flat[d - 1, 1, :], noaa2)[0])

    for name, corrs in [('EOF1', corrs1), ('EOF2', corrs2)]:
        print(f'{name}: mean={corrs.mean():.4f}, std={corrs.std():.4f}, '
              f'min={corrs.min():.4f} (DOY {corrs.argmin()+1}), '
              f'max={corrs.max():.4f} (DOY {corrs.argmax()+1})')

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    doys = np.arange(1, 367)
    ax.plot(doys, corrs1, label='EOF1', linewidth=1)
    ax.plot(doys, corrs2, label='EOF2', linewidth=1)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('|Pearson r|')
    ax.set_title('Spatial Correlation: Computed vs NOAA Reference EOFs')
    ax.set_xlim(1, 366)
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Plot saved to {plot_path}')


if __name__ == '__main__':
    main()
