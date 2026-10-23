import os
import argparse
import numpy as np
import xarray as xr

from mjo.utils.RMM.eof import load_olr, regrid_to_2p5
from mjo.utils.ROMI.eof import LAT_MIN, LAT_MAX, fill_and_drop, romi_smooth, project_romi, load_romi_eofs
from mjo.utils.ROMI.io import save_romi_indices


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute observed ROMI from NOAA OLR using precomputed per-DOY EOFs.')
    parser.add_argument('--olr', required=True, help='NOAA interpolated daily OLR netCDF (olr.day.mean.nc)')
    parser.add_argument('--eof-dir', required=True, help='Directory with eof.nc and seasonal_cycle.nc (from ROMI/eof.py)')
    parser.add_argument('--output', required=True, help='Output ROMI text file')
    parser.add_argument('--start-date', default='1985-10-04', help='First date to compute (default: %(default)s)')
    parser.add_argument('--end-date', default='2022-12-31', help='Last date to compute (default: %(default)s)')
    return parser.parse_args()


def main():
    """Compute observed ROMI time series from NOAA OLR using precomputed per-DOY EOFs."""
    args = parse_args()
    start_date, end_date = args.start_date, args.end_date
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Load and regrid OLR to 2.5 deg, subset to 20S-20N
    olr_data = load_olr(args.olr)['olr'].to_dataset()
    olr_2p5 = regrid_to_2p5(olr_data).sel(lat=slice(LAT_MIN, LAT_MAX))

    # Load extra lead-in days for the 40-day running mean warm-up
    lead_in = 50
    lead_start = str((np.datetime64(start_date) - np.timedelta64(lead_in, 'D')))
    olr_2p5 = olr_2p5.sel(time=slice(lead_start, end_date)).load()

    # Load precomputed seasonal cycle and EOFs
    seasonal_cycle_ds = xr.open_dataset(os.path.join(args.eof_dir, 'seasonal_cycle.nc'))
    eofs_flat, sigma = load_romi_eofs(args.eof_dir)
    print(f'Loaded EOFs: shape {eofs_flat.shape}, sigma={sigma:.6f}')

    # Remove seasonal cycle (mean + 3 harmonics)
    seasonal = seasonal_cycle_ds['olr'].sel(dayofyear=olr_2p5.time.dt.dayofyear)
    anom = (olr_2p5 - seasonal)['olr']

    # Fill missing days and drop unfillable edges
    anom_np, all_times, all_doys = fill_and_drop(anom)
    T = anom_np.shape[0]
    print(f'After fill+drop: T={T}')

    # 40-day trailing mean removal and 9-day trailing running mean
    smoothed = romi_smooth(anom_np.reshape(T, -1))

    # Trim lead-in days
    date_mask = all_times >= np.datetime64(start_date)
    smoothed = smoothed[date_mask]
    times = all_times[date_mask]
    doys = all_doys[date_mask]
    print(f'After trimming lead-in: T={smoothed.shape[0]}')

    # Project onto DOY-specific EOFs and normalize by sigma
    romi1, romi2 = project_romi(smoothed, doys, eofs_flat, sigma)
    print(f'ROMI1 std: {romi1.std():.3f}, ROMI2 std: {romi2.std():.3f}')

    save_romi_indices(
        time=times, ROMI1=romi1, ROMI2=romi2,
        filename=args.output,
        method_str='Kiladis14_ROMI:_OLR_only',
    )


if __name__ == '__main__':
    main()
