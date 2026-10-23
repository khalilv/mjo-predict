import os
import numpy as np
import xarray as xr
import xesmf as xe

from mjo.utils.ROMI.eof import fill_missing_days
from mjo.utils.ROMI.io import save_romi_indices


def main():
    """Compute observed ROMI time series from NOAA OLR using precomputed per-DOY EOFs."""
    start_date = '1985-10-04'
    end_date = '2022-12-31'

    olr_file_path = '/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc'
    reference_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/ROMI/EOF'
    save_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/ROMI/observed'
    os.makedirs(save_dir, exist_ok=True)

    # Load and regrid OLR to 2.5 deg, subset to 20S-20N
    raw_olr_ds = xr.open_dataset(olr_file_path).isel(time=slice(249, None))
    olr_data = raw_olr_ds['olr'].to_dataset()
    target_lat = np.arange(-90, 90, 2.5)
    target_lon = np.arange(0, 360, 2.5)
    regridder = xe.Regridder(
        olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True
    )
    olr_2p5 = regridder(olr_data).sel(lat=slice(-20, 20))

    # Load extra lead-in days for the 40-day running mean warm-up
    lead_in = 50
    lead_start = str((np.datetime64(start_date) - np.timedelta64(lead_in, 'D')))
    olr_2p5 = olr_2p5.sel(time=slice(lead_start, end_date)).load()

    # Load precomputed seasonal cycle and EOFs
    seasonal_cycle_ds = xr.open_dataset(os.path.join(reference_dir, 'seasonal_cycle.nc'))
    eof_ds = xr.open_dataset(os.path.join(reference_dir, 'eof.nc'))
    sigma = float(eof_ds.attrs['sigma'])
    eofs = eof_ds['EOF'].sel(mode=[1, 2]).values
    npix = eofs.shape[2] * eofs.shape[3]
    eofs_flat = eofs.reshape(366, 2, npix)
    print(f'Loaded EOFs: shape {eofs.shape}, sigma={sigma:.6f}')

    # Remove seasonal cycle (mean + 3 harmonics)
    seasonal = seasonal_cycle_ds['olr'].sel(dayofyear=olr_2p5.time.dt.dayofyear)
    anom = (olr_2p5 - seasonal)['olr']

    # Fill missing days and drop unfillable edges
    anom_np = anom.values.astype(np.float64)
    anom_np = fill_missing_days(anom_np)
    good_t = ~np.isnan(anom_np.reshape(anom_np.shape[0], -1)).any(axis=1)
    anom_np = anom_np[good_t]
    all_times = anom.time.values[good_t]
    all_doys = anom.time.dt.dayofyear.values[good_t]
    T = anom_np.shape[0]
    flat = anom_np.reshape(T, npix)
    print(f'After fill+drop: T={T}')

    # 40-day trailing mean removal
    cumsum = np.cumsum(flat, axis=0)
    rm40 = np.zeros_like(flat)
    rm40[39:] = (cumsum[39:] - np.vstack([np.zeros((1, npix)), cumsum[:-40]])) / 40.0
    for i in range(39):
        rm40[i] = cumsum[i] / (i + 1)
    detrended = flat - rm40

    # 9-day trailing running mean (tapered at start with expanding mean)
    cumsum_dt = np.cumsum(detrended, axis=0)
    smoothed = np.zeros_like(detrended)
    smoothed[8:] = (cumsum_dt[8:] - np.vstack([np.zeros((1, npix)), cumsum_dt[:-9]])) / 9.0
    for i in range(8):
        smoothed[i] = cumsum_dt[i] / (i + 1)

    # Trim lead-in days
    date_mask = all_times >= np.datetime64(start_date)
    smoothed = smoothed[date_mask]
    times = all_times[date_mask]
    doys = all_doys[date_mask]
    print(f'After trimming lead-in: T={smoothed.shape[0]}')

    # Project onto DOY-specific EOFs and normalize by sigma
    eof_idx = np.minimum(doys, 366) - 1
    day_eofs = eofs_flat[eof_idx]
    romi1 = np.einsum('tp,tp->t', smoothed, day_eofs[:, 0, :]) / sigma
    romi2 = np.einsum('tp,tp->t', smoothed, day_eofs[:, 1, :]) / sigma
    print(f'ROMI1 std: {romi1.std():.3f}, ROMI2 std: {romi2.std():.3f}')

    save_romi_indices(
        time=times, ROMI1=romi1, ROMI2=romi2,
        filename=os.path.join(save_dir, 'romi.txt'),
        method_str='Kiladis14_ROMI:_OLR_only',
    )
    print(f'Saved to {os.path.join(save_dir, "romi.txt")}')


if __name__ == '__main__':
    main()
