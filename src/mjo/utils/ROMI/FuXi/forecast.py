import os
import glob
import warnings
import xarray as xr
import numpy as np
import xesmf as xe
from tqdm import tqdm
from mjo.utils.ROMI.eof import fill_missing_days
from mjo.utils.RMM.io import save_rmm_indices
from mjo.utils.RMM.FuXi.utils import walk_to_forecast_dir

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS.*")


def main():
    """Compute ROMI indices from FuXi ensemble-mean OLR forecasts using precomputed per-DOY EOFs.

    Only the ensemble mean is processed; per-member ROMI is skipped. Requires
    mean.py to have been run first (presence of `mean/.done` sentinel).

    Preprocessing matches observation.py exactly:
    1. Remove seasonal cycle (mean + 3 harmonics)
    2. 40-day trailing mean removal
    3. 9-day trailing running mean
    4. Project onto per-DOY EOFs, normalize by sigma
    """
    olr_file_path = '/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc'
    reference_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/ROMI/EOF'
    save_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/ROMI/FuXi'
    forecast_root_dir = '/glade/derecho/scratch/kvirji/DATA/MJO/FuXi/'
    history_days = 49  # 40 (trailing mean) + 9 (running mean) warm-up

    os.makedirs(save_dir, exist_ok=True)

    # Load reference data
    seasonal_cycle_ds = xr.open_dataset(os.path.join(reference_dir, 'seasonal_cycle.nc')).load()
    eof_ds = xr.open_dataset(os.path.join(reference_dir, 'eof.nc')).load()
    sigma = float(eof_ds.attrs['sigma'])
    eofs = eof_ds['EOF'].sel(mode=[1, 2]).values
    npix = eofs.shape[2] * eofs.shape[3]
    eofs_flat = eofs.reshape(366, 2, npix)

    # Load observed OLR for history
    raw_olr_ds = xr.open_dataset(olr_file_path).isel(time=slice(249, None))
    gt_olr_data = raw_olr_ds['olr'].to_dataset().load()
    raw_olr_ds.close()

    # Regrid observed OLR to 2.5 deg, subset to 20S-20N
    target_lat = np.arange(-90, 90, 2.5)
    target_lon = np.arange(0, 360, 2.5)
    gt_regridder = xe.Regridder(gt_olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    gt_olr_2p5 = gt_regridder(gt_olr_data).sel(lat=slice(-20, 20))

    start_dates = sorted(
        d for d in os.listdir(forecast_root_dir)
        if os.path.isdir(os.path.join(forecast_root_dir, d)) and d.isdigit() and len(d) == 8
    )
    forecast_regridder = None

    for start_date in tqdm(start_dates, 'Processing init dates'):
        root = os.path.join(forecast_root_dir, start_date)
        forecast_dir = walk_to_forecast_dir(root)
        mean_dir = os.path.join(forecast_dir, 'mean')

        # Skip dates whose mean isn't finished yet
        if not os.path.exists(os.path.join(mean_dir, '.done')):
            print(f'skip {start_date}: mean not ready', flush=True)
            continue

        out_dir_name = f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}'
        out_dir = os.path.join(save_dir, out_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'mean.txt')
        if os.path.exists(out_path):
            continue

        # Load ensemble-mean forecast (already formatted by mean.py: absolute time, channels split)
        forecast_files = sorted(glob.glob(os.path.join(mean_dir, '*.nc')))
        forecast_ds = xr.open_mfdataset(forecast_files, combine='by_coords', parallel=False).load()
        init_time = np.datetime64(f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}')

        # Regrid forecast OLR to 2.5 deg, subset to 20S-20N
        if not forecast_regridder:
            forecast_regridder = xe.Regridder(forecast_ds, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
        forecast_2p5 = forecast_regridder(forecast_ds)
        forecast_olr = (forecast_2p5['ttr'] * -1).to_dataset(name='olr').sel(lat=slice(-20, 20))
        forecast_ds.close()

        # Observed history before init date
        gt_end = init_time
        gt_start = gt_end - np.timedelta64(history_days - 1, 'D')
        gt_slice = gt_olr_2p5.sel(time=slice(gt_start, gt_end))

        # Concatenate history + forecast
        combined = xr.concat([gt_slice, forecast_olr], dim='time')

        # Remove seasonal cycle
        seasonal = seasonal_cycle_ds['olr'].sel(dayofyear=combined.time.dt.dayofyear)
        anom = (combined - seasonal)['olr']

        # Fill missing days
        anom_np = anom.values.astype(np.float64)
        anom_np = fill_missing_days(anom_np)
        good_t = ~np.isnan(anom_np.reshape(anom_np.shape[0], -1)).any(axis=1)
        anom_np = anom_np[good_t]
        times = anom.time.values[good_t]
        doys = anom.time.dt.dayofyear.values[good_t]
        T = anom_np.shape[0]
        flat = anom_np.reshape(T, npix)

        # 40-day trailing mean removal
        cumsum = np.cumsum(flat, axis=0)
        rm40 = np.zeros_like(flat)
        rm40[39:] = (cumsum[39:] - np.vstack([np.zeros((1, npix)), cumsum[:-40]])) / 40.0
        for i in range(39):
            rm40[i] = cumsum[i] / (i + 1)
        detrended = flat - rm40

        # 9-day trailing running mean (tapered at start)
        cumsum_dt = np.cumsum(detrended, axis=0)
        smoothed = np.zeros_like(detrended)
        smoothed[8:] = (cumsum_dt[8:] - np.vstack([np.zeros((1, npix)), cumsum_dt[:-9]])) / 9.0
        for i in range(8):
            smoothed[i] = cumsum_dt[i] / (i + 1)

        # Keep only forecast days (after init date)
        forecast_mask = times > init_time
        smoothed = smoothed[forecast_mask]
        fc_times = times[forecast_mask]
        fc_doys = doys[forecast_mask]

        # Project onto per-DOY EOFs, normalize by sigma
        eof_idx = np.minimum(fc_doys, 366) - 1
        day_eofs = eofs_flat[eof_idx]
        romi1 = np.einsum('tp,tp->t', smoothed, day_eofs[:, 0, :]) / sigma
        romi2 = np.einsum('tp,tp->t', smoothed, day_eofs[:, 1, :]) / sigma

        save_rmm_indices(
            time=fc_times, RMM1=romi1, RMM2=romi2,
            filename=out_path, method_str='FuXi',
        )


if __name__ == "__main__":
    main()
