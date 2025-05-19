import xarray as xr
import numpy as np
import xesmf as xe
from scipy.linalg import lstsq
from matplotlib import pyplot as plt
import os 

def harmonic_terms(fraction):
    return np.array([
        np.ones_like(fraction),
        np.sin(2 * np.pi * fraction),
        np.cos(2 * np.pi * fraction),
        np.sin(4 * np.pi * fraction),
        np.cos(4 * np.pi * fraction),
        np.sin(6 * np.pi * fraction),
        np.cos(6 * np.pi * fraction)
    ])

def harmonic_matrix(time):
    day_of_year = time.dt.dayofyear - 1
    fraction = day_of_year / 365.25
    m = fraction.values.reshape(-1, 1)
    return np.concatenate([harmonic_terms(m_i) for m_i in m], axis=1).T

def compute_harmonics(group, coeffs):
    day_of_year = group['time.dayofyear'].values[0]
    fraction = (day_of_year - 1) / 365.25
    terms = harmonic_terms(fraction)
    return (coeffs * terms).sum(dim='coeff')

def fit_harmonics(ts, X):
    valid = ~np.isnan(ts)
    ts_clean = ts[valid]
    X_clean = X[valid, :]
    if len(ts_clean) < X_clean.shape[1]:
        raise ValueError("Insufficient data points to fit harmonics")
    coeffs, _, _, _ = lstsq(X_clean, ts_clean)
    return coeffs

def compute_seasonal_cycle(ref_data_2p5d):
    ref_X = harmonic_matrix(ref_data_2p5d.time)
    coeffs = xr.apply_ufunc(
        fit_harmonics,
        ref_data_2p5d,
        input_core_dims=[['time']],
        output_core_dims=[['coeff']],
        exclude_dims={'time'},
        kwargs={'X': ref_X},
        vectorize=True,
        output_dtypes=[np.float64],
        output_sizes={'coeff': 7}
    )
    seasonal_cycle = ref_data_2p5d.groupby('time.dayofyear').map(
        lambda group: compute_harmonics(group, coeffs)
    )
    return seasonal_cycle

def detrend_anomalies(anomalies, window = 120):
    rolled = anomalies.rolling(time=window, center=False).construct("window_dim")

    window_mean = rolled.mean("window_dim", skipna=True)

    detrended = anomalies - window_mean
    return detrended.isel(time=slice(window - 1, None))


def latitude_band_average(anomalies):
    anomalies_latitude_band = anomalies.sel(lat=slice(-15,15))

    lat_weights = np.cos(np.deg2rad(anomalies_latitude_band.lat))
    lat_weights_norm = lat_weights / lat_weights.mean()

    return anomalies_latitude_band.weighted(lat_weights_norm).mean(dim='lat')

def get_zonal_average_of_temporal_std(latitude_band_avg): 
    return latitude_band_avg.std(dim='time').mean(dim='lon')

def main():
    reference_start = '1979-09-07'
    reference_end = '2001-12-31'
    olr_file_path = '/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc'
    era5_file_path = '/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr'
    save_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/EOF/reference_period_{reference_start}_to_{reference_end}'
    reverse_eof = True
    negate_eof1 = False
    negate_eof2 = True 

    os.makedirs(save_dir, exist_ok=True)

    raw_olr_ds = xr.open_dataset(olr_file_path).isel(time=slice(249, None)) #remove some corrupted dates
    raw_era5_ds = xr.open_zarr(era5_file_path)

    olr_data = raw_olr_ds['olr'].to_dataset()
    u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850')
    u200_data = raw_era5_ds['u_component_of_wind'].sel(level=200, drop=True).to_dataset(name='u200')

    # regrid to 2.5°
    target_lat = np.arange(-90, 90, 2.5)
    target_lon = np.arange(0, 360, 2.5)
    olr_regridder = xe.Regridder(olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    u850_regridder = xe.Regridder(u850_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    u200_regridder = xe.Regridder(u200_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)

    olr_data_2p5d = olr_regridder(olr_data)
    u850_data_2p5d = u850_regridder(u850_data)
    u200_data_2p5d = u200_regridder(u200_data)

    reference_period = slice(reference_start, reference_end)

    # extract ref period
    ref_olr_data_2p5d = olr_data_2p5d.sel(time=reference_period).load()
    ref_u850_data_2p5d = u850_data_2p5d.sel(time=reference_period).load()
    ref_u200_data_2p5d = u200_data_2p5d.sel(time=reference_period).load()

    # compute seasonal cycle
    olr_seasonal_cycle = compute_seasonal_cycle(ref_olr_data_2p5d)
    u850_seasonal_cycle = compute_seasonal_cycle(ref_u850_data_2p5d)
    u200_seasonal_cycle = compute_seasonal_cycle(ref_u200_data_2p5d)

    seasonal_cycle_ds = xr.Dataset({
        'olr': olr_seasonal_cycle['olr'],
        'u850': u850_seasonal_cycle['u850'], 
        'u200': u200_seasonal_cycle['u200']
    })
    seasonal_cycle_ds.to_netcdf(os.path.join(save_dir, 'seasonal_cycle.nc'))

    olr_seasonal_cycle = olr_seasonal_cycle.sel(dayofyear=ref_olr_data_2p5d.time.dt.dayofyear)
    u850_seasonal_cycle = u850_seasonal_cycle.sel(dayofyear=ref_u850_data_2p5d.time.dt.dayofyear)
    u200_seasonal_cycle = u200_seasonal_cycle.sel(dayofyear=ref_u200_data_2p5d.time.dt.dayofyear)

    # subtract mean and fisrt three harmonics
    olr_anomalies = ref_olr_data_2p5d - olr_seasonal_cycle
    u850_anomalies = ref_u850_data_2p5d - u850_seasonal_cycle
    u200_anomalies = ref_u200_data_2p5d - u200_seasonal_cycle

    detrended_olr_anomalies = detrend_anomalies(olr_anomalies)
    detrended_u850_anomalies = detrend_anomalies(u850_anomalies)
    detrended_u200_anomalies = detrend_anomalies(u200_anomalies)

    detrended_olr_anomalies_latitude_band_avg = latitude_band_average(detrended_olr_anomalies)
    detrended_u850_anomalies_latitude_band_avg = latitude_band_average(detrended_u850_anomalies)
    detrended_u200_anomalies_latitude_band_avg = latitude_band_average(detrended_u200_anomalies)

    olr_normalization_factor = get_zonal_average_of_temporal_std(detrended_olr_anomalies_latitude_band_avg)
    u850_normalization_factor = get_zonal_average_of_temporal_std(detrended_u850_anomalies_latitude_band_avg)
    u200_normalization_factor = get_zonal_average_of_temporal_std(detrended_u200_anomalies_latitude_band_avg)

    detrended_olr_anomalies_latitude_band_avg_norm = detrended_olr_anomalies_latitude_band_avg / olr_normalization_factor
    detrended_u850_anomalies_latitude_band_avg_norm = detrended_u850_anomalies_latitude_band_avg / u850_normalization_factor
    detrended_u200_anomalies_latitude_band_avg_norm = detrended_u200_anomalies_latitude_band_avg / u200_normalization_factor

    detrended_olr_anomalies_latitude_band_avg_norm = detrended_olr_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')
    detrended_u850_anomalies_latitude_band_avg_norm = detrended_u850_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')
    detrended_u200_anomalies_latitude_band_avg_norm = detrended_u200_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')

    olr, u850, u200 = xr.align(
        detrended_olr_anomalies_latitude_band_avg_norm,
        detrended_u850_anomalies_latitude_band_avg_norm,
        detrended_u200_anomalies_latitude_band_avg_norm,
        join='inner'
    )

    X = xr.concat([olr['olr'], u850['u850'], u200['u200']], dim='lon')  # (time, 3 × lon)    

    U, S, Vt = np.linalg.svd(X.values, full_matrices=False)

    explained_variance = (S ** 2) / np.sum(S ** 2)

    # bar plot of explained variance
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(S)), explained_variance * 100)
    plt.title("Explained Variance of Principal Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'explained_variance.png'), dpi=300)
    plt.close()

    print(f"EOF1 explains {explained_variance[0]*100:.2f}%, EOF2 explains {explained_variance[1]*100:.2f}%")

    EOF1, EOF2 = (Vt[1], Vt[0]) if reverse_eof else (Vt[0], Vt[1])
    EOF1 = -1*EOF1 if negate_eof1 else EOF1
    EOF2 = -1*EOF2 if negate_eof2 else EOF2

    EOF1_reshaped = EOF1.reshape(3, -1)
    EOF2_reshaped = EOF2.reshape(3, -1)
    longitudes = olr.lon.values
    labels = ['OLR', 'U850', 'U200']

    # plot and save EOF1
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.plot(longitudes, EOF1_reshaped[i], label=labels[i])
    plt.title("EOF1")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Eigenvector")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eof1.png'), dpi=300)
    plt.close()

    # plot and save EOF2
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.plot(longitudes, EOF2_reshaped[i], label=labels[i])
    plt.title("EOF2")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Eigenvector")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eof2.png'), dpi=300)
    plt.close()

    RMM1 = X.values @ EOF1
    RMM2 = X.values @ EOF2

    RMM1_std = RMM1.std()
    RMM2_std = RMM2.std()

    RMM1_norm = RMM1 / RMM1_std
    RMM2_norm = RMM2 / RMM1_std

    normalization_factor_ds = xr.Dataset({
        'olr': olr_normalization_factor['olr'],
        'u850': u850_normalization_factor['u850'], 
        'u200': u200_normalization_factor['u200'],
        'RMM1_std': RMM1_std,
        'RMM2_std': RMM2_std,
    })
    normalization_factor_ds.to_netcdf(os.path.join(save_dir, 'normalization_factor.nc'))

    EOF_ds = xr.Dataset({
        'EOF1': EOF1,
        'EOF2': EOF2, 
    })
    EOF_ds.to_netcdf(os.path.join(save_dir, 'eof.nc'))

    print('Completed analysis')


if __name__ == "__main__":
    main()
