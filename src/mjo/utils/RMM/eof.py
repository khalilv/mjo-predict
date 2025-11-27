import os
import xarray as xr
import numpy as np
import xesmf as xe
from scipy.linalg import lstsq
from mjo.utils.plot import eof_explained_variance_plot, eof_pattern_plot


def harmonic_terms(fraction):
    """Compute first three harmonic terms for seasonal cycle fitting."""
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
    """Create design matrix for harmonic regression."""
    day_of_year = time.dt.dayofyear - 1
    fraction = day_of_year / 365.25
    m = fraction.values.reshape(-1, 1)
    return np.concatenate([harmonic_terms(m_i) for m_i in m], axis=1).T


def compute_harmonics(group, coeffs):
    """Compute seasonal cycle from harmonic coefficients for a specific day of year."""
    day_of_year = group['time.dayofyear'].values[0]
    fraction = (day_of_year - 1) / 365.25
    terms = harmonic_terms(fraction)
    return (coeffs * terms).sum(dim='coeff')


def fit_harmonics(ts, X):
    """Fit harmonic regression to time series using least squares."""
    valid = ~np.isnan(ts)
    ts_clean = ts[valid]
    X_clean = X[valid, :]
    if len(ts_clean) < X_clean.shape[1]:
        raise ValueError("Insufficient data points to fit harmonics")
    coeffs, _, _, _ = lstsq(X_clean, ts_clean)
    return coeffs


def compute_seasonal_cycle(ref_data_2p5d):
    """Compute seasonal cycle using first three harmonics."""
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


def detrend_anomalies(anomalies, window=120):
    """Remove low-frequency variability using running mean."""
    rolled = anomalies.rolling(time=window, center=False).construct("window_dim")
    window_mean = rolled.mean("window_dim", skipna=True)
    detrended = anomalies - window_mean
    # Remove first window-1 points where running mean isn't fully defined
    return detrended.isel(time=slice(window - 1, None))


def latitude_band_average(anomalies):
    """Compute cosine-weighted average over 15°S-15°N."""
    anomalies_latitude_band = anomalies.sel(lat=slice(-15, 15))
    lat_weights = np.cos(np.deg2rad(anomalies_latitude_band.lat))
    lat_weights_norm = lat_weights / lat_weights.mean()
    return anomalies_latitude_band.weighted(lat_weights_norm).mean(dim='lat')


def get_zonal_average_of_temporal_std(latitude_band_avg):
    """Compute zonal mean of temporal standard deviation."""
    return latitude_band_avg.std(dim='time').mean(dim='lon')

def main():
    """Compute EOFs for RMM index calculation using reference period data."""
    # Reference period for EOF calculation (following Wheeler & Hendon 2004)
    reference_start = '1979-09-07'
    reference_end = '2001-12-31'

    # File paths
    olr_file_path = '/glade/derecho/scratch/kvirji/DATA/NOAA/OLR/PSL_interpolated/olr.day.mean.nc'
    era5_file_path = '/glade/derecho/scratch/kvirji/DATA/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr'
    save_dir = f'/glade/derecho/scratch/kvirji/DATA/MJO/U250/EOF'

    # EOF ordering and sign conventions
    reverse_eof = True  # Swap EOF1 and EOF2 if needed
    negate_eof1 = True  # Flip sign of EOF1 if needed
    negate_eof2 = False  # Flip sign of EOF2 if needed

    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    raw_olr_ds = xr.open_dataset(olr_file_path).isel(time=slice(249, None))  # Remove corrupted dates
    raw_era5_ds = xr.open_zarr(era5_file_path)

    # Extract variables
    olr_data = raw_olr_ds['olr'].to_dataset()
    u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850')
    u200_data = raw_era5_ds['u_component_of_wind'].sel(level=200, drop=True).to_dataset(name='u200')

    # Regrid to 2.5° resolution
    target_lat = np.arange(-90, 90, 2.5)
    target_lon = np.arange(0, 360, 2.5)
    olr_regridder = xe.Regridder(olr_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    u850_regridder = xe.Regridder(u850_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)
    u200_regridder = xe.Regridder(u200_data, {'lat': target_lat, 'lon': target_lon}, 'bilinear', periodic=True)

    olr_data_2p5d = olr_regridder(olr_data)
    u850_data_2p5d = u850_regridder(u850_data)
    u200_data_2p5d = u200_regridder(u200_data)

    # Extract reference period
    reference_period = slice(reference_start, reference_end)
    ref_olr_data_2p5d = olr_data_2p5d.sel(time=reference_period).load()
    ref_u850_data_2p5d = u850_data_2p5d.sel(time=reference_period).load()
    ref_u200_data_2p5d = u200_data_2p5d.sel(time=reference_period).load()

    # Compute seasonal cycle using first three harmonics
    olr_seasonal_cycle = compute_seasonal_cycle(ref_olr_data_2p5d)
    u850_seasonal_cycle = compute_seasonal_cycle(ref_u850_data_2p5d)
    u200_seasonal_cycle = compute_seasonal_cycle(ref_u200_data_2p5d)

    seasonal_cycle_ds = xr.Dataset({
        'olr': olr_seasonal_cycle['olr'],
        'u850': u850_seasonal_cycle['u850'], 
        'u200': u200_seasonal_cycle['u200']
    })
    seasonal_cycle_ds.to_netcdf(os.path.join(save_dir, 'seasonal_cycle.nc'))

    # Align seasonal cycle with actual dates
    olr_seasonal_cycle = olr_seasonal_cycle.sel(dayofyear=ref_olr_data_2p5d.time.dt.dayofyear)
    u850_seasonal_cycle = u850_seasonal_cycle.sel(dayofyear=ref_u850_data_2p5d.time.dt.dayofyear)
    u200_seasonal_cycle = u200_seasonal_cycle.sel(dayofyear=ref_u200_data_2p5d.time.dt.dayofyear)

    # Remove seasonal cycle
    olr_anomalies = ref_olr_data_2p5d - olr_seasonal_cycle
    u850_anomalies = ref_u850_data_2p5d - u850_seasonal_cycle
    u200_anomalies = ref_u200_data_2p5d - u200_seasonal_cycle

    # Detrend using 120-day running mean
    detrended_olr_anomalies = detrend_anomalies(olr_anomalies)
    detrended_u850_anomalies = detrend_anomalies(u850_anomalies)
    detrended_u200_anomalies = detrend_anomalies(u200_anomalies)

    # Average over 15°S-15°N latitude band
    detrended_olr_anomalies_latitude_band_avg = latitude_band_average(detrended_olr_anomalies)
    detrended_u850_anomalies_latitude_band_avg = latitude_band_average(detrended_u850_anomalies)
    detrended_u200_anomalies_latitude_band_avg = latitude_band_average(detrended_u200_anomalies)

    # Compute normalization factors (zonal mean of temporal std)
    olr_normalization_factor = get_zonal_average_of_temporal_std(detrended_olr_anomalies_latitude_band_avg)
    u850_normalization_factor = get_zonal_average_of_temporal_std(detrended_u850_anomalies_latitude_band_avg)
    u200_normalization_factor = get_zonal_average_of_temporal_std(detrended_u200_anomalies_latitude_band_avg)

    # Normalize by standard deviation
    detrended_olr_anomalies_latitude_band_avg_norm = detrended_olr_anomalies_latitude_band_avg / olr_normalization_factor
    detrended_u850_anomalies_latitude_band_avg_norm = detrended_u850_anomalies_latitude_band_avg / u850_normalization_factor
    detrended_u200_anomalies_latitude_band_avg_norm = detrended_u200_anomalies_latitude_band_avg / u200_normalization_factor

    # Drop missing values
    detrended_olr_anomalies_latitude_band_avg_norm = detrended_olr_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')
    detrended_u850_anomalies_latitude_band_avg_norm = detrended_u850_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')
    detrended_u200_anomalies_latitude_band_avg_norm = detrended_u200_anomalies_latitude_band_avg_norm.dropna(dim='time', how='any')

    # Align timesteps across all variables
    olr, u850, u200 = xr.align(
        detrended_olr_anomalies_latitude_band_avg_norm,
        detrended_u850_anomalies_latitude_band_avg_norm,
        detrended_u200_anomalies_latitude_band_avg_norm,
        join='inner'
    )

    # Combine data into single array (time, 3 × lon)
    X = xr.concat([olr['olr'], u850['u850'], u200['u200']], dim='lon')

    # Compute SVD to get EOFs
    _, S, Vt = np.linalg.svd(X.values, full_matrices=False)
    explained_variance = (S ** 2) / np.sum(S ** 2)

    # Plot explained variance
    eof_explained_variance_plot(
        explained_variance=explained_variance,
        output_filename=os.path.join(save_dir, 'explained_variance.png')
    )

    print(f"EOF1 explains {explained_variance[0]*100:.2f}%, EOF2 explains {explained_variance[1]*100:.2f}%")

    # Apply EOF ordering and sign conventions
    EOF1, EOF2 = (Vt[1], Vt[0]) if reverse_eof else (Vt[0], Vt[1])
    EOF1 = -1*EOF1 if negate_eof1 else EOF1
    EOF2 = -1*EOF2 if negate_eof2 else EOF2

    EOF1_reshaped = EOF1.reshape(3, -1)
    EOF2_reshaped = EOF2.reshape(3, -1)
    longitudes = olr.lon.values
    variable_labels = ['OLR', 'U850', 'U200']

    # Plot EOF patterns
    eof_pattern_plot(
        eof_vector=EOF1_reshaped,
        longitudes=longitudes,
        variable_labels=variable_labels,
        eof_number=1,
        output_filename=os.path.join(save_dir, 'eof1.png')
    )

    eof_pattern_plot(
        eof_vector=EOF2_reshaped,
        longitudes=longitudes,
        variable_labels=variable_labels,
        eof_number=2,
        output_filename=os.path.join(save_dir, 'eof2.png')
    )

    RMM1 = X.values @ EOF1
    RMM2 = X.values @ EOF2

    RMM1_std = RMM1.std()
    RMM2_std = RMM2.std()

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
