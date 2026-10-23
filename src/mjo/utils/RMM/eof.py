import os
import argparse
import xarray as xr
import numpy as np
import xesmf as xe
from scipy.linalg import lstsq
from mjo.utils.plot import eof_explained_variance_plot, eof_pattern_plot

# Reference period for EOFs, seasonal cycle and normalization (following Wheeler & Hendon 2004)
REFERENCE_START = '1979-09-07'
REFERENCE_END = '2001-12-31'

# Common 2.5 degree grid
TARGET_LAT = np.arange(-90, 90, 2.5)
TARGET_LON = np.arange(0, 360, 2.5)


def load_olr(path):
    """Open NOAA interpolated daily OLR, dropping the corrupted initial dates."""
    return xr.open_dataset(path).isel(time=slice(249, None))


def regrid_to_2p5(ds):
    """Bilinearly regrid a dataset to the common 2.5 degree grid."""
    regridder = xe.Regridder(ds, {'lat': TARGET_LAT, 'lon': TARGET_LON}, 'bilinear', periodic=True)
    return regridder(ds)


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


def project_rmm(olr_ds, u850_ds, u200_ds, seasonal_cycle_ds, normalization_factor_ds, EOF_ds):
    """Project daily olr/u850/u200 fields onto precomputed EOFs and return (time, RMM1, RMM2)."""
    # Remove seasonal cycle
    olr_anomalies = olr_ds - seasonal_cycle_ds['olr'].sel(dayofyear=olr_ds.time.dt.dayofyear)
    u850_anomalies = u850_ds - seasonal_cycle_ds['u850'].sel(dayofyear=u850_ds.time.dt.dayofyear)
    u200_anomalies = u200_ds - seasonal_cycle_ds['u200'].sel(dayofyear=u200_ds.time.dt.dayofyear)

    # Detrend using 120-day running mean, average over 15S-15N and normalize by reference factors
    olr = latitude_band_average(detrend_anomalies(olr_anomalies)) / normalization_factor_ds['olr']
    u850 = latitude_band_average(detrend_anomalies(u850_anomalies)) / normalization_factor_ds['u850']
    u200 = latitude_band_average(detrend_anomalies(u200_anomalies)) / normalization_factor_ds['u200']

    # Drop missing values and align timesteps across all variables
    olr, u850, u200 = xr.align(
        olr.dropna(dim='time', how='any'),
        u850.dropna(dim='time', how='any'),
        u200.dropna(dim='time', how='any'),
        join='inner'
    )

    # Combine into single array (time, 3 x lon) and project onto EOFs
    X = xr.concat([olr['olr'], u850['u850'], u200['u200']], dim='lon')
    RMM1 = X.values @ EOF_ds['EOF1'].values
    RMM2 = X.values @ EOF_ds['EOF2'].values

    # Normalize RMM indices using reference period standard deviations
    RMM1 = RMM1 / normalization_factor_ds['RMM1_std'].values
    RMM2 = RMM2 / normalization_factor_ds['RMM2_std'].values
    return olr.time.values, RMM1, RMM2



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute RMM EOFs, seasonal cycle and normalization factors over the reference period.')
    parser.add_argument('--olr', required=True, help='NOAA interpolated daily OLR netCDF (olr.day.mean.nc)')
    parser.add_argument('--era5', required=True, help='ERA5 daily zarr with u_component_of_wind on pressure levels')
    parser.add_argument('--output-dir', required=True, help='Directory to write eof.nc, seasonal_cycle.nc, normalization_factor.nc and plots')
    return parser.parse_args()


def main():
    """Compute EOFs for RMM index calculation using reference period data."""
    args = parse_args()
    save_dir = args.output_dir

    # EOF ordering and sign conventions
    reverse_eof = True  # Swap EOF1 and EOF2
    negate_eof1 = True  # Flip sign of EOF1
    negate_eof2 = False  # Flip sign of EOF2

    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    raw_olr_ds = load_olr(args.olr)
    raw_era5_ds = xr.open_zarr(args.era5)

    # Extract variables
    olr_data = raw_olr_ds['olr'].to_dataset()
    u850_data = raw_era5_ds['u_component_of_wind'].sel(level=850, drop=True).to_dataset(name='u850')
    u200_data = raw_era5_ds['u_component_of_wind'].sel(level=200, drop=True).to_dataset(name='u200')

    # Regrid to 2.5° resolution
    olr_data_2p5d = regrid_to_2p5(olr_data)
    u850_data_2p5d = regrid_to_2p5(u850_data)
    u200_data_2p5d = regrid_to_2p5(u200_data)

    # Extract reference period
    reference_period = slice(REFERENCE_START, REFERENCE_END)
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

    # Standard deviations of the unnormalized PCs, used to normalize RMM1/RMM2
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

    print(f'Saved EOFs to {save_dir}')


if __name__ == "__main__":
    main()
