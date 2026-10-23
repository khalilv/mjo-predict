import os
import argparse
import numpy as np
import xarray as xr
from scipy.sparse.linalg import svds

from mjo.utils.RMM.eof import compute_seasonal_cycle, load_olr, regrid_to_2p5

# Reference period (starts after the 176-day OLR gap in 1985)
REFERENCE_START = '1985-10-04'
REFERENCE_END = '2012-12-31'
LAT_MIN, LAT_MAX = -20.0, 20.0


def eof_filter(olr_anom_ref, period_min=30.0, period_max=96.0):
    """Keep 30-96 day eastward-propagating (freq * wn < 0 in numpy FFT convention) and zonal-mean signal of a (T, nlat, nlon) array."""
    T, nlat, nlon = olr_anom_ref.shape
    spec = np.fft.fft2(olr_anom_ref, axes=(0, 2))
    freqs = np.fft.fftfreq(T, d=1.0)
    wns = np.fft.fftfreq(nlon) * nlon
    freq_grid, wn_grid = np.meshgrid(freqs, wns, indexing='ij')
    abs_freq = np.abs(freq_grid)
    freq_min, freq_max = 1.0 / period_max, 1.0 / period_min
    mask = ((freq_grid * wn_grid < 0) | (wn_grid == 0)) & \
           (abs_freq >= freq_min) & (abs_freq <= freq_max)
    spec *= mask[:, None, :]
    return np.real(np.fft.ifft2(spec, axes=(0, 2)))


def broadband_filter(olr_anom_ref, period_min=20.0, period_max=96.0):
    """Apply a 20-96 day temporal bandpass (all wavenumbers) to a (T, nlat, nlon) array."""
    T = olr_anom_ref.shape[0]
    spec = np.fft.rfft(olr_anom_ref, axis=0)
    freqs = np.fft.rfftfreq(T, d=1.0)
    band_mask = (freqs >= 1.0 / period_max) & (freqs <= 1.0 / period_min)
    spec[~band_mask] = 0.0
    return np.fft.irfft(spec, n=T, axis=0)


def fill_missing_days(arr):
    """Linearly interpolate NaN timesteps in a (T, nlat, nlon) array in place; unbracketed edge NaNs are left as NaN."""
    T = arr.shape[0]
    flat = arr.reshape(T, -1)
    bad = np.isnan(flat).any(axis=1)
    if not bad.any():
        return arr
    t = np.arange(T)
    good_t = t[~bad]
    interior = bad & (t > good_t[0]) & (t < good_t[-1])
    interior_idx = np.where(interior)[0]
    for p in range(flat.shape[1]):
        flat[interior_idx, p] = np.interp(interior_idx, good_t, flat[good_t, p])
    return flat.reshape(arr.shape)


def build_doy_window_masks(doys, half_width=60, period=365):
    """Return (366, T) masks of samples within +-half_width days of each DOY (circular; DOY 366 reuses DOY 365)."""
    doys = np.asarray(doys)
    target = np.minimum(np.arange(1, 367), 365)
    diff = np.abs(doys[None, :] - target[:, None]) % period
    return np.minimum(diff, period - diff) <= half_width


def fill_and_drop(anom):
    """Fill missing days of an OLR anomaly DataArray and drop unfillable edges; return (values, times, doys)."""
    anom_np = fill_missing_days(anom.values.astype(np.float64))
    good_t = ~np.isnan(anom_np.reshape(anom_np.shape[0], -1)).any(axis=1)
    return anom_np[good_t], anom.time.values[good_t], anom.time.dt.dayofyear.values[good_t]


def romi_smooth(flat):
    """Remove the 40-day trailing mean and apply a 9-day trailing running mean to (T, npix) anomalies (expanding means at the start)."""
    npix = flat.shape[1]

    # 40-day trailing mean removal
    cumsum = np.cumsum(flat, axis=0)
    rm40 = np.zeros_like(flat)
    rm40[39:] = (cumsum[39:] - np.vstack([np.zeros((1, npix)), cumsum[:-40]])) / 40.0
    for i in range(39):
        rm40[i] = cumsum[i] / (i + 1)
    detrended = flat - rm40

    # 9-day trailing running mean
    cumsum_dt = np.cumsum(detrended, axis=0)
    smoothed = np.zeros_like(detrended)
    smoothed[8:] = (cumsum_dt[8:] - np.vstack([np.zeros((1, npix)), cumsum_dt[:-9]])) / 9.0
    for i in range(8):
        smoothed[i] = cumsum_dt[i] / (i + 1)
    return smoothed


def project_romi(smoothed, doys, eofs_flat, sigma):
    """Project smoothed (T, npix) anomalies onto DOY-specific EOFs and normalize by sigma; return (ROMI1, ROMI2)."""
    eof_idx = np.minimum(doys, 366) - 1
    day_eofs = eofs_flat[eof_idx]
    romi1 = np.einsum('tp,tp->t', smoothed, day_eofs[:, 0, :]) / sigma
    romi2 = np.einsum('tp,tp->t', smoothed, day_eofs[:, 1, :]) / sigma
    return romi1, romi2


def load_romi_eofs(eof_dir):
    """Load per-DOY ROMI EOFs from eof_dir/eof.nc; return (eofs_flat (366, 2, npix), sigma)."""
    eof_ds = xr.open_dataset(os.path.join(eof_dir, 'eof.nc')).load()
    sigma = float(eof_ds.attrs['sigma'])
    eofs = eof_ds['EOF'].sel(mode=[1, 2]).values
    return eofs.reshape(366, 2, eofs.shape[2] * eofs.shape[3]), sigma


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute per-DOY ROMI EOFs, sigma and seasonal cycle from NOAA OLR.')
    parser.add_argument('--olr', required=True, help='NOAA interpolated daily OLR netCDF (olr.day.mean.nc)')
    parser.add_argument('--output-dir', required=True, help='Directory to write eof.nc, seasonal_cycle.nc, normalization_factor.nc')
    return parser.parse_args()


def main():
    """Compute per-DOY ROMI EOFs from NOAA OLR."""
    args = parse_args()
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load and regrid OLR to 2.5 deg, subset to 20S-20N
    olr_data = load_olr(args.olr)['olr'].to_dataset()
    olr_2p5 = regrid_to_2p5(olr_data).sel(lat=slice(LAT_MIN, LAT_MAX))
    nlat, nlon = olr_2p5.sizes['lat'], olr_2p5.sizes['lon']
    assert nlat == 17 and nlon == 144

    # Extract reference period
    ref_olr = olr_2p5.sel(time=slice(REFERENCE_START, REFERENCE_END)).load()

    # Remove seasonal cycle (mean + 3 harmonics)
    seasonal_cycle = compute_seasonal_cycle(ref_olr)
    seasonal_cycle.to_netcdf(os.path.join(save_dir, 'seasonal_cycle.nc'))
    ref_seasonal = seasonal_cycle.sel(dayofyear=ref_olr.time.dt.dayofyear)
    ref_olr_anom = (ref_olr - ref_seasonal)['olr']

    # Fill missing days and drop unfillable edges
    ref_olr_anom_np, _, ref_doys = fill_and_drop(ref_olr_anom)
    T_ref = ref_olr_anom_np.shape[0]
    print(f'Reference period: T_ref={T_ref} days')

    # Apply spectral filters
    print('Applying 30-96 day eastward filter...')
    olr_eof_filtered = eof_filter(ref_olr_anom_np)
    print('Applying 20-96 day broadband filter...')
    olr_broadband = broadband_filter(ref_olr_anom_np)

    # Compute per-DOY EOFs using ±60-day sliding window
    doy_masks = build_doy_window_masks(ref_doys)
    npix = nlat * nlon
    eofs = np.zeros((366, 2, npix), dtype=np.float64)
    var_explained = np.zeros((366, 3), dtype=np.float64)
    filtered_flat = olr_eof_filtered.reshape(T_ref, npix)

    print('Computing per-DOY EOFs...')
    for d in range(366):
        samples = filtered_flat[doy_masks[d]]
        samples = samples - samples.mean(axis=0, keepdims=True)
        _, S, Vt = svds(samples, k=3)
        S, Vt = S[::-1], Vt[::-1]  # svds returns ascending singular values
        eofs[d, 0, :] = Vt[0]
        eofs[d, 1, :] = Vt[1]
        total = np.sum(samples ** 2)
        var_explained[d] = (S[:3] ** 2) / total
        if (d + 1) % 30 == 0:
            print(f'  DOY {d + 1}: var1={var_explained[d, 0]:.3f}, '
                  f'var2={var_explained[d, 1]:.3f}, var3={var_explained[d, 2]:.3f}')

    # Enforce sign consistency across DOYs using DOY 1 as reference.
    # EOF1 and EOF2 are signed independently — SVD can flip either one
    # arbitrarily even when adjacent DOYs share most of their samples.
    ref_eof1 = eofs[0, 0, :].copy()
    ref_eof2 = eofs[0, 1, :].copy()
    for d in range(366):
        if np.dot(eofs[d, 0, :], ref_eof1) < 0:
            eofs[d, 0, :] *= -1
        if np.dot(eofs[d, 1, :], ref_eof2) < 0:
            eofs[d, 1, :] *= -1

    # Compute sigma = std(PC1) from broadband-filtered projection
    print('Computing sigma...')
    broadband_flat = olr_broadband.reshape(T_ref, npix)
    eof_idx = np.minimum(ref_doys, 366) - 1
    pc1_series = np.einsum('tp,tp->t', broadband_flat, eofs[eof_idx, 0, :])
    sigma = float(pc1_series.std())
    print(f'sigma = {sigma:.6f}')

    # Save outputs
    eofs_da = xr.DataArray(
        eofs.reshape(366, 2, nlat, nlon),
        dims=('dayofyear', 'mode', 'lat', 'lon'),
        coords={
            'dayofyear': np.arange(1, 367),
            'mode': [1, 2],
            'lat': olr_2p5.lat.values,
            'lon': olr_2p5.lon.values,
        },
        name='EOF',
    )
    var_da = xr.DataArray(
        var_explained,
        dims=('dayofyear', 'mode'),
        coords={'dayofyear': np.arange(1, 367), 'mode': [1, 2, 3]},
        name='var_explained',
    )
    out_ds = xr.Dataset({'EOF': eofs_da, 'var_explained': var_da})
    out_ds.attrs['sigma'] = sigma
    out_ds.attrs['reference_start'] = REFERENCE_START
    out_ds.attrs['reference_end'] = REFERENCE_END
    out_ds.to_netcdf(os.path.join(save_dir, 'eof.nc'))

    sigma_ds = xr.Dataset({'sigma': ((), sigma)})
    sigma_ds.to_netcdf(os.path.join(save_dir, 'normalization_factor.nc'))
    print(f'Saved EOFs and sigma to {save_dir}')


if __name__ == '__main__':
    main()
