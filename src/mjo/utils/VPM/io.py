from mjo.utils.RMM.io import MISSING_VALUE, save_indices, load_indices


def save_vpm_indices(time, VPM1, VPM2, filename, method_str="NOAA_VPM", MISSING_VAL=MISSING_VALUE):
    """Save VPM indices to text file in standard format (same layout as RMM)."""
    save_indices(time, VPM1, VPM2, filename, ("VPM1", "VPM2"), method_str, MISSING_VAL)


def load_vpm_indices(filepath, start_year=1979, end_year=2025, MISSING_VAL=MISSING_VALUE):
    """Load VPM indices from text file and return as DataFrame with VPM1/VPM2/phase/amplitude columns."""
    return load_indices(filepath, ("VPM1", "VPM2"), start_year, end_year, MISSING_VAL)
