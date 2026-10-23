from mjo.utils.RMM.io import MISSING_VALUE, save_indices, load_indices


def save_romi_indices(time, ROMI1, ROMI2, filename, method_str="Kiladis14_ROMI:_OLR_only", MISSING_VAL=MISSING_VALUE):
    """Save ROMI indices to text file in standard format (same layout as RMM)."""
    save_indices(time, ROMI1, ROMI2, filename, ("ROMI1", "ROMI2"), method_str, MISSING_VAL)


def load_romi_indices(filepath, start_year=1979, end_year=2025, MISSING_VAL=MISSING_VALUE):
    """Load ROMI indices from text file and return as DataFrame with ROMI1/ROMI2/phase/amplitude columns."""
    return load_indices(filepath, ("ROMI1", "ROMI2"), start_year, end_year, MISSING_VAL)
