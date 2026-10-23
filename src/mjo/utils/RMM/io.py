import pandas as pd
import numpy as np

MISSING_VALUE = 9.99999962e35
MISSING_PHASE = 999


def save_indices(time, pc1, pc2, filename, names=("RMM1", "RMM2"), method_str="", MISSING_VAL=MISSING_VALUE):
    """Save a pair of MJO index PCs to text file in the BoM RMM format (year month day PC1 PC2 phase amplitude method)."""
    df = pd.DataFrame({"date": pd.to_datetime(time), names[0]: pc1, names[1]: pc2})
    df = df.set_index("date")

    # Fill in any missing dates in the range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)

    pc1_vals = df[names[0]].values
    pc2_vals = df[names[1]].values
    missing_mask = np.isnan(pc1_vals) | np.isnan(pc2_vals)

    # Compute amplitude and phase (8 phases, each 45 degrees)
    amplitude = np.sqrt(pc1_vals**2 + pc2_vals**2)
    angle = (np.arctan2(pc2_vals, pc1_vals) * 180 / np.pi + 180) % 360
    phase = np.floor(angle / 45) + 1

    # Fill missing values with standard missing value codes
    pc1_vals[missing_mask] = MISSING_VAL
    pc2_vals[missing_mask] = MISSING_VAL
    amplitude[missing_mask] = MISSING_VAL
    phase[missing_mask] = MISSING_PHASE

    with open(filename, "w") as f:
        for date, r1, r2, ph, amp, is_missing in zip(full_range, pc1_vals, pc2_vals, phase, amplitude, missing_mask):
            if is_missing:
                f.write(
                    f"{date.year:10d}{date.month:11d}{date.day:11d}"
                    f"{r1:15.8E}{r2:15.8E}{int(ph):13d}{amp:15.8E}  {'Missing_value':<30}\n"
                )
            else:
                f.write(
                    f"{date.year:10d}{date.month:11d}{date.day:11d}"
                    f"{r1:13.7f}{r2:15.7f}{int(ph):13d}{amp:13.7f}  {method_str:<30}\n"
                )

    print(f"Saved {names[0][:-1]} indices to: {filename}")


def load_indices(filepath, names=("RMM1", "RMM2"), start_year=1979, end_year=2025, MISSING_VAL=MISSING_VALUE):
    """Load a pair of MJO index PCs from a BoM-format text file into a daily DataFrame (missing values as NaN)."""
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines and empty lines
            if not line or line.startswith(("RMM", "ROMI", "VPM", "year")):
                continue

            parts = line.split()
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if year < start_year or year > end_year:
                continue

            pc1 = float(parts[3])
            pc2 = float(parts[4])
            phase = int(float(parts[5]))
            amplitude = float(parts[6])

            # Convert missing value codes to NaN
            pc1 = np.nan if pc1 == MISSING_VAL else pc1
            pc2 = np.nan if pc2 == MISSING_VAL else pc2
            amplitude = np.nan if amplitude == MISSING_VAL else amplitude
            phase = np.nan if phase == MISSING_PHASE else phase

            data.append([year, month, day, pc1, pc2, phase, amplitude])

    columns = [names[0], names[1], "phase", "amplitude"]
    df = pd.DataFrame(data, columns=["year", "month", "day"] + columns)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date")
    df = df[columns].sort_index()

    # Fill in any missing dates in the range
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)

    return df


def save_rmm_indices(time, RMM1, RMM2, filename, method_str="WH04_method:_OLR_&_ERA5", MISSING_VAL=MISSING_VALUE):
    """Save RMM indices to text file in standard format."""
    save_indices(time, RMM1, RMM2, filename, ("RMM1", "RMM2"), method_str, MISSING_VAL)


def load_rmm_indices(filepath, start_year=1979, end_year=2025, MISSING_VAL=MISSING_VALUE):
    """Load RMM indices from text file and return as DataFrame with RMM1/RMM2/phase/amplitude columns."""
    return load_indices(filepath, ("RMM1", "RMM2"), start_year, end_year, MISSING_VAL)
