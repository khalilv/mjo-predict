import pandas as pd
import numpy as np


def save_romi_indices(time, ROMI1, ROMI2, filename, method_str="Kiladis14_ROMI:_OLR_only", MISSING_VAL=9.99999962e35):
    """Save ROMI indices to text file in standard format (same layout as RMM)."""
    df = pd.DataFrame({
        "date": pd.to_datetime(time),
        "ROMI1": ROMI1,
        "ROMI2": ROMI2,
    })
    df = df.set_index("date")

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)

    ROMI1_vals = df["ROMI1"].values
    ROMI2_vals = df["ROMI2"].values
    missing_mask = np.isnan(ROMI1_vals) | np.isnan(ROMI2_vals)

    amplitude = np.sqrt(ROMI1_vals**2 + ROMI2_vals**2)
    angle = (np.arctan2(ROMI2_vals, ROMI1_vals) * 180 / np.pi + 180) % 360
    phase = np.floor(angle / 45) + 1

    ROMI1_vals[missing_mask] = MISSING_VAL
    ROMI2_vals[missing_mask] = MISSING_VAL
    amplitude[missing_mask] = MISSING_VAL
    phase[missing_mask] = 999

    with open(filename, "w") as f:
        for date, r1, r2, ph, amp, is_missing in zip(full_range, ROMI1_vals, ROMI2_vals, phase, amplitude, missing_mask):
            method = "Missing_value" if is_missing else method_str
            if is_missing:
                f.write(
                    f"{date.year:10d}{date.month:11d}{date.day:11d}"
                    f"{r1:15.8E}{r2:15.8E}{int(ph):13d}{amp:15.8E}  {method:<30}\n"
                )
            else:
                f.write(
                    f"{date.year:10d}{date.month:11d}{date.day:11d}"
                    f"{r1:13.7f}{r2:15.7f}{int(ph):13d}{amp:13.7f}  {method:<30}\n"
                )

    print(f"Saved ROMI indices to: {filename}")


def load_romi_indices(filepath, start_year=1979, end_year=2025, MISSING_VAL=9.99999962e35):
    """Load ROMI indices from text file and return as DataFrame with ROMI1/ROMI2 columns."""
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("ROMI") or line.startswith("RMM") or line.startswith("year"):
                continue

            parts = line.split()
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])

            if year < start_year or year > end_year:
                continue

            romi1 = float(parts[3])
            romi2 = float(parts[4])
            phase = int(float(parts[5]))
            amplitude = float(parts[6])

            romi1 = np.nan if romi1 == MISSING_VAL else romi1
            romi2 = np.nan if romi2 == MISSING_VAL else romi2
            amplitude = np.nan if amplitude == MISSING_VAL else amplitude
            phase = np.nan if phase == 999 else phase

            data.append([year, month, day, romi1, romi2, phase, amplitude])

    df = pd.DataFrame(data, columns=["year", "month", "day", "ROMI1", "ROMI2", "phase", "amplitude"])
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date")
    df = df[["ROMI1", "ROMI2", "phase", "amplitude"]].sort_index()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)

    return df
