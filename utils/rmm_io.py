import pandas as pd
import numpy as np

def save_rmm_indices(time, RMM1, RMM2, filename, method_str="WH04_method:_OLR_&_ERA5", MISSING_VAL=9.99999962e35):

    df = pd.DataFrame({
        "date": pd.to_datetime(time),
        "RMM1": RMM1,
        "RMM2": RMM2
    })
    df = df.set_index("date")

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)

    RMM1_vals = df["RMM1"].values
    RMM2_vals = df["RMM2"].values

    missing_mask = np.isnan(RMM1_vals) | np.isnan(RMM2_vals)

    amplitude = np.sqrt(RMM1_vals**2 + RMM2_vals**2)
    angle = (np.arctan2(RMM2_vals, RMM1_vals) * 180 / np.pi + 180) % 360
    phase = np.floor(angle / 45) + 1
    
    # fill missing
    RMM1_vals[missing_mask] = MISSING_VAL
    RMM2_vals[missing_mask] = MISSING_VAL
    amplitude[missing_mask] = MISSING_VAL
    phase[missing_mask] = 999

    with open(filename, "w") as f:
        for date, r1, r2, ph, amp, is_missing in zip(full_range, RMM1_vals, RMM2_vals, phase, amplitude, missing_mask):
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

    print(f"Saved RMM indices to: {filename}")

def load_rmm_indices(filepath, start_year=1979, end_year=2025, MISSING_VAL=9.99999962e35):
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("RMM") or line.startswith("year"):
                continue

            parts = line.split()
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])

            if year < start_year or year > end_year:
                continue

            rmm1 = float(parts[3])
            rmm2 = float(parts[4])
            phase = int(float(parts[5]))
            amplitude = float(parts[6])

            rmm1 = np.nan if rmm1 == MISSING_VAL else rmm1
            rmm2 = np.nan if rmm2 == MISSING_VAL else rmm2
            amplitude = np.nan if amplitude == MISSING_VAL else amplitude
            phase = np.nan if phase == 999 else phase 

            data.append([year, month, day, rmm1, rmm2, phase, amplitude])

    df = pd.DataFrame(data, columns=["year", "month", "day", "RMM1", "RMM2", "phase", "amplitude"])
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date")
    df = df[["RMM1", "RMM2", "phase", "amplitude"]].sort_index()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)

    return df

