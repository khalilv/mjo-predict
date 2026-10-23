# MJO Forecasting with Deep Learning

Code and data for [Virji, A & Velicogna (2026), *On the importance of historical context and bias correction in deep learning MJO prediction models*, npj Climate and Atmospheric Science 9, 193](https://doi.org/10.1038/s41612-026-01459-9).

We forecast the Madden-Julian Oscillation (MJO) with time-series models (TFT, TSMixer) that learn from historical MJO activity, and bias-correct FuXi-S2S forecasts with mean-bias, multivariate linear regression (MVLR) and LSTM correction models.

**Explore the results:** [project page](https://geruoa.github.io/fusion-outlook/papers/mjo-prediction.html) · [interactive dashboard](https://geruoa.github.io/fusion-outlook/dashboards/mjo.html) (skill by lead time, bias, seasonality and a browser for all 2,080 FuXi-S2S forecasts against observations).

## Installation

```bash
git clone https://github.com/khalilv/mjo-predict.git
cd mjo-predict

conda env create -f environment.yml
conda activate mjo
pip install -e .

# unzip the shipped data (creates data/FuXi and data/corrected)
cd data && unzip FuXi_2002-2011.zip && unzip FuXi_2012-2021.zip && unzip corrected_2019-2021.zip && cd ..
```

Run all commands from the repository root; config paths are relative to it.

## Data

| File | Contents |
|---|---|
| `data/FuXi_2002-2011.zip`, `data/FuXi_2012-2021.zip` | FuXi-S2S forecast RMM indices for all 2,080 initializations (2002–2021): `FuXi/YYYY-MM-DD/{00..50,mean}.txt`, 51 members and the ensemble mean, 42 lead days |
| `data/corrected_2019-2021.zip` | Bias-corrected FuXi-S2S forecasts for the 312 test initializations (2019–2021): `corrected/{mean_bias,mvlr,lstm}/YYYY-MM-DD.txt`, and `corrected/BoM/...` for the models trained and evaluated on BoM RMM |
| `data/observed/rmm.txt` | Observed RMM indices (1980–2022) computed from NOAA OLR and ERA5 winds, used as ground truth |
| `data/observed/bom_rmm.74toRealtime.txt` | RMM indices from the [Bureau of Meteorology](https://www.bom.gov.au/climate/mjo/) |

All index files share one text format, one row per day:

```
Year  Month  Day  RMM1  RMM2  Phase  Amplitude  Method
```

Load them with `mjo.utils.RMM.io.load_rmm_indices`.

Not included: the raw inputs used to compute indices (NOAA OLR, ERA5 from [WeatherBench 2](https://console.cloud.google.com/storage/browser/weatherbench2), raw [FuXi-S2S forecasts](https://huggingface.co/datasets/FudanFuXi/FuXi-S2S)), the RMM EOFs, NOAA ROMI and VPM indices (download from [NOAA PSL](https://psl.noaa.gov/mjo/)) and trained model checkpoints.

## Reproducing the paper

| Figure | Command |
|---|---|
| Fig. 2, 4 (BCOR/BMSE, by amplitude) | `python -m mjo.utils.analysis.lead_time_errors --config configs/analysis/lead_time_errors.yaml` |
| Fig. 3 (amplitude/phase bias) | `python -m mjo.utils.analysis.lead_time_bias --config configs/analysis/lead_time_bias.yaml` |
| Fig. 5 (vs BoM) | `python -m mjo.utils.analysis.lead_time_errors --config configs/analysis/lead_time_errors_BoM.yaml` |
| Fig. 7 (phase-space composite) | `python -m mjo.utils.analysis.phase_space_composite --config configs/analysis/phase_space_composite.yaml` |
| Fig. 8 (seasonality) | `python -m mjo.utils.analysis.monthly_errors --config configs/analysis/monthly_errors.yaml` |
| Fig. 1 (history only) | train `configs/{TFT,TSMixer}/history_only/*`, then `python -m mjo.utils.analysis.lead_time_heatmaps --config configs/analysis/history_only_lead_time_heatmaps.yaml` |
| Fig. 6 (history + FuXi-S2S) | train `configs/{TFT,TSMixer}/fusion/*`, then `... --config configs/analysis/fusion_lead_time_heatmaps.yaml` |
| Supp. Fig. S1 (with ROMI, VPM) | train `configs/{TFT,TSMixer}/history_only_noaa/*`, then `... --config configs/analysis/history_only_lead_time_heatmaps_with_noaa_indices.yaml` |

Figs. 2–5, 7 and 8 need only the shipped data; `bash scripts/make_figures.sh` runs them all and writes to `plots/`. The bootstrap significance (1,000 samples) makes Figs. 2–5 slow.

The no-history (∅) columns of Figs. 1 and 6 came from runs with the model inputs zeroed out, which the configs do not reproduce.

### Training

```bash
# model inputs: observed RMM (Figs. 1, 6) and FuXi-S2S forecasts (Fig. 6, LSTM correction)
python src/mjo/utils/RMM/preprocess.py                 # -> data/preprocessed
python src/mjo/utils/RMM/FuXi/preprocess.py            # -> data/preprocessed/FuXi
# Supp. Fig. S1 inputs, with NOAA ROMI and VPM as extra history
python src/mjo/utils/RMM/preprocess.py --romi noaa_romi.txt --vpm noaa_vpm.txt --output-dir data/preprocessed_with_noaa

# time-series models, one config per lookback window (outputs in exps/)
python src/mjo/TFT/train.py --config configs/TFT/history_only/tft_10d_hist.yaml
python src/mjo/TSMixer/train.py --config configs/TSMixer/fusion/tsmixer_10d_hist.yaml

# bias correction: mean bias and MVLR (-> exps/baselines), and one LSTM per MJO phase group
python src/mjo/utils/baselines/baselines.py --config configs/baselines/fuxi_baselines.yaml
for g in 18 23 45 67; do python src/mjo/DLBC/train.py --config configs/DLBC/LSTM/dlbc-p$g.yaml; done
# each phase-group run saves its own test initializations; combine them into one directory
mkdir -p exps/DLBC/LSTM/combined && cp exps/DLBC/LSTM/p*/logs/version_0/outputs/*.txt exps/DLBC/LSTM/combined/
```

Hyperparameters can be tuned with `src/mjo/{TFT,TSMixer,DLBC}/tune.py --config <config>`. Point the analysis configs at your own outputs in `exps/` to evaluate retrained models; by default they use the published forecasts in `data/corrected`.

### Computing MJO indices from raw data

Each script takes `--help`. RMM follows Wheeler & Hendon (2004) with a 1979-09-07 to 2001-12-31 reference period, using OLR and zonal winds at 850 and 250 hPa (`--upper-level`; the ECMWF S2S archive provides 200 hPa).

```bash
python src/mjo/utils/RMM/eof.py --olr OLR.nc --era5 ERA5.zarr --output-dir EOF_DIR                       # EOFs, seasonal cycle, normalization
python src/mjo/utils/RMM/observation.py --olr OLR.nc --era5 ERA5.zarr --eof-dir EOF_DIR                  # observed RMM
bash scripts/download_fuxi.sh RAW_DIR                                                                    # raw FuXi-S2S forecasts
python src/mjo/utils/RMM/FuXi/forecast.py --olr OLR.nc --era5 ERA5.zarr --eof-dir EOF_DIR --forecast-dir RAW_DIR   # FuXi-S2S RMM
python src/mjo/utils/RMM/ecmwf.py --olr OLR.nc --era5 ERA5.zarr --eof-dir EOF_DIR --forecast-dir S2S_DIR --output-dir OUT   # ECMWF S2S RMM
```

`src/mjo/utils/ROMI/` computes ROMI from OLR in the same way (`eof.py`, `observation.py`, `FuXi/forecast.py`).

### Dashboard data

`python scripts/export_dashboard.py --out <dir>` writes the JSON behind the [dashboard](https://geruoa.github.io/fusion-outlook/dashboards/mjo.html) from the shipped data.

## Repository structure

```
mjo-predict/
├── configs/
│   ├── TFT/, TSMixer/    # history_only (Fig. 1), fusion (Fig. 6), history_only_noaa (Supp. Fig. S1)
│   ├── DLBC/LSTM/        # LSTM bias correction, one config per MJO phase group
│   ├── baselines/        # mean bias and MVLR correction
│   └── analysis/         # figure configs
├── data/                 # shipped indices (see Data)
├── scripts/              # make_figures.sh, download_fuxi.sh, export_dashboard.py
├── src/mjo/
│   ├── TFT/, TSMixer/    # time-series models (adapted from Darts)
│   ├── DLBC/             # LSTM bias correction
│   └── utils/
│       ├── RMM/, ROMI/, VPM/   # index computation and I/O
│       ├── baselines/          # mean bias and MVLR
│       └── analysis/           # metrics, bootstrap significance, figures
├── environment.yml
└── pyproject.toml
```

## Citation

```bibtex
@article{virji2026mjo,
  title   = {On the importance of historical context and bias correction in deep learning MJO prediction models},
  author  = {Virji, Khalil and A, Geruo and Velicogna, Isabella},
  journal = {npj Climate and Atmospheric Science},
  volume  = {9},
  pages   = {193},
  year    = {2026},
  doi     = {10.1038/s41612-026-01459-9}
}
```
