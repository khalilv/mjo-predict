"""Export the data behind the Fusion Outlook MJO dashboard (dashboards/mjo.html).

Reads the unzipped data/FuXi, data/corrected and data/observed and writes JSON in the layout the dashboard expects:

    <out>/skill.json          skill/bias curves with bootstrap significance, seasonality
    <out>/fuxi/index.json     list of FuXi-S2S initialization dates
    <out>/fuxi/<year>.json    per-init ensemble mean, members, observed and corrected RMM
    <out>/fuxi_stats.json     ensemble-mean vs member amplitude by lead (damping diagnostic)

Usage:
    python scripts/export_dashboard.py --out <dir> [--n-boot 1000]
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import (
    load_forecast, compute_bcor, compute_bmse,
    compute_amplitude_bias, compute_phase_bias,
    block_bootstrap_per_lead_significance,
)
from mjo.utils.analysis.lead_time_errors import compute_metric_across_leads
from mjo.utils.analysis.lead_time_bias import compute_bias_across_leads
from mjo.utils.analysis.monthly_errors import compute_metric_across_months

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data')
TEST = ('2019-01-01', '2022-01-01')
SEASON = ('2002-01-01', '2018-01-01')
MAX_LT = 42

# dashboard key -> directory under data/corrected (ERA5-based truth, BoM truth)
METHODS = {
    'mean': ('mean_bias', 'BoM/mean_bias'),
    'mvlr': ('mvlr', 'BoM/mvlr'),
    'lstm': ('lstm', 'BoM/lstm'),
}


def curves(model_dfs, fuxi_dfs, truth, boot, filter_type=None):
    """BCOR and BMSE per lead with per-lead significance vs FuXi for one subset.

    filter_type follows the repo convention: 'high' excludes high-amplitude
    forecasts (weak-MJO subset), 'low' excludes low-amplitude forecasts.
    """
    filt = dict(fuxi_ds=fuxi_dfs, filter_type=filter_type) if filter_type else {}
    all_dfs = list(model_dfs.values()) + [fuxi_dfs]
    lts = [MAX_LT] * len(all_dfs)
    out = {}
    for name, fn, hib in (('bcor', compute_bcor, True), ('bmse', compute_bmse, False)):
        sig = block_bootstrap_per_lead_significance(all_dfs, lts, truth, fn, higher_is_better=hib, **boot, **filt)
        res = {}
        for (k, dfs), s in zip(list(model_dfs.items()) + [('fuxi', fuxi_dfs)], sig + [None]):
            v = compute_metric_across_leads(dfs, MAX_LT, truth, fn, **filt)
            if name == 'bmse':
                v = v.sum(axis=1)
            res[k] = {'v': np.round(v, 4).tolist(), 'sig': [bool(x) for x in s] if s is not None else [True] * MAX_LT}
        out[name] = res
    return out


def biases(model_dfs, fuxi_dfs, truth, boot):
    all_dfs = list(model_dfs.values()) + [fuxi_dfs]
    lts = [MAX_LT] * len(all_dfs)
    out = {}
    for name, kind, fn in (('amp_bias', 'amplitude', compute_amplitude_bias), ('phase_bias', 'phase', compute_phase_bias)):
        sig = block_bootstrap_per_lead_significance(all_dfs, lts, truth, fn, higher_is_better=True, **boot)
        res = {}
        for (k, dfs), s in zip(list(model_dfs.items()) + [('fuxi', fuxi_dfs)], sig + [None]):
            v = compute_bias_across_leads(dfs, kind, MAX_LT, truth)
            res[k] = {'v': np.round(v, 4).tolist(), 'sig': [bool(x) for x in s] if s is not None else [True] * MAX_LT}
        out[name] = res
    return out


def skillful_days(v, threshold=0.5):
    """Number of leading lead days with BCOR above the threshold."""
    below = np.flatnonzero(np.asarray(v) <= threshold)
    return int(below[0]) if below.size else MAX_LT


def load_set(root, start, end, member=None):
    dfs, _, dates = load_forecast(root, start, end, member=member)
    return dfs, [d.replace('.txt', '') for d in dates]


def typical_start(test_dates, fuxi_dfs, lstm_dfs, truth):
    """Default browser start: a typical case for the LSTM, not the best one.

    Among test starts with observed amplitude >= 1 on the start date, pick the one whose
    weeks 3-6 BMSE ratio (LSTM / FuXi-S2S) is closest to the median ratio over all starts.
    """
    def bmse(df):
        obs = truth.loc[df.index[14:MAX_LT]]
        return float(((df.RMM1.values[14:MAX_LT] - obs.RMM1.values) ** 2 + (df.RMM2.values[14:MAX_LT] - obs.RMM2.values) ** 2).mean())
    ratios = [bmse(l) / bmse(f) for f, l in zip(fuxi_dfs, lstm_dfs)]
    med = float(np.median(ratios))
    strong = [(abs(r - med), d) for d, r in zip(test_dates, ratios) if truth.loc[pd.Timestamp(d)].amplitude >= 1]
    return min(strong)[1]


def damping_stats(means, members, dates):
    """Ensemble-mean vs member amplitude by lead, for all starts and strong starts.

    means: (n, 42, 2), members: (n, 51, 42, 2). Strong starts have ensemble-mean
    amplitude >= 1 on lead day 1. Spread is the RMS distance of members from the ensemble mean.
    """
    amp_mean = np.hypot(means[..., 0], means[..., 1])
    amp_mem = np.hypot(members[..., 0], members[..., 1])
    spread = np.sqrt(((members - means[:, None]) ** 2).sum(-1).mean(1))
    strong = amp_mean[:, 0] >= 1
    r = lambda a: np.round(a, 4).tolist()
    return {
        'n_init': len(dates), 'first': dates[0], 'last': dates[-1], 'lead': list(range(1, MAX_LT + 1)),
        'amp_ensmean_all': r(amp_mean.mean(0)), 'amp_member_all': r(amp_mem.mean((0, 1))), 'spread_all': r(spread.mean(0)),
        'amp_ensmean_strong': r(amp_mean[strong].mean(0)), 'amp_member_strong': r(amp_mem[strong].mean((0, 1))),
        'spread_strong': r(spread[strong].mean(0)), 'n_strong': int(strong.sum()),
    }


def browser(fuxi_root, truth, corrected, out_dir):
    """Per-year JSON: ensemble mean m, members e (x100 ints), observed o (+ o0 at init), corrected c.

    Also writes fuxi_stats.json (see damping_stats) from the same full-precision members.
    """
    os.makedirs(os.path.join(out_dir, 'fuxi'), exist_ok=True)
    dates = sorted(os.listdir(fuxi_root))
    by_year = {}
    obs = truth[['RMM1', 'RMM2']]
    all_means, all_members = [], []
    for d in dates:
        ddir = os.path.join(fuxi_root, d)
        mean = load_rmm_indices(os.path.join(ddir, 'mean.txt'))
        assert len(mean) == MAX_LT and mean.index[0] == pd.Timestamp(d) + timedelta(days=1), d
        members, full = [], []
        for k in range(51):
            mdf = load_rmm_indices(os.path.join(ddir, f'{k:02d}.txt'))
            full.append(mdf[['RMM1', 'RMM2']].values)
            members.append(np.round(mdf[['RMM1', 'RMM2']].values * 100).astype(int).ravel().tolist())
        all_means.append(mean[['RMM1', 'RMM2']].values)
        all_members.append(np.stack(full))
        rec = {
            'm': np.round(mean[['RMM1', 'RMM2']].values, 3).tolist(),
            'e': members,
            'o0': np.round(obs.loc[pd.Timestamp(d)].values, 3).tolist(),
            'o': np.round(obs.loc[mean.index].values, 3).tolist(),
        }
        c = {k: np.round(v[d][['RMM1', 'RMM2']].values, 3).tolist() for k, v in corrected.items() if d in v}
        if c:
            rec['c'] = c
        by_year.setdefault(d[:4], {})[d] = rec
    for y, recs in by_year.items():
        with open(os.path.join(out_dir, 'fuxi', f'{y}.json'), 'w') as f:
            json.dump(recs, f, separators=(',', ':'))
    with open(os.path.join(out_dir, 'fuxi', 'index.json'), 'w') as f:
        json.dump({'dates': dates, 'corrected': sorted(next(iter(corrected.values())).keys())}, f, separators=(',', ':'))
    with open(os.path.join(out_dir, 'fuxi_stats.json'), 'w') as f:
        json.dump(damping_stats(np.stack(all_means), np.stack(all_members), dates), f, separators=(',', ':'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='output directory (e.g. <site>/data/mjo)')
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--keep-lookback', help='existing skill.json whose "lookback" block is carried over')
    p.add_argument('--skip-browser', action='store_true')
    args = p.parse_args()
    boot = dict(n_samples=args.n_boot, block_size=1, ci_level=0.95, seed=args.seed)

    fuxi_root = os.path.join(DATA, 'FuXi')
    corrected_root = os.path.join(DATA, 'corrected')
    era5 = load_rmm_indices(os.path.join(DATA, 'observed', 'rmm.txt'))
    bom = load_rmm_indices(os.path.join(DATA, 'observed', 'bom_rmm.74toRealtime.txt'))

    fuxi_test, test_dates = load_set(fuxi_root, *TEST, member='mean')
    model, model_bom = {}, {}
    for k, (d_era5, d_bom) in METHODS.items():
        model[k], dates = load_set(os.path.join(corrected_root, d_era5), *TEST)
        assert dates == test_dates, f'{k}: init dates differ from FuXi'
        model_bom[k], dates = load_set(os.path.join(corrected_root, d_bom), *TEST)
        assert dates == test_dates, f'{k} (BoM): init dates differ from FuXi'

    skill = {'curves': {}}
    for subset, truth, mdl, ft in (('all', era5, model, None), ('high', era5, model, 'low'),
                                   ('low', era5, model, 'high'), ('bom', bom, model_bom, None)):
        res = curves(mdl, fuxi_test, truth, boot, ft)
        skill['curves'][f'{subset}_bcor'] = res['bcor']
        skill['curves'][f'{subset}_bmse'] = res['bmse']
    skill['curves'].update(biases(model, fuxi_test, era5, boot))

    fuxi_season, _ = load_set(fuxi_root, *SEASON, member='mean')
    monthly = compute_metric_across_months(fuxi_season, MAX_LT, era5, compute_bcor)
    season = [np.round(monthly[m], 4).tolist() for m in range(1, 13)]
    skill['season_bcor'] = season
    skill['season_bins'] = [[round(np.floor(v * 10) / 10, 1) for v in row] for row in season]

    if args.keep_lookback:
        with open(args.keep_lookback) as f:
            skill['lookback'] = json.load(f)['lookback']

    skill['default_start'] = typical_start(test_dates, fuxi_test, model['lstm'], era5)
    skill['horizons'] = {s:{k: skillful_days(v['v']) for k, v in skill['curves'][f'{s}_bcor'].items()}
                         for s in ('all', 'high', 'low', 'bom')}
    skill['meta'] = {
        'source': 'Virji, A & Velicogna (2026) npj Clim Atmos Sci 9:193, doi:10.1038/s41612-026-01459-9',
        'method': 'Computed from the forecast and observed RMM files in github.com/khalilv/mjo-predict '
                  '(scripts/export_dashboard.py). Significance: paired bootstrap vs FuXi-S2S, '
                  f'{args.n_boot} samples, 95% CI, seed {args.seed}.',
        'evaluation': f'Test inits {test_dates[0]} to {test_dates[-1]} (n={len(test_dates)}); '
                      'observed RMM from NOAA OLR + ERA5 U850/U250; BoM subset verified against BoM RMM. '
                      'Strong/weak subsets use FuXi-S2S ensemble-mean amplitude >= 1 / < 1 at each lead.',
        'season': f'FuXi-S2S BCOR by init month, inits {SEASON[0]} to {SEASON[1]} (exclusive).',
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'skill.json'), 'w') as f:
        json.dump(skill, f, separators=(',', ':'))
    print('Skillful days (BCOR > 0.5):', json.dumps(skill['horizons'], indent=1))

    if not args.skip_browser:
        corrected = {}
        for k, (d_era5, _) in METHODS.items():
            dfs, dates = load_set(os.path.join(corrected_root, d_era5), *TEST)
            corrected[k] = dict(zip(dates, dfs))
        browser(fuxi_root, era5, corrected, args.out)


if __name__ == '__main__':
    main()
