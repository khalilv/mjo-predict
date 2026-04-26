import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import load_config
from mjo.utils.plot import phase_space_plot


def to_rmm_convention(df):
    """Apply NOAA OMI/ROMI -> RMM-convention mapping: (PC1, PC2) -> (PC2, -PC1).

    Per NOAA PSL: to compare OMI/ROMI directly with RMM, swap PC ordering and
    flip the sign of (original) PC1 so OMI(PC2) is analogous to RMM(PC1) and
    -OMI(PC1) is analogous to RMM(PC2). Same applies to ROMI.
    """
    df = df.copy()
    new_rmm1 = df['RMM2'].values
    new_rmm2 = -df['RMM1'].values
    df['RMM1'] = new_rmm1
    df['RMM2'] = new_rmm2
    return df


def _slice_to_window(df, start, end, source):
    """Slice df to [start, end] inclusive and assert no NaNs."""
    sliced = df.loc[start:end]
    assert not sliced.isnull().values.any(), f"NaNs found in slice from {source}"
    return sliced


def load_trajectories(
    start_date: str,
    trajectory_length: int,
    references: List[Dict[str, Any]],
    forecasts: List[Dict[str, Any]],
    ensembles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Load all phase-space trajectories from the unified config sections.

    references: continuous time-series files sliced by trajectory window. Each
        entry must have {path, label, color, linestyle}; one entry must have
        primary: true (used for the date alignment reference).
    forecasts: per-init-date single-file forecasts at <path>/<init_date>.txt.
    ensembles: per-init-date member directories at <path>/<init_date>/<member>.txt.
        Each entry has {path, label, color, linestyle, members: [...]}.

    Any entry may set transform_to_rmm: true to apply the OMI/ROMI -> RMM swap.
    """
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = start_date_dt + timedelta(days=trajectory_length - 1)
    init_date = (start_date_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    primaries = [r for r in references if r.get('primary')]
    assert len(primaries) == 1, \
        f"Exactly one reference must have primary: true (found {len(primaries)})"
    primary_path = primaries[0]['path']

    primary_df = load_rmm_indices(primary_path)
    if primaries[0].get('transform_to_rmm'):
        primary_df = to_rmm_convention(primary_df)
    primary_slice = _slice_to_window(primary_df, start_date_dt, end_date_dt, primary_path)

    trajectories: List[Dict[str, Any]] = []

    for ref in references:
        if ref.get('primary'):
            df_slice = primary_slice
        else:
            df = load_rmm_indices(ref['path'])
            if ref.get('transform_to_rmm'):
                df = to_rmm_convention(df)
            df_slice = _slice_to_window(df, start_date_dt, end_date_dt, ref['path'])
            assert (primary_slice.index == df_slice.index).all(), \
                f"Reference {ref['path']} dates don't align with primary"
        trajectories.append({
            'rmm1': df_slice['RMM1'].values,
            'rmm2': df_slice['RMM2'].values,
            'label': ref['label'],
            'color': ref['color'],
            'linestyle': ref.get('linestyle', '-'),
            'alpha': ref.get('alpha', 0.85),
        })

    for fc in forecasts:
        df = load_rmm_indices(os.path.join(fc['path'], f'{init_date}.txt'))
        if fc.get('transform_to_rmm'):
            df = to_rmm_convention(df)
        df_slice = _slice_to_window(df, start_date_dt, end_date_dt, fc['path'])
        assert (primary_slice.index == df_slice.index).all(), \
            f"Forecast {fc['path']} dates don't align with primary"
        trajectories.append({
            'rmm1': df_slice['RMM1'].values,
            'rmm2': df_slice['RMM2'].values,
            'label': fc['label'],
            'color': fc['color'],
            'linestyle': fc.get('linestyle', '-'),
            'alpha': fc.get('alpha', 0.85),
        })

    for ens in ensembles:
        members = ens.get('members', ['mean'])
        for member in members:
            df = load_rmm_indices(os.path.join(ens['path'], init_date, f'{member}.txt'))
            if ens.get('transform_to_rmm'):
                df = to_rmm_convention(df)
            df_slice = _slice_to_window(df, start_date_dt, end_date_dt, ens['path'])
            assert (primary_slice.index == df_slice.index).all(), \
                f"Ensemble {ens['path']}/{member} dates don't align with primary"
            label = ens['label'] if len(members) == 1 else f"{ens['label']} {member}"
            trajectories.append({
                'rmm1': df_slice['RMM1'].values,
                'rmm2': df_slice['RMM2'].values,
                'label': label,
                'color': ens['color'],
                'linestyle': ens.get('linestyle', '-'),
                'alpha': ens.get('alpha', 0.85),
            })

    return trajectories


def main():
    """Main function to generate phase space trajectory plot."""
    parser = argparse.ArgumentParser(description='Generate phase space trajectory plot')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config)

    start_date = config['trajectory']['start_date']
    trajectory_length = config['trajectory']['length']
    references = config.get('references', [])
    forecasts = config.get('forecasts', [])
    ensembles = config.get('ensembles', [])
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    plot_settings = config.get('plot_settings', {})
    font_settings = plot_settings.get('fonts', {})
    axis_settings = plot_settings.get('axis', {})
    title_template = plot_settings.get('title', {}).get('template', '')

    plot_kwargs = {
        'fontsize_title': font_settings.get('title', 14),
        'fontsize_axis_label': font_settings.get('axis_label', 12),
        'fontsize_region_label': font_settings.get('region_label', 11),
        'fontsize_phase_label': font_settings.get('phase_label', 11),
        'fontsize_legend': font_settings.get('legend', 11),
        'fontweight_title': font_settings.get('title_weight', 'bold'),
        'fontweight_title_label': font_settings.get('title_label_weight'),
        'font_family': font_settings.get('family'),
        'figsize': tuple(plot_settings.get('figsize', [8, 8])),
        'xlim': tuple(axis_settings.get('xlim', [-4, 4])),
        'ylim': tuple(axis_settings.get('ylim', [-4, 4])),
        'marker_interval': plot_settings.get('marker_interval', 5),
        'legend_loc': plot_settings.get('legend_loc', 'best'),
    }

    trajectories = load_trajectories(
        start_date=start_date,
        trajectory_length=trajectory_length,
        references=references,
        forecasts=forecasts,
        ensembles=ensembles,
    )

    title = title_template.format(
        trajectory_length=trajectory_length,
        start_date=start_date
    ) if title_template else None

    output_filename = f'{start_date}.png'
    phase_space_plot(
        trajectories=trajectories,
        title=title,
        output_filename=os.path.join(output_dir, output_filename),
        **plot_kwargs
    )


if __name__ == "__main__":
    main()
