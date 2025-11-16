import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from mjo.utils.RMM.io import load_rmm_indices
from mjo.utils.analysis.utils import load_config
from mjo.utils.plot import phase_space_plot


def load_trajectory_data(
    start_date: str,
    trajectory_length: int,
    forecast_dirs: List[str],
    ensemble_dirs: List[Dict[str, Any]],
    ground_truth_path: str,
    bom_path: Optional[str] = None,
    ecmwf_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and slice trajectory data for phase space plotting."""
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = start_date_dt + timedelta(days=trajectory_length - 1)
    init_date = str(start_date_dt - timedelta(days=1)).split(' ')[0]

    # Load ground truth
    gt_df = load_rmm_indices(ground_truth_path)
    gt_slice = gt_df.loc[start_date_dt:end_date_dt]
    assert not gt_slice.isnull().values.any(), "NaNs found in ground truth dataframe slice"

    # Load BoM if provided
    bom_rmm1, bom_rmm2 = None, None
    if bom_path:
        bom_df = load_rmm_indices(bom_path)
        bom_slice = bom_df.loc[start_date_dt:end_date_dt]
        assert not bom_slice.isnull().values.any(), "NaNs found in BoM dataframe slice"
        assert (gt_slice.index == bom_slice.index).all(), \
            "Ground truth and BoM dataframe slices do not contain the same dates"
        bom_rmm1 = bom_slice.RMM1.values
        bom_rmm2 = bom_slice.RMM2.values

    # Load ECMWF if provided
    ecmwf_rmm1, ecmwf_rmm2, ecmwf_timestamps = None, None, None
    if ecmwf_path:
        ecmwf_df = load_rmm_indices(ecmwf_path)
        ecmwf_slice = ecmwf_df[:end_date_dt]
        ecmwf_rmm1 = ecmwf_slice.RMM1.values
        ecmwf_rmm2 = ecmwf_slice.RMM2.values
        ecmwf_timestamps = ecmwf_slice.index.values

    # Load forecast data
    predict_dfs = []
    labels = []

    for forecast_dir in forecast_dirs:
        pred_df = load_rmm_indices(os.path.join(forecast_dir, f'{init_date}.txt'))
        pred_df_slice = pred_df.loc[start_date:end_date_dt]
        assert not pred_df_slice.isnull().values.any(), \
            f"NaNs found in predict dataframe slice in {forecast_dir}"
        assert (gt_slice.index == pred_df_slice.index).all(), \
            f"Predict dataframe slice from {forecast_dir} and ground truth do not contain the same dates"
        predict_dfs.append(pred_df_slice)

    # Load ensemble data
    for ensemble_config in ensemble_dirs:
        ensemble_dir = ensemble_config['path']
        ensemble_label = ensemble_config['label']
        ensemble_members = ensemble_config.get('members', ['mean'])

        for member in ensemble_members:
            pred_df = load_rmm_indices(os.path.join(ensemble_dir, init_date, f'{member}.txt'))
            pred_df_slice = pred_df.loc[start_date:end_date_dt]
            assert not pred_df_slice.isnull().values.any(), \
                f"NaNs found in predict dataframe slice in {ensemble_dir}"
            assert (gt_slice.index == pred_df_slice.index).all(), \
                f"Predict dataframe slice from {ensemble_dir} and ground truth do not contain the same dates"
            predict_dfs.append(pred_df_slice)
            labels.append(f'{ensemble_label} {member}')

    return {
        'predict_dfs': predict_dfs,
        'labels': labels,
        'gt_slice': gt_slice,
        'bom_rmm1': bom_rmm1,
        'bom_rmm2': bom_rmm2,
        'ecmwf_rmm1': ecmwf_rmm1,
        'ecmwf_rmm2': ecmwf_rmm2,
        'ecmwf_timestamps': ecmwf_timestamps,
    }


def main():
    """Main function to generate phase space trajectory plot."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate phase space trajectory plot')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract configuration
    start_date = config['trajectory']['start_date']
    trajectory_length = config['trajectory']['length']
    forecast_dirs = config.get('forecast_dirs', [])
    forecast_labels = config.get('forecast_labels', [])
    ensemble_dirs = config.get('ensemble_dirs', [])
    ground_truth_path = config['paths']['ground_truth']
    bom_path = config['paths'].get('bom')
    ecmwf_path = config['paths'].get('ecmwf')
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Plot settings
    plot_settings = config.get('plot_settings', {})
    font_settings = plot_settings.get('fonts', {})
    axis_settings = plot_settings.get('axis', {})
    colormap = plot_settings.get('colormap', {})
    title_template = plot_settings.get('title', {}).get('template', '')

    # Build plot kwargs
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
        'colormap': colormap.get('name', 'tab10'),
        'marker_interval': plot_settings.get('marker_interval', 5),
        'legend_loc': plot_settings.get('legend_loc', 'best'),
    }

    # Load trajectory data
    data = load_trajectory_data(
        start_date=start_date,
        trajectory_length=trajectory_length,
        forecast_dirs=forecast_dirs,
        ensemble_dirs=ensemble_dirs,
        ground_truth_path=ground_truth_path,
        bom_path=bom_path,
        ecmwf_path=ecmwf_path,
    )

    # Generate title from template
    title = title_template.format(
        trajectory_length=trajectory_length,
        start_date=start_date
    ) if title_template else None

    # Generate plot
    output_filename = f'{start_date}.png'
    phase_space_plot(
        pred_rmm1s=[p.RMM1.values for p in data['predict_dfs']],
        gt_rmm1=data['gt_slice'].RMM1.values,
        pred_rmm2s=[p.RMM2.values for p in data['predict_dfs']],
        gt_rmm2=data['gt_slice'].RMM2.values,
        gt_label=config.get('gt_label', 'Observed'),
        labels=forecast_labels + data['labels'],
        bom_rmm1=data['bom_rmm1'],
        bom_rmm2=data['bom_rmm2'],
        ecmwf_rmm1=data['ecmwf_rmm1'],
        ecmwf_rmm2=data['ecmwf_rmm2'],
        ecmwf_timestamps=data['ecmwf_timestamps'],
        title=title,
        output_filename=os.path.join(output_dir, output_filename),
        **plot_kwargs
    )


if __name__ == "__main__":
    main()
