import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import cm
from typing import Sequence, Tuple, List, Union, Optional, Literal, Dict
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

def correlation_scatter_plot(pred_rmm1, gt_rmm1, pred_rmm2, gt_rmm2, pred_amplitude, gt_amplitude, pred_label = None, gt_label = None, output_filename = None):
    pred_label = pred_label if pred_label else 'Predictions'
    gt_label = gt_label if gt_label else 'Ground Truth'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    rmm1_corr = np.corrcoef(pred_rmm1, gt_rmm1)[0, 1]
    axes[0].scatter(pred_rmm1, gt_rmm1, alpha=0.8, s=2)
    axes[0].set_title(f"RMM1 correlation. r = {rmm1_corr:.3f}")
    axes[0].set_xlabel(pred_label)
    axes[0].set_ylabel(gt_label)
    axes[0].plot([pred_rmm1.min(), pred_rmm1.max()], [pred_rmm1.min(), pred_rmm1.max()], 'k--')
    axes[0].grid(True)

    rmm2_corr = np.corrcoef(pred_rmm2, gt_rmm2)[0, 1]
    axes[1].scatter(pred_rmm2, gt_rmm2, alpha=0.8, s=2)
    axes[1].set_title(f"RMM2 correlation. r = {rmm2_corr:.3f}")
    axes[1].set_xlabel(pred_label)
    axes[1].set_ylabel(gt_label)
    axes[1].plot([pred_rmm2.min(), pred_rmm2.max()], [pred_rmm2.min(), pred_rmm2.max()], 'k--')
    axes[1].grid(True)

    amplitude_corr = np.corrcoef(pred_rmm1, gt_rmm1)[0, 1]
    axes[2].scatter(pred_amplitude, gt_amplitude, alpha=0.8, s=2)
    axes[2].set_title(f"Amplitude correlation. r = {amplitude_corr:.3f}")
    axes[2].set_xlabel(pred_label)
    axes[2].set_ylabel(gt_label)
    axes[2].plot([pred_amplitude.min(), pred_amplitude.max()], [pred_amplitude.min(), pred_amplitude.max()], 'k--')
    axes[2].grid(True)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def phase_space_plot(
    pred_rmm1s: List[np.ndarray],
    gt_rmm1: np.ndarray,
    pred_rmm2s: List[np.ndarray],
    gt_rmm2: np.ndarray,
    labels: List[str],
    gt_label: str,
    bom_rmm1: Optional[np.ndarray] = None,
    bom_rmm2: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    output_filename: Optional[str] = None,
    ecmwf_rmm1: Optional[np.ndarray] = None,
    ecmwf_rmm2: Optional[np.ndarray] = None,
    ecmwf_timestamps: Optional[np.ndarray] = None,
    *,
    figsize: Tuple[float, float] = (8, 8),
    xlim: Tuple[float, float] = (-4, 4),
    ylim: Tuple[float, float] = (-4, 4),
    colormap: str = 'tab10',
    fontsize_title: int = 14,
    fontsize_axis_label: int = 12,
    fontsize_region_label: int = 11,
    fontsize_phase_label: int = 11,
    fontsize_legend: int = 11,
    fontweight_title: str = 'bold',
    fontweight_title_label: Optional[str] = None,
    font_family: Optional[str] = None,
    marker_interval: int = 5,
    legend_loc: str = 'best',
) -> plt.Figure:
    """Create a phase space trajectory plot showing RMM1 vs RMM2.

    Args:
        pred_rmm1s: List of predicted RMM1 arrays
        gt_rmm1: Ground truth RMM1 array
        pred_rmm2s: List of predicted RMM2 arrays
        gt_rmm2: Ground truth RMM2 array
        labels: Labels for each prediction
        gt_label: Label for ground truth
        bom_rmm1: Optional BoM RMM1 array
        bom_rmm2: Optional BoM RMM2 array
        title: Plot title
        output_filename: Path to save figure
        ecmwf_rmm1: Optional ECMWF RMM1 array
        ecmwf_rmm2: Optional ECMWF RMM2 array
        ecmwf_timestamps: Optional ECMWF timestamps
        figsize: Figure size
        xlim: X-axis limits
        ylim: Y-axis limits
        colormap: Colormap name for predictions
        fontsize_title: Font size for title
        fontsize_axis_label: Font size for axis labels
        fontsize_region_label: Font size for region labels
        fontsize_phase_label: Font size for phase labels
        fontsize_legend: Font size for legend
        fontweight_title: Font weight for title
        fontweight_title_label: Font weight for title label (e.g., "(a)")
        font_family: Font family
        marker_interval: Interval for trajectory markers
        legend_loc: Legend location

    Returns:
        Matplotlib figure object
    """

    # Set font family if provided
    if font_family:
        plt.rcParams['font.family'] = font_family

    def _add_region_labels():
        # Position labels based on xlim/ylim (at ~87.5% of the limit)
        x_pos = xlim[1] * 0.875
        y_pos = ylim[1] * 0.875
        font_props = {'fontsize': fontsize_region_label, 'ha': 'center', 'va': 'center'}
        if font_family:
            font_props['fontfamily'] = font_family

        plt.text(0, -y_pos, "Indian Ocean", **font_props)
        plt.text(x_pos, 0, "Maritime Continent", rotation=-90, **font_props)
        plt.text(0, y_pos, "Western Pacific", **font_props)
        plt.text(-x_pos, 0, "Western Hem.& Africa", rotation=90, **font_props)

    def _draw_phase_wedges():
        # Scale radius based on xlim/ylim (use average of the two)
        max_lim = (xlim[1] + ylim[1]) / 2
        r = math.sqrt(2 * max_lim * max_lim)
        phase_labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

        for i in range(8):
            angle = (5 + i) * np.pi / 4
            x = np.cos(angle)
            y = np.sin(angle)
            plt.plot([0, r * x], [0, r * y], color='lightgray', lw=0.8, zorder=0)

        font_props = {'fontsize': fontsize_phase_label, 'ha': 'center', 'va': 'center', 'color': 'gray'}
        if font_family:
            font_props['fontfamily'] = font_family

        for i, label in enumerate(phase_labels):
            angle = (4.5 + i) * np.pi / 4
            x = max_lim * np.cos(angle)
            y = max_lim * np.sin(angle)
            plt.text(x, y, label, **font_props)

    assert len(labels) == len(pred_rmm1s), 'Number of labels must match number of prediction sources'
    fig = plt.figure(figsize=figsize)

    ax = plt.gca()
    # Draw a unit circle in light gray
    unit_circle = mpatches.Circle((0, 0), 1, edgecolor='lightgray', facecolor='none', linewidth=0.8, linestyle='-', zorder=0)
    ax.add_patch(unit_circle)

    # Use configured colormap
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(labels)))

    for i, label in enumerate(labels):
        plt.plot(pred_rmm1s[i], pred_rmm2s[i], color=colors[i], alpha=0.8, label=label)
        plt.plot(pred_rmm1s[i][0], pred_rmm2s[i][0], color=colors[i], marker='o')  # start
        for lt in range(marker_interval, len(pred_rmm1s[i]), marker_interval):
            plt.plot(pred_rmm1s[i][lt], pred_rmm2s[i][lt], color=colors[i], marker='.')

    if bom_rmm1 is not None and bom_rmm2 is not None:
        plt.plot(bom_rmm1, bom_rmm2, color='lightgray', linestyle='--', alpha=0.75, label='BoM')
        plt.plot(bom_rmm1[0], bom_rmm2[0], color='lightgray', marker='o')  # start
        for lt in range(marker_interval, len(bom_rmm1), marker_interval):
            plt.plot(bom_rmm1[lt], bom_rmm2[lt], color='lightgray', marker='.')

    if ecmwf_rmm1 is not None and ecmwf_rmm2 is not None:
        plt.plot(ecmwf_rmm1, ecmwf_rmm2, color='red', alpha=0.8, label='ECMWF')
        # Start at marker_interval-1 to match original behavior (which started at 4 with interval 5)
        for lt in range(marker_interval - 1, len(ecmwf_rmm1), marker_interval):
            plt.plot(ecmwf_rmm1[lt], ecmwf_rmm2[lt], color='red', marker='.')

    plt.plot(gt_rmm1, gt_rmm2, color='black', linestyle='--', alpha=0.75, label=gt_label)
    plt.plot(gt_rmm1[0], gt_rmm2[0], color='black', marker='o')  # start
    for lt in range(marker_interval, len(gt_rmm1), marker_interval):
        plt.plot(gt_rmm1[lt], gt_rmm2[lt], color='black', marker='.')

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    _draw_phase_wedges()
    _add_region_labels()

    # Apply font settings to axis labels
    font_props = {'fontsize': fontsize_axis_label}
    if font_family:
        font_props['fontfamily'] = font_family
    plt.xlabel("RMM1", **font_props)
    plt.ylabel("RMM2", **font_props)

    # Handle title with optional bold label support (e.g., "(a)")
    if title:
        # Check if title starts with a label like "(a)", "(b)", etc.
        if title and title.startswith('(') and ')' in title[:5]:
            close_paren = title.index(')')
            label_part = title[:close_paren+1]
            rest_part = title[close_paren+1:].strip()
            if fontweight_title_label:
                plt.title(label_part + ' ' + rest_part, fontsize=fontsize_title,
                         fontweight=fontweight_title, pad=10)
                # Make just the label bold if specified
                ax.title.set_text('')
                ax.text(0.5, 1.02, label_part, transform=ax.transAxes,
                       fontsize=fontsize_title, fontweight=fontweight_title_label,
                       ha='center', va='bottom')
                ax.text(0.5 + len(label_part)*0.012, 1.02, rest_part, transform=ax.transAxes,
                       fontsize=fontsize_title, fontweight=fontweight_title,
                       ha='left', va='bottom')
            else:
                plt.title(title, fontsize=fontsize_title, fontweight=fontweight_title, pad=10)
        else:
            plt.title(title, fontsize=fontsize_title, fontweight=fontweight_title, pad=10)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect('equal')

    # prevent duplicate legend entries
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    plt.legend(unique.values(), unique.keys(), loc=legend_loc, fontsize=fontsize_legend)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    return fig

def phase_space_composite_plot(
    pred_composites: List[Dict[int, Tuple[np.ndarray, np.ndarray]]],
    gt_composites: Dict[int, Tuple[np.ndarray, np.ndarray]],
    labels: List[str],
    gt_label: str,
    bom_composites: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    title: Optional[str] = None,
    output_filename: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (8, 8),
    xlim: Tuple[float, float] = (-1.75, 1.75),
    ylim: Tuple[float, float] = (-1.75, 1.75),
    colormap: str = 'tab10',
    fontsize_title: int = 14,
    fontsize_axis_label: int = 12,
    fontsize_region_label: int = 11,
    fontsize_phase_label: int = 11,
    fontsize_legend: int = 11,
    fontweight_title: str = 'bold',
    fontweight_title_label: Optional[str] = None,
    font_family: Optional[str] = None,
    marker_interval: int = 5,
    legend_loc: str = 'best',
) -> plt.Figure:
    """Create a composite phase space trajectory plot showing RMM1 vs RMM2.

    Composite plots aggregate trajectories by initialization phase, showing
    the average behavior for each phase.

    Args:
        pred_composites: List of composite dictionaries for each prediction source.
            Each composite is a dict mapping phase (1-8) to (rmm1, rmm2) tuple.
        gt_composites: Ground truth composite dictionary mapping phase to (rmm1, rmm2)
        labels: Labels for each prediction source
        gt_label: Label for ground truth
        bom_composites: Optional BoM composite dictionary
        title: Plot title
        output_filename: Path to save figure
        figsize: Figure size
        xlim: X-axis limits
        ylim: Y-axis limits
        colormap: Colormap name for predictions
        fontsize_title: Font size for title
        fontsize_axis_label: Font size for axis labels
        fontsize_region_label: Font size for region labels
        fontsize_phase_label: Font size for phase labels
        fontsize_legend: Font size for legend
        fontweight_title: Font weight for title
        fontweight_title_label: Font weight for title label (e.g., "(a)")
        font_family: Font family
        marker_interval: Interval for trajectory markers
        legend_loc: Legend location

    Returns:
        Matplotlib figure object
    """

    # Set font family if provided
    if font_family:
        plt.rcParams['font.family'] = font_family

    def _add_region_labels():
        # Position labels based on xlim/ylim (at ~87.5% of the limit)
        x_pos = xlim[1] * 0.857  # 1.5/1.75
        y_pos = ylim[1] * 0.857
        font_props = {'fontsize': fontsize_region_label, 'ha': 'center', 'va': 'center'}
        if font_family:
            font_props['fontfamily'] = font_family

        plt.text(0, -y_pos, "Indian Ocean", **font_props)
        plt.text(x_pos, 0, "Maritime Continent", rotation=-90, **font_props)
        plt.text(0, y_pos, "Western Pacific", **font_props)
        plt.text(-x_pos, 0, "Western Hem.& Africa", rotation=90, **font_props)

    def _draw_phase_wedges():
        # Scale radius based on xlim/ylim (use average of the two)
        max_lim = (xlim[1] + ylim[1]) / 2
        r = math.sqrt(2 * max_lim * max_lim)
        phase_labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

        for i in range(8):
            angle = (5 + i) * np.pi / 4
            x = np.cos(angle)
            y = np.sin(angle)
            plt.plot([0, r * x], [0, r * y], color='lightgray', lw=0.8, zorder=0)

        font_props = {'fontsize': fontsize_phase_label, 'ha': 'center', 'va': 'center', 'color': 'gray'}
        if font_family:
            font_props['fontfamily'] = font_family

        for i, label in enumerate(phase_labels):
            angle = (4.5 + i) * np.pi / 4
            x = max_lim * np.cos(angle)
            y = max_lim * np.sin(angle)
            plt.text(x, y, label, **font_props)

    assert len(labels) == len(pred_composites), 'Number of labels must match number of prediction sources'
    fig = plt.figure(figsize=figsize)

    ax = plt.gca()
    # Draw a unit circle in light gray
    unit_circle = mpatches.Circle((0, 0), 1, edgecolor='lightgray', facecolor='none', linewidth=0.8, linestyle='-', zorder=0)
    ax.add_patch(unit_circle)

    # Use configured colormap
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(labels)))

    for i, label in enumerate(labels):
        for _, (rmm1, rmm2) in pred_composites[i].items():
            plt.plot(rmm1, rmm2, color=colors[i], alpha=0.8, label=label)
            plt.plot(rmm1[0], rmm2[0], color=colors[i], marker='o')  # start
            for lt in range(marker_interval, len(rmm1), marker_interval):
                plt.plot(rmm1[lt], rmm2[lt], color=colors[i], marker='.')

    for _, (rmm1, rmm2) in gt_composites.items():
        plt.plot(rmm1, rmm2, color='black', linestyle='--', alpha=0.75, label=gt_label)
        plt.plot(rmm1[0], rmm2[0], color='black', marker='o')  # start
        for lt in range(marker_interval, len(rmm1), marker_interval):
            plt.plot(rmm1[lt], rmm2[lt], color='black', marker='.')

    if bom_composites is not None:
        for _, (rmm1, rmm2) in bom_composites.items():
            plt.plot(rmm1, rmm2, color='lightgray', linestyle='--', alpha=0.75, label='BoM')
            plt.plot(rmm1[0], rmm2[0], color='lightgray', marker='o')  # start
            for lt in range(marker_interval, len(rmm1), marker_interval):
                plt.plot(rmm1[lt], rmm2[lt], color='lightgray', marker='.')

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    _draw_phase_wedges()
    _add_region_labels()

    # Apply font settings to axis labels
    font_props = {'fontsize': fontsize_axis_label}
    if font_family:
        font_props['fontfamily'] = font_family
    plt.xlabel("RMM1", **font_props)
    plt.ylabel("RMM2", **font_props)

    # Handle title with optional bold label support (e.g., "(a)")
    if title:
        # Check if title starts with a label like "(a)", "(b)", etc.
        if title.startswith('(') and ')' in title[:5]:
            close_paren = title.index(')')
            label_part = title[:close_paren+1]
            rest_part = title[close_paren+1:].strip()
            if fontweight_title_label:
                plt.title(label_part + ' ' + rest_part, fontsize=fontsize_title,
                         fontweight=fontweight_title, pad=10)
                # Make just the label bold if specified
                ax.title.set_text('')
                ax.text(0.5, 1.02, label_part, transform=ax.transAxes,
                       fontsize=fontsize_title, fontweight=fontweight_title_label,
                       ha='center', va='bottom')
                ax.text(0.5 + len(label_part)*0.012, 1.02, rest_part, transform=ax.transAxes,
                       fontsize=fontsize_title, fontweight=fontweight_title,
                       ha='left', va='bottom')
            else:
                plt.title(title, fontsize=fontsize_title, fontweight=fontweight_title, pad=10)
        else:
            plt.title(title, fontsize=fontsize_title, fontweight=fontweight_title, pad=10)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect('equal')

    # prevent duplicate legend entries
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    plt.legend(unique.values(), unique.keys(), loc=legend_loc, fontsize=fontsize_legend)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    return fig

def bivariate_correlation_vs_lead_time_plot(lead_times, correlations, labels, output_filename=None):
    assert len(correlations) == len(labels), 'Number of labels must match number of correlation sources'
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(labels):
        plt.plot(lead_times[i], correlations[i], color=colors[i], label=label)
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1) 
    plt.xlabel('Lead time (days)')
    plt.ylabel('Bivariate correlation')
    # plt.title('Bivariate Correlation vs Lead Time')
    plt.legend()
    ax = plt.gca()
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(0, 43)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))   # grid every day
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x % 10 == 0 else ""))  # label every 10 days
    # plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5)
    # plt.grid(which='major', axis='y')
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def bivariate_mse_vs_lead_time_plot(lead_times, bmsea, bmsep, labels, output_filename=None, combined=False):
    assert len(bmsea) == len(labels), 'Number of labels must match number of bmsea sources'
    assert len(bmsep) == len(labels), 'Number of labels must match number of bmsep sources'

    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(labels):
        if combined:
            plt.plot(lead_times[i], bmsea[i] + bmsep[i], color=colors[i], linestyle='-', label=f'{label}')
        else:
            plt.plot(lead_times[i], bmsea[i], color=colors[i], linestyle='-', label=f'{label} (Amplitude)')
            plt.plot(lead_times[i], bmsep[i], color=colors[i], linestyle='--', label=f'{label} (Phase)')
    plt.xlabel('Lead time (days)')
    plt.ylabel('Bivariate mean squared error')
    # plt.title('Bivariate Mean Squared Error vs Lead Time')
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.set_ylim(0, 2.0)
    ax.set_xlim(0, 43)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))   # grid every day
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x % 10 == 0 else ""))  # label every 10 days
    # plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5)
    # plt.grid(which='major', axis='y')
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def bivariate_mse_vs_init_date_plot(init_dates, bmse, labels, output_filename=None):
    assert len(bmse) == len(labels), 'Number of labels must match number of bmse sources'
    assert len(init_dates) == len(bmse), 'Number of bmse sources must match number of init dates'

    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(labels):
        plt.plot(init_dates[i], bmse[i], color=colors[i], linestyle='-', label=f'{label}')
    plt.xlabel('Date')
    plt.ylabel('Bivariate Mean Squared Error')
    plt.title('Bivariate Mean Squared Error vs Init Date')
    plt.legend()
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def lead_time_skill_subplot(
    lead_times: Sequence[np.ndarray],
    metric_data: Sequence[np.ndarray],
    labels: Sequence[str],
    metric_type: Literal['bcor', 'bmse'],
    ax: Optional[plt.Axes] = None,
    *,
    title: Optional[str] = None,
    x_label: Optional[str] = 'Lead time (days)',
    y_label: Optional[str] = None,
    show_legend: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = (0, 43),
    threshold: Optional[float] = None,
    threshold_style: dict = None,
    fontsize_title: int = 14,
    fontsize_axis_label: int = 12,
    fontsize_tick_label: int = 11,
    fontsize_legend: int = 11,
    fontweight_title: str = 'bold',
    fontweight_title_label: Optional[str] = None,
    font_family: Optional[str] = None,
    colormap: str = 'viridis',
    legend_loc: str = 'best',
    # BMSE-specific parameters
    bmsea: Optional[Sequence[np.ndarray]] = None,
    bmsep: Optional[Sequence[np.ndarray]] = None,
    combined: bool = False,
) -> plt.Axes:
    """Unified plotting function for lead time metrics (BCOR or BMSE).

    Args:
        lead_times: Lead time arrays for each model
        metric_data: Metric values (BCOR or combined BMSE)
        labels: Model labels
        metric_type: 'bcor' or 'bmse'
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        x_label: X-axis label (None to hide)
        y_label: Y-axis label (None to hide, defaults based on metric_type)
        show_legend: Whether to show legend
        ylim: Y-axis limits
        xlim: X-axis limits
        threshold: Horizontal threshold line value
        threshold_style: Dict of threshold line style kwargs
        fontsize_*: Font sizes for various elements
        fontweight_title: Font weight for title ('normal', 'bold', etc.)
        colormap: Colormap name
        legend_loc: Legend location
        bmsea: BMSE amplitude errors (for bmse type with combined=False)
        bmsep: BMSE phase errors (for bmse type with combined=False)
        combined: Whether to plot combined BMSE (for bmse type)

    Returns:
        Matplotlib axes object
    """
    # Set default y_label based on metric type if not provided
    if y_label is None:
        if metric_type == 'bcor':
            y_label = 'Bivariate correlation'
        elif metric_type == 'bmse':
            y_label = 'Bivariate mean squared error'

    # Set default ylim based on metric type if not provided
    if ylim is None:
        if metric_type == 'bcor':
            ylim = (0.3, 1.0)
        elif metric_type == 'bmse':
            ylim = (0, 2.0)

    # Set default threshold for bcor
    if threshold is None and metric_type == 'bcor':
        threshold = 0.5

    # Default threshold style
    if threshold_style is None:
        threshold_style = {'color': 'black', 'linestyle': '--', 'linewidth': 1}

    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Generate colors
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(labels)))

    # Plot data
    if metric_type == 'bcor':
        for i, label in enumerate(labels):
            ax.plot(lead_times[i], metric_data[i], color=colors[i], label=label)
    elif metric_type == 'bmse':
        if combined or (bmsea is None and bmsep is None):
            # Plot combined BMSE
            for i, label in enumerate(labels):
                ax.plot(lead_times[i], metric_data[i], color=colors[i], linestyle='-', label=label)
        else:
            # Plot separate amplitude and phase errors
            assert bmsea is not None and bmsep is not None, 'Must provide bmsea and bmsep for non-combined BMSE plot'
            for i, label in enumerate(labels):
                ax.plot(lead_times[i], bmsea[i], color=colors[i], linestyle='-', label=f'{label} (Amplitude)')
                ax.plot(lead_times[i], bmsep[i], color=colors[i], linestyle='--', label=f'{label} (Phase)')

    # Add threshold line if specified
    if threshold is not None:
        ax.axhline(threshold, **threshold_style)

    # Set font family if specified
    if font_family:
        plt.rcParams['font.family'] = font_family

    # Set labels and title
    label_kwargs = {'fontsize': fontsize_axis_label}
    if font_family:
        label_kwargs['family'] = font_family

    if x_label:
        ax.set_xlabel(x_label, **label_kwargs)
    if y_label:
        ax.set_ylabel(y_label, **label_kwargs)
    if title:
        # Check if we should bold only the label part (e.g., "(a)")
        import re
        label_match = re.match(r'^(\([a-zA-Z0-9]+\))\s*(.*)$', title)
        if label_match and fontweight_title_label == 'bold':
            label_part = label_match.group(1)
            text_part = label_match.group(2)
            title_str = f"$\\mathbf{{{label_part}}}$ {text_part}"
            title_kwargs = {'fontsize': fontsize_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(title_str, **title_kwargs)
        else:
            title_kwargs = {'fontsize': fontsize_title, 'fontweight': fontweight_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(title, **title_kwargs)

    # Set limits
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    # Configure x-axis ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x % 10 == 0 else ""))

    # Configure tick label size
    ax.tick_params(axis='both', labelsize=fontsize_tick_label)

    # Add legend
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=fontsize_legend)

    return ax


def lead_time_bcor_bmse_plot(
    lead_times: Sequence[np.ndarray],
    correlations: Sequence[np.ndarray],
    bmsea: Sequence[np.ndarray],
    bmsep: Sequence[np.ndarray],
    labels: Sequence[str],
    output_filename: Optional[str] = None,
    *,
    titles: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
    combined_bmse: bool = True,
    figsize: Tuple[float, float] = (16, 5),
    **plot_kwargs
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Create side-by-side BCOR and BMSE plots.

    Args:
        lead_times: Lead time arrays for each model
        correlations: Bivariate correlation arrays
        bmsea: BMSE amplitude error arrays
        bmsep: BMSE phase error arrays
        labels: Model labels
        output_filename: Path to save figure
        titles: Tuple of (bcor_title, bmse_title)
        combined_bmse: Whether to plot combined BMSE
        figsize: Figure size
        **plot_kwargs: Additional kwargs passed to lead_time_metric_plot

    Returns:
        Figure and tuple of (bcor_ax, bmse_ax)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Compute combined BMSE if needed
    if combined_bmse:
        bmse_combined = [bmsea[i] + bmsep[i] for i in range(len(bmsea))]
    else:
        bmse_combined = None

    # Plot BCOR
    lead_time_skill_subplot(
        lead_times=lead_times,
        metric_data=correlations,
        labels=labels,
        metric_type='bcor',
        ax=ax1,
        title=titles[0],
        **plot_kwargs
    )

    # Plot BMSE
    lead_time_skill_subplot(
        lead_times=lead_times,
        metric_data=bmse_combined if combined_bmse else bmsea,
        labels=labels,
        metric_type='bmse',
        ax=ax2,
        title=titles[1],
        bmsea=bmsea if not combined_bmse else None,
        bmsep=bmsep if not combined_bmse else None,
        combined=combined_bmse,
        **plot_kwargs
    )

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, (ax1, ax2)


def lead_time_bias_plot(
    lead_times: Sequence[np.ndarray],
    amplitude_bias: Sequence[np.ndarray],
    phase_bias: Sequence[np.ndarray],
    labels: Sequence[str],
    output_filename: Optional[str] = None,
    *,
    titles: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
    figsize: Tuple[float, float] = (16, 5),
    **plot_kwargs
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Create side-by-side amplitude and phase bias plots.

    Args:
        lead_times: Lead time arrays for each model
        amplitude_bias: Amplitude bias arrays for each model
        phase_bias: Phase bias arrays for each model
        labels: Model labels
        output_filename: Path to save figure
        titles: Tuple of (amplitude_title, phase_title)
        figsize: Figure size
        **plot_kwargs: Additional kwargs passed to lead_time_bias_subplot
                       (can include ylim_amplitude and ylim_phase for separate y-limits)

    Returns:
        Figure and tuple of (amplitude_ax, phase_ax)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract amplitude and phase specific parameters if provided
    ylim_amplitude = plot_kwargs.pop('ylim_amplitude', None)
    ylim_phase = plot_kwargs.pop('ylim_phase', None)
    y_label_amplitude = plot_kwargs.pop('y_label_amplitude', None)
    y_label_phase = plot_kwargs.pop('y_label_phase', None)

    # Plot amplitude bias
    lead_time_bias_subplot(
        lead_times=lead_times,
        bias_data=amplitude_bias,
        labels=labels,
        bias_type='amplitude',
        ax=ax1,
        title=titles[0],
        ylim=ylim_amplitude,
        y_label=y_label_amplitude,
        **plot_kwargs
    )

    # Plot phase bias
    lead_time_bias_subplot(
        lead_times=lead_times,
        bias_data=phase_bias,
        labels=labels,
        bias_type='phase',
        ax=ax2,
        title=titles[1],
        ylim=ylim_phase,
        y_label=y_label_phase,
        **plot_kwargs
    )

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, (ax1, ax2)


def lead_time_bias_subplot(
    lead_times: Sequence[np.ndarray],
    bias_data: Sequence[np.ndarray],
    labels: Sequence[str],
    bias_type: Literal['amplitude', 'phase'],
    ax: Optional[plt.Axes] = None,
    *,
    title: Optional[str] = None,
    x_label: Optional[str] = 'Lead time (days)',
    y_label: Optional[str] = None,
    show_legend: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = (0, 43),
    show_zero_line: bool = True,
    fontsize_title: int = 14,
    fontsize_axis_label: int = 12,
    fontsize_tick_label: int = 11,
    fontsize_legend: int = 11,
    fontweight_title: str = 'bold',
    fontweight_title_label: Optional[str] = None,
    font_family: Optional[str] = None,
    colormap: str = 'viridis',
    legend_loc: str = 'best',
) -> plt.Axes:
    """Plot a single bias subplot (amplitude or phase).

    Args:
        lead_times: Lead time arrays for each model
        bias_data: Bias values for each model
        labels: Model labels
        bias_type: Type of bias ('amplitude' or 'phase')
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label (defaults based on bias_type)
        show_legend: Whether to show legend
        ylim: Y-axis limits
        xlim: X-axis limits
        show_zero_line: Whether to show horizontal line at y=0
        fontsize_title: Font size for title
        fontsize_axis_label: Font size for axis labels
        fontsize_tick_label: Font size for tick labels
        fontsize_legend: Font size for legend
        fontweight_title: Font weight for title
        fontweight_title_label: Font weight for title label (e.g., "(a)")
        font_family: Font family
        colormap: Colormap name
        legend_loc: Legend location

    Returns:
        Matplotlib axes object
    """
    # Set default y_label based on bias type if not provided
    if y_label is None:
        if bias_type == 'amplitude':
            y_label = 'Amplitude bias'
        elif bias_type == 'phase':
            y_label = 'Phase bias (radians)'

    # Set default ylim if not provided
    if ylim is None:
        ylim = (-1, 0.5)

    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Generate colors
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(labels)))

    # Plot data
    for i, label in enumerate(labels):
        ax.plot(lead_times[i], bias_data[i], color=colors[i], label=label)

    # Add zero line if specified
    if show_zero_line:
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

    # Set font family if specified
    if font_family:
        plt.rcParams['font.family'] = font_family

    # Set labels and title
    label_kwargs = {'fontsize': fontsize_axis_label}
    if font_family:
        label_kwargs['family'] = font_family

    if x_label:
        ax.set_xlabel(x_label, **label_kwargs)
    if y_label:
        ax.set_ylabel(y_label, **label_kwargs)

    if title:
        # Check if we should bold only the label part (e.g., "(a)")
        import re
        label_match = re.match(r'^(\([a-zA-Z0-9]+\))\s*(.*)$', title)
        if label_match and fontweight_title_label == 'bold':
            label_part = label_match.group(1)
            text_part = label_match.group(2)
            title_str = f"$\\mathbf{{{label_part}}}$ {text_part}"
            title_kwargs = {'fontsize': fontsize_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(title_str, **title_kwargs)
        else:
            title_kwargs = {'fontsize': fontsize_title, 'fontweight': fontweight_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(title, **title_kwargs)

    # Set limits
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    # Configure x-axis ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x % 10 == 0 else ""))

    # Configure tick label size
    ax.tick_params(axis='both', labelsize=fontsize_tick_label)

    # Add legend
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=fontsize_legend)

    return ax


def lead_time_bias_amplitude_grid(
    lead_times: Sequence[np.ndarray],
    low_amp_amplitude_bias: Sequence[np.ndarray],
    low_amp_phase_bias: Sequence[np.ndarray],
    high_amp_amplitude_bias: Sequence[np.ndarray],
    high_amp_phase_bias: Sequence[np.ndarray],
    labels: Sequence[str],
    output_filename: Optional[str] = None,
    *,
    titles: Optional[Tuple[Optional[str], ...]] = (None, None, None, None),
    figsize: Tuple[float, float] = (16, 10),
    **plot_kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """Create 2x2 grid of amplitude-filtered bias plots.

    Grid layout:
        Row 1: Low amplitude - Amplitude bias (left), Phase bias (right)
        Row 2: High amplitude - Amplitude bias (left), Phase bias (right)

    Args:
        lead_times: Lead time arrays for each model
        low_amp_amplitude_bias: Low amplitude forecast amplitude bias
        low_amp_phase_bias: Low amplitude forecast phase bias
        high_amp_amplitude_bias: High amplitude forecast amplitude bias
        high_amp_phase_bias: High amplitude forecast phase bias
        labels: Model labels
        output_filename: Path to save figure
        titles: Tuple of 4 titles (low_amp, low_phase, high_amp, high_phase)
        figsize: Figure size
        **plot_kwargs: Additional kwargs passed to lead_time_bias_subplot
                       (can include ylim_amplitude and ylim_phase for separate y-limits)

    Returns:
        Figure and 2x2 array of axes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract amplitude and phase specific parameters if provided
    ylim_amplitude = plot_kwargs.pop('ylim_amplitude', None)
    ylim_phase = plot_kwargs.pop('ylim_phase', None)
    y_label_amplitude = plot_kwargs.pop('y_label_amplitude', None)
    y_label_phase = plot_kwargs.pop('y_label_phase', None)

    # Row 1: Low amplitude forecasts
    # Low amplitude - Amplitude bias (top-left)
    lead_time_bias_subplot(
        lead_times=lead_times,
        bias_data=low_amp_amplitude_bias,
        labels=labels,
        bias_type='amplitude',
        ax=axes[0, 0],
        title=titles[0],
        ylim=ylim_amplitude,
        y_label=y_label_amplitude,
        **plot_kwargs
    )

    # Low amplitude - Phase bias (top-right)
    lead_time_bias_subplot(
        lead_times=lead_times,
        bias_data=low_amp_phase_bias,
        labels=labels,
        bias_type='phase',
        ax=axes[0, 1],
        title=titles[1],
        ylim=ylim_phase,
        y_label=y_label_phase,
        **plot_kwargs
    )

    # Row 2: High amplitude forecasts
    # High amplitude - Amplitude bias (bottom-left)
    lead_time_bias_subplot(
        lead_times=lead_times,
        bias_data=high_amp_amplitude_bias,
        labels=labels,
        bias_type='amplitude',
        ax=axes[1, 0],
        title=titles[2],
        ylim=ylim_amplitude,
        y_label=y_label_amplitude,
        **plot_kwargs
    )

    # High amplitude - Phase bias (bottom-right)
    lead_time_bias_subplot(
        lead_times=lead_times,
        bias_data=high_amp_phase_bias,
        labels=labels,
        bias_type='phase',
        ax=axes[1, 1],
        title=titles[3],
        ylim=ylim_phase,
        y_label=y_label_phase,
        **plot_kwargs
    )

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, axes


def lead_time_bcor_bmse_amplitude_grid(
    lead_times: Sequence[np.ndarray],
    low_amp_correlations: Sequence[np.ndarray],
    low_amp_bmsea: Sequence[np.ndarray],
    low_amp_bmsep: Sequence[np.ndarray],
    high_amp_correlations: Sequence[np.ndarray],
    high_amp_bmsea: Sequence[np.ndarray],
    high_amp_bmsep: Sequence[np.ndarray],
    labels: Sequence[str],
    output_filename: Optional[str] = None,
    *,
    titles: Optional[Tuple[Optional[str], ...]] = (None, None, None, None),
    combined_bmse: bool = True,
    figsize: Tuple[float, float] = (16, 10),
    **plot_kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """Create 2x2 grid of amplitude-filtered BCOR and BMSE plots.

    Grid layout:
        Row 1: Low amplitude BCOR (left), Low amplitude BMSE (right)
        Row 2: High amplitude BCOR (left), High amplitude BMSE (right)

    Args:
        lead_times: Lead time arrays for each model
        low_amp_correlations: Low amplitude bivariate correlations
        low_amp_bmsea: Low amplitude BMSE amplitude errors
        low_amp_bmsep: Low amplitude BMSE phase errors
        high_amp_correlations: High amplitude bivariate correlations
        high_amp_bmsea: High amplitude BMSE amplitude errors
        high_amp_bmsep: High amplitude BMSE phase errors
        labels: Model labels
        output_filename: Path to save figure
        titles: Tuple of 4 titles (low_bcor, low_bmse, high_bcor, high_bmse)
        combined_bmse: Whether to plot combined BMSE
        figsize: Figure size
        **plot_kwargs: Additional kwargs passed to lead_time_metric_plot

    Returns:
        Figure and 2x2 array of axes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Compute combined BMSE if needed
    if combined_bmse:
        low_bmse_combined = [low_amp_bmsea[i] + low_amp_bmsep[i] for i in range(len(low_amp_bmsea))]
        high_bmse_combined = [high_amp_bmsea[i] + high_amp_bmsep[i] for i in range(len(high_amp_bmsea))]
    else:
        low_bmse_combined = None
        high_bmse_combined = None

    # Row 1: Low amplitude
    # Low amplitude BCOR (top-left)
    lead_time_skill_subplot(
        lead_times=lead_times,
        metric_data=low_amp_correlations,
        labels=labels,
        metric_type='bcor',
        ax=axes[0, 0],
        title=titles[0],
        **plot_kwargs
    )

    # Low amplitude BMSE (top-right)
    lead_time_skill_subplot(
        lead_times=lead_times,
        metric_data=low_bmse_combined if combined_bmse else low_amp_bmsea,
        labels=labels,
        metric_type='bmse',
        ax=axes[0, 1],
        title=titles[1],
        bmsea=low_amp_bmsea if not combined_bmse else None,
        bmsep=low_amp_bmsep if not combined_bmse else None,
        combined=combined_bmse,
        **plot_kwargs
    )

    # Row 2: High amplitude
    # High amplitude BCOR (bottom-left)
    lead_time_skill_subplot(
        lead_times=lead_times,
        metric_data=high_amp_correlations,
        labels=labels,
        metric_type='bcor',
        ax=axes[1, 0],
        title=titles[2],
        **plot_kwargs
    )

    # High amplitude BMSE (bottom-right)
    lead_time_skill_subplot(
        lead_times=lead_times,
        metric_data=high_bmse_combined if combined_bmse else high_amp_bmsea,
        labels=labels,
        metric_type='bmse',
        ax=axes[1, 1],
        title=titles[3],
        bmsea=high_amp_bmsea if not combined_bmse else None,
        bmsep=high_amp_bmsep if not combined_bmse else None,
        combined=combined_bmse,
        **plot_kwargs
    )

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, axes


def lag_correlation_plot(corrs: np.ndarray, timesteps: np.ndarray, variables: list, title: str, output_filename: str = None):
    colors = plt.cm.viridis(np.linspace(0, 1, len(variables)))

    for i,var in enumerate(variables):
        plt.plot(timesteps, corrs[i], linestyle='-', color=colors[i], label=var)
    plt.xlabel('History (days)')
    plt.ylabel('Correlation coeff')
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def bivariate_correlation_by_month_plot(
    bcor_dict: Dict[int, np.ndarray],
    output_filename: Optional[str] = None,
    *,
    title: Optional[str] = None,
    x_label: Optional[str] = 'Month',
    y_label: Optional[str] = 'Lead time (days)',
    colorbar_label: Optional[str] = 'Bivariate correlation',
    levels: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    cmap: str = "YlGnBu",
    figsize: Tuple[float, float] = (10, 6),
    fontsize_title: int = 14,
    fontsize_axis_label: int = 12,
    fontsize_tick_label: int = 11,
    fontsize_colorbar_label: int = 12,
    fontweight_title: str = 'bold',
    fontweight_title_label: Optional[str] = None,
    font_family: Optional[str] = None,
) -> plt.Figure:
    """Create a contour plot of bivariate correlation by month and lead time.

    Args:
        bcor_dict: Dictionary mapping month (1-12) to correlation arrays
        output_filename: Path to save figure
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        colorbar_label: Colorbar label
        levels: Contour levels (defaults to np.linspace(0.1, 1.0, 10))
        threshold: Threshold value for contour line
        cmap: Colormap name
        figsize: Figure size
        fontsize_title: Font size for title
        fontsize_axis_label: Font size for axis labels
        fontsize_tick_label: Font size for tick labels
        fontsize_colorbar_label: Font size for colorbar label
        fontweight_title: Font weight for title
        fontweight_title_label: Font weight for title label (e.g., "(a)")
        font_family: Font family

    Returns:
        Matplotlib figure object
    """
    if levels is None:
        levels = np.linspace(0.1, 1.0, 10)

    months = sorted(bcor_dict.keys())
    lead_times = len(next(iter(bcor_dict.values())))  # assume all months have same lead time length

    # Build metric matrix of shape (lead_time, 12)
    metric_array = np.stack([bcor_dict[m] for m in months], axis=-1)

    fig, ax = plt.subplots(figsize=figsize)

    # Set font family if specified
    if font_family:
        plt.rcParams['font.family'] = font_family

    # Contourf plot
    c = ax.contourf(months, np.arange(1, lead_times + 1), metric_array, levels=levels, cmap=cmap)

    # Contour line at threshold
    cs = ax.contour(months, np.arange(1, lead_times + 1), metric_array, levels=[threshold], colors=['blue', 'red'])
    ax.clabel(cs, fmt='%.1f')

    # Month labels
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Set labels and title
    label_kwargs = {'fontsize': fontsize_axis_label}
    if font_family:
        label_kwargs['family'] = font_family

    if x_label:
        ax.set_xlabel(x_label, **label_kwargs)
    if y_label:
        ax.set_ylabel(y_label, **label_kwargs)

    if title:
        # Check if we should bold only the label part (e.g., "(a)")
        import re
        label_match = re.match(r'^(\([a-zA-Z0-9]+\))\s*(.*)$', title)
        if label_match and fontweight_title_label == 'bold':
            label_part = label_match.group(1)
            text_part = label_match.group(2)
            title_str = f"$\\mathbf{{{label_part}}}$ {text_part}"
            title_kwargs = {'fontsize': fontsize_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(title_str, **title_kwargs)
        else:
            title_kwargs = {'fontsize': fontsize_title, 'fontweight': fontweight_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(title, **title_kwargs)

    # Configure tick label size
    ax.tick_params(axis='both', labelsize=fontsize_tick_label)

    # Colorbar (vertical on right side, with 1.0 at bottom)
    cbar = plt.colorbar(c, orientation='vertical', pad=0.05)
    cbar.ax.invert_yaxis()  # Reverse so 1.0 is at bottom
    if colorbar_label:
        colorbar_kwargs = {'fontsize': fontsize_colorbar_label}
        if font_family:
            colorbar_kwargs['family'] = font_family
        cbar.set_label(colorbar_label, **colorbar_kwargs)

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    return fig

def bivariate_correlation_vs_lead_time_heatmap(
    lead_times: Sequence[Sequence[Union[int, float]]],     # (N, T_i)
    lookbacks: Sequence[Sequence[Union[int, float]]],      # (N, L_i)
    correlations: Sequence[np.ndarray],                    # (N, L_i, T_i)
    labels: Sequence[str],                                 # (N,)
    output_filename: str = None,
    *,
    threshold: float = 0.5,
    bar_half_width: float = 0.35,  # half the horizontal length of each bar (x-index units)
    fuxi_correlations: Optional[np.ndarray] = None,        # (T_i,) or None
    # Font sizes
    fontsize_title: int = 14,
    fontsize_axis_label: int = 12,
    fontsize_tick_label: int = 11,
    fontsize_legend: int = 11,
    fontsize_colorbar_label: int = 12,
    fontweight_title: str = 'bold',
    fontweight_title_label: Optional[str] = None,
    font_family: Optional[str] = None,
    # Text labels
    x_label: str = "Lookback window (days)",
    y_label: str = "Lead time (days)",
    colorbar_label: str = "Bivariate correlation",
    legend_label_template: str = "{threshold} skill threshold",
    zero_symbol: str = r"$\varnothing$",
    fuxi_label: str = "FuXi",
    # Colormap settings
    cmap: str = "YlGnBu",
    color_bounds: Optional[Sequence[float]] = None,
    # Legend settings
    legend_loc: str = "upper right",
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    N heatmaps (one per model) with x=lookback (categories), y=lead time (categories).
    Optionally, if fuxi_correlations is provided, plot a narrow heatmap (1 column) in the middle.
    Shared Blues colorbar. For each lookback column, draw a short horizontal black bar
    at the first lead-time *index* where correlation falls below `threshold`
    (bar placed at the *lower edge* of that day cell). Y ticks are placed on the
    *upper edge* of cells so the max day label sits exactly at the top border.
    """
    from matplotlib import colors as mcolors

    def first_crossing_yindex(vals: np.ndarray, thr: float = 0.5) -> float:
        v = np.asarray(vals, dtype=float)
        if v[0] < thr:
            return 0.0
        for j in range(len(v) - 1):
            v0, v1 = v[j], v[j + 1]
            if (v0 >= thr) and (v1 < thr):
                if v1 == v0:
                    return float(j)
                frac = (thr - v0) / (v1 - v0)
                return float(j) + float(frac)
        return float("nan")

    N = len(labels)
    if not (len(lead_times) == len(lookbacks) == len(correlations) == N):
        raise ValueError("lead_times, lookbacks, correlations, and labels must all have length N.")

    has_fuxi = fuxi_correlations is not None
    if has_fuxi:
        fuxi_correlations = np.asarray(fuxi_correlations, dtype=float)
        if fuxi_correlations.ndim != 1:
            raise ValueError("fuxi_correlations should be a 1D array (T_i,), got shape {}".format(fuxi_correlations.shape))
    num_panels = N + (1 if has_fuxi else 0)

    # Key CHANGE: Specify widths so that the middle panel is very narrow if fuxi_correlations is given
    if has_fuxi:
        width = 0.15
        widths = [1] * (num_panels // 2) + [width] + [1] * (num_panels - 1 - num_panels // 2)
        total_width = sum(widths)
        figsize = (5.6 * (sum(widths)), 4.3)
        fig, axes = plt.subplots(1, num_panels, figsize=figsize, gridspec_kw={'width_ratios': widths}, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, num_panels, figsize=(5.6 * num_panels, 4.3), constrained_layout=True)
    if num_panels == 1:
        axes = [axes]
    
    # Set font family if specified
    if font_family:
        plt.rcParams['font.family'] = font_family

    # Use provided color bounds or default
    if color_bounds is None:
        bounds = np.arange(0.0, 1.0 + 0.1, 0.1)
    else:
        bounds = np.array(color_bounds)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=plt.get_cmap(cmap).N, clip=True)
    last_im = None

    axis_indices = list(range(num_panels))
    if has_fuxi:
        middle = num_panels // 2
        model_axes = [i for i in range(num_panels) if i != middle]
        fuxi_axis = middle
    else:
        model_axes = list(range(num_panels))
        fuxi_axis = None

    current_model = 0
    for i in range(num_panels):
        if has_fuxi and i == fuxi_axis:
            continue
        ax = axes[i]
        lab = labels[current_model]
        arr = np.asarray(correlations[current_model], dtype=float)
        leads = np.asarray(lead_times[current_model])
        looks = np.asarray(lookbacks[current_model])
        current_model += 1

        if arr.ndim != 2:
            raise ValueError(f"correlations[{current_model-1}] must be 2D, got {arr.shape}.")
        L_i, T_i = arr.shape
        if len(leads) != T_i:
            raise ValueError(f"lead_times[{current_model-1}] length {len(leads)} != correlations[{current_model-1}].shape[1] {T_i}.")
        if len(looks) != L_i:
            raise ValueError(f"lookbacks[{current_model-1}] length {len(looks)} != correlations[{current_model-1}].shape[0] {L_i}.")

        try:
            order_looks = np.argsort(looks.astype(float))
        except Exception:
            order_looks = np.argsort(looks.astype(str))
        looks_sorted = looks[order_looks]
        arr = arr[order_looks, :]

        try:
            order_leads = np.argsort(leads.astype(float))
        except Exception:
            order_leads = np.argsort(leads.astype(str))
        leads_sorted = leads[order_leads]
        arr = arr[:, order_leads]

        A = arr.T

        im = ax.imshow(A, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        last_im = im

        ax.set_xticks(np.arange(L_i))
        ax.set_xticklabels([zero_symbol if x == 0 else str(x) for x in looks_sorted], fontsize=fontsize_tick_label)

        step = 3
        yt_idx = np.arange(0, T_i, step)
        ax.set_yticks(yt_idx + 0.5)
        ax.set_yticklabels([str(leads_sorted[j]) for j in yt_idx], fontsize=fontsize_tick_label)

        label_kwargs = {'fontsize': fontsize_axis_label}
        if font_family:
            label_kwargs['family'] = font_family

        ax.set_xlabel(x_label, **label_kwargs)
        # Y-label and y-ticks: show only on the leftmost subplot
        if i == model_axes[0]:
            ax.set_ylabel(y_label, **label_kwargs)
        else:
            # Hide y-ticks and labels on all but the leftmost
            ax.yaxis.set_ticks_position('none')
            ax.yaxis.set_ticklabels([])
            ax.set_ylabel("")

        # Apply title with optional label bolding
        import re
        label_match = re.match(r'^(\([a-zA-Z0-9]+\))\s*(.*)$', lab)
        if label_match and fontweight_title_label == 'bold':
            label_part = label_match.group(1)
            text_part = label_match.group(2)
            title_str = f"$\\mathbf{{{label_part}}}$ {text_part}"
            title_kwargs = {'fontsize': fontsize_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(title_str, **title_kwargs)
        else:
            title_kwargs = {'fontsize': fontsize_title, 'fontweight': fontweight_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(lab, **title_kwargs)

        for c in range(L_i):
            ycross = first_crossing_yindex(A[:, c], thr=threshold)
            if np.isnan(ycross):
                continue
            y_low_edge = np.floor(ycross) - 0.5
            y_low_edge = max(-0.5, min(T_i - 0.5, y_low_edge))
            if ycross > 0:
                y_low_edge += 1
            ax.hlines(y_low_edge, c - bar_half_width, c + bar_half_width, colors="k", linewidth=1.3)

        bar_proxy = Line2D([0], [0], color="k", lw=1.3, label=legend_label_template.format(threshold=threshold))
        # Legend only for rightmost axis
        # For case with fuxi: the rightmost model_axis is max(model_axes); otherwise, it's N-1
        show_legend = (i == (model_axes[-1]))
        if show_legend:
            ax.legend(handles=[bar_proxy], loc=legend_loc, frameon=True, fancybox=True, framealpha=0.95, fontsize=fontsize_legend)

        ax.set_xlim(-0.5, L_i - 0.5)
        ax.set_ylim(-0.5, T_i - 0.5)

    # Optional fuxi panel: Draw in the middle
    if has_fuxi:
        ax = axes[fuxi_axis]
        T_i = fuxi_correlations.shape[0]
        fuxi_arr = fuxi_correlations
        fuxi_A = fuxi_arr.reshape(-1, 1)
        im = ax.imshow(
            fuxi_A, origin="lower", aspect="auto", cmap=cmap, norm=norm
        )
        last_im = im

        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")

        # Apply title with optional label bolding
        import re
        label_match = re.match(r'^(\([a-zA-Z0-9]+\))\s*(.*)$', fuxi_label)
        if label_match and fontweight_title_label == 'bold':
            label_part = label_match.group(1)
            text_part = label_match.group(2)
            title_str = f"$\\mathbf{{{label_part}}}$ {text_part}"
            title_kwargs = {'fontsize': fontsize_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(title_str, **title_kwargs)
        else:
            title_kwargs = {'fontsize': fontsize_title, 'fontweight': fontweight_title}
            if font_family:
                title_kwargs['family'] = font_family
            ax.set_title(fuxi_label, **title_kwargs)
        ax.set_xlabel("")

        ycross = first_crossing_yindex(fuxi_A[:, 0], thr=threshold)
        if not np.isnan(ycross):
            y_low_edge = np.floor(ycross) - 0.5
            y_low_edge = max(-0.5, min(T_i - 0.5, y_low_edge))
            if ycross > 0:
                y_low_edge += 1
            ax.hlines(y_low_edge, -bar_half_width, bar_half_width, colors="k", linewidth=1.3)

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, T_i - 0.5)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.9, pad=0.02, boundaries=bounds, ticks=bounds, spacing="proportional")
        cbar.set_label(colorbar_label, fontsize=fontsize_colorbar_label)
        cbar.ax.tick_params(labelsize=fontsize_tick_label)
        cbar.ax.invert_yaxis()

    if output_filename:
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")

    return fig, list(axes)

def hexbin_skill_vs_amplitude_plot(
    data: np.ndarray,                 # shape (T, N, 2) -> [:, :, 0]=amplitude, [:, :, 1]=skill
    output_filename: str,
    improvement = False,
    forecasted_amplitude = False,
    *,
    amp_range: Tuple[float, float] = (0.0, 3.5),
    skill_range: Tuple[float, float] = (-1.5, 1.5),
    gridsize: int = 60,
    boundaries: Optional[Sequence[int]] = None,  # discrete count bins for colorbar
    cmap: str = "YlGnBu",
):
    """
    Creates a 3-panel hexbin of skill vs amplitude for weeks 1+2, 3+4, 5+6.

    Weeks are defined as:
      - Weeks 1+2: leads 1–14  -> data[0:14]
      - Weeks 3+4: leads 15–28 -> data[14:28]
      - Weeks 5+6: leads 29–42 -> data[28:42]
    """
    assert data.ndim == 3 and data.shape[2] == 2, "Expected data with shape (T, N, 2)."
    T = data.shape[0]
    if T < 42:
        raise ValueError("Expected at least 42 lead times along axis 0.")

    # Default discrete boundaries tuned for counts mostly <30 but allowing up to ~120.
    # Edit this list if you want different steps.
    if boundaries is None:
        boundaries = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]  # Fibonacci-ish up to >120
    boundaries = np.asarray(boundaries, dtype=float)

    # Build a discrete norm using the provided boundaries
    base = plt.get_cmap("YlGnBu")
    cmap = ListedColormap(base(np.linspace(0.15, 1.0, 256)))  # truncate light end

    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)

    # Week slices in 0-based indexing for leads 1..42
    week_slices = [(0, 14), (14, 28), (28, 42)]
    titles = ["Weeks 1-2", "Weeks 3-4", "Weeks 5-6"]

    fig = plt.figure(figsize=(13.5, 4.2))
    gs  = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)

    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    axes.append(fig.add_subplot(gs[0, 2], sharey=axes[0]))
    cax  = fig.add_subplot(gs[0, 3])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    hexbins = []
    counts_per_panel = []


    for ax, (start, end), title in zip(axes, week_slices, titles):
        # Extract amplitude & skill across the selected leads and flatten
        panel = data[start:end, :, :]  # shape (leads_window, N, 2)
        amp = panel[..., 0].ravel()
        skill = panel[..., 1].ravel()

        # Filter finite values and within the plotting ranges
        m = np.isfinite(amp) & np.isfinite(skill)
        amp, skill = amp[m], skill[m]

        # Clip to plotting range so extreme outliers don’t skew autoscaling of hexbin
        in_range = (amp_range[0] <= amp) & (amp <= amp_range[1]) & \
                   (skill_range[0] <= skill) & (skill <= skill_range[1])
        amp, skill = amp[in_range], skill[in_range]
        counts_per_panel.append(amp.size)

        hb = ax.hexbin(
            amp, skill,
            gridsize=gridsize,
            extent=(amp_range[0], amp_range[1], skill_range[0], skill_range[1]),
            mincnt=1,            # only color cells with >=1 point; empty cells stay background
            cmap=cmap,
            norm=norm,           # discrete bins
            linewidths=0.0
        )
        hexbins.append(hb)

        ax.set_title(title)
        ax.set_xlim(amp_range)
        ax.set_ylim(skill_range)
        ax.set_xlabel("Forecasted amplitude" if forecasted_amplitude else "Initial amplitude")


    xticks = np.arange(amp_range[0], amp_range[1], 0.5)
    for ax in axes:
        ax.set_xticks(xticks)
    
    axes[0].set_ylabel("Improvement in bivariate correlation" if improvement else 'Bivariate correlation')


    cb = fig.colorbar(hexbins[0], cax=cax, boundaries=boundaries, spacing="proportional")
    cb.set_label("Count")
    # Pick readable tick labels (every other boundary except the first)
    cb.set_ticks(boundaries[[4,6,7,8,9,10]])
    fig.tight_layout()
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def histogram_skill_vs_phase_plot(
    data: np.ndarray,                 # shape (Y, T=42, N, 2): [:,:,:,0]=phase, [:,:,:,1]=skill
    output_filename: str,
    labels,                           # e.g., ['2019','2020','2021'] length = Y
    improvement = False,
    forecasted_phase = False,
    *,
    phase_values=tuple(range(1, 9)),  # phases 1..8
    figsize=(12, 4),
    bar_alpha=0.95,
    skill_range=(0, 1),
    cmap_name="YlGnBu",
    legend_position="bottom"          # "bottom" (default) or "right"
):
    assert data.ndim == 4 and data.shape[-1] == 2, "data must be (Y, T, N, 2)"
    Y, T, N, _ = data.shape
    assert T >= 42, "Expected at least 42 lead times"
    assert len(labels) == Y, "labels length must match Y"

    week_groups = [
        ("Weeks 1-2", 1, 14),
        ("Weeks 3-4", 15, 28),
        ("Weeks 5-6", 29, 42),
    ]

    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(v) for v in np.linspace(0.25, 0.9, Y)] if Y > 1 else [cmap(0.7)]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    x = np.arange(len(phase_values), dtype=float)

    n_years = Y
    total_width = 0.85
    bar_width = total_width / n_years
    offsets = (np.arange(n_years) - (n_years - 1) / 2) * bar_width

    # draw bars
    for ax, (title, start_lt, end_lt) in zip(axes, week_groups):
        lt_slice = slice(start_lt - 1, end_lt)

        for y_idx in range(Y):
            phases = data[y_idx, lt_slice, :, 0].reshape(-1)
            skills = data[y_idx, lt_slice, :, 1].reshape(-1)

            valid = np.isfinite(phases) & np.isfinite(skills)
            phases_i = phases[valid].astype(int)
            skills_v = skills[valid]

            means = np.full(len(phase_values), np.nan)
            for i, p in enumerate(phase_values):
                m = phases_i == p
                if np.any(m):
                    means[i] = np.nanmean(skills_v[m])

            means_plot = np.where(np.isnan(means), 0.0, means)

            ax.bar(
                x + offsets[y_idx],
                means_plot,
                width=bar_width * 0.95,
                alpha=bar_alpha,
                label=labels[y_idx],          # keep labels here; we'll grab them for fig.legend
                color=colors[y_idx],
                edgecolor='black',
                linewidth=0.6,
            )

        ax.set_title(title)
        ax.set_xlabel("Forecasted phase" if forecasted_phase else "Initial phase")
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in phase_values])
        ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_ylim(*skill_range)

    axes[0].set_ylabel("Improvement in bivariate correlation" if improvement else 'Bivariate correlation')

    # ----- Figure-level legend placement -----
    handles, leg_labels = axes[0].get_legend_handles_labels()
    # remove per-axes legends
    for ax in axes:
        ax.legend_.remove() if ax.get_legend() else None

    if legend_position == "right":
        # make space on the right
        plt.subplots_adjust(right=0.84)
        fig.legend(
            handles, leg_labels,
            loc="center left",
            bbox_to_anchor=(0.86, 0.5),
            frameon=False,
        )
    else:  # "bottom"
        # make space at the bottom
        plt.subplots_adjust(bottom=0.30)
        fig.legend(
            handles, leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=min(Y, 6),
            frameon=False,
        )

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def bivariate_correlation_by_month_multi_year_plot(
    bcor_dicts,                          # list[dict[int->array_like]] length=3; each dict: {month(1..12): (lead_time,)}
    label: str,
    year_titles=None,                     # optional list[str] length=3 for subplot titles (e.g., ["2019","2020","2021"])
    output_filename: str = None,
    *,
    levels = np.linspace(0.1, 1.0, 10),   # discretized colorbar levels (inclusive of min/max for nice bins)
    threshold: float = 0.5,
    cmap: str = "YlGnBu",                 # “YlGnBu” is the Matplotlib name
    colorbar_orientation: str = "horizontal"
):
    """
    Makes 3 subplots (one per year) of bivariate correlation (skill) as a function of month (x) and lead time (y),
    with a shared, discretized colorbar in YlGnBu. Draws a contour line at `threshold`.
    """
    assert len(bcor_dicts) == 3, "Pass exactly 3 dicts (one per year)."

    # Consistent month order and lead-time length
    months = list(range(1, 13))
    lead_times = len(next(iter(bcor_dicts[0].values())))
    for d in bcor_dicts:
        assert all(m in d for m in months), "Each dict must have months 1..12."
        assert all(len(d[m]) == lead_times for m in months), "All months must share the same lead-time length."

    # Stack each year's metric into (lead_time, 12)
    metric_arrays = [np.stack([d[m] for m in months], axis=-1) for d in bcor_dicts]

    # Discretization: shared bounds across all subplots
    vmin, vmax = float(np.min(levels)), float(np.max(levels))
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Plot each year
    contour_sets = []
    for i, (ax, arr) in enumerate(zip(axes, metric_arrays)):
        c = ax.contourf(
            months,
            np.arange(1, lead_times + 1),
            arr,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        contour_sets.append(c)

        # Threshold contour line (drawn over the filled contours)
        cs = ax.contour(
            months,
            np.arange(1, lead_times + 1),
            arr,
            levels=[threshold],
            colors="black",
            linewidths=1.0,
        )
        ax.clabel(cs, fmt="%.2f", inline=True, fontsize=11)

        # Month ticks & labels
        ax.set_xticks(months)
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=0)

        # Titles per subplot
        if year_titles and len(year_titles) == 3:
            ax.set_title(year_titles[i])
        else:
            ax.set_title(f"Year {i+1}")

        # Y label only on the first subplot
        if i == 0:
            ax.set_ylabel("Lead time (days)")

    # Shared discretized colorbar
    # Use the last contour set; all share the same norm/levels so any works.
    cbar = fig.colorbar(
        contour_sets[-1],
        ax=axes,
        orientation=colorbar_orientation,
        pad=0.4 if colorbar_orientation == "horizontal" else 0.02,
        fraction=0.05,
        aspect=40,
        ticks=levels
    )
    cbar.set_label(label)

    # Nice layout
    plt.tight_layout(rect=(0, 0.195, 1, 0.95))  # leave bottom space for the bar

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def scatter_amplitudes_by_init_date_tripanel(
    data,                         # (T, 42, 2): [...,0]=amplitude, [...,1]=correct {0,1}
    dates,                        # length T
    *,
    colors_cmap="YlGnBu",
    point_size=18,
    super_title="FuXi-S2S amplitudes by initialization date",
    x_label="Forecasted amplitude",
    y_label="Initialization date",
    sort_by_date=True,
    invert_y=False,
    jitter_days=0.0,              # e.g., 0.0003 ≈ 26 seconds
    y_tick_stride=1,
    output_filename=None
):
    data   = np.asarray(data)
    assert data.ndim == 3 and data.shape[1] == 42 and data.shape[2] == 2, "data must be (T,42,2)"
    T      = data.shape[0]
    assert len(dates) == T, "len(dates) must equal T"

    amps       = data[..., 0]
    is_correct = data[..., 1].astype(bool)

    dates_np = np.array(pd.to_datetime(dates)).astype("datetime64[ns]")
    if sort_by_date:
        order = np.argsort(dates_np)
        amps, is_correct, dates_np = amps[order], is_correct[order], dates_np[order]

    dates_dt = pd.to_datetime(dates_np).to_pydatetime()
    y_num    = mdates.date2num(dates_dt)

    def jitter(y):
        if jitter_days <= 0: return y
        return y + np.random.uniform(-jitter_days, jitter_days, size=y.shape)

    cmap = plt.get_cmap(colors_cmap)
    # 6 colors for 6 weeks, spaced through the colormap
    panel_colors = [cmap(x) for x in np.linspace(0.95, 0.45, 6)]
    panel_titles = [f"Week {i+1}" for i in range(6)]
    week_bins = [(i*7+1, (i+1)*7) for i in range(6)]  # [(1,7), (8,14), ..., (36,42)]

    fig, axes = plt.subplots(1, 6, figsize=(28, 8), sharey=True)

    for ax, (lo, hi), color, title in zip(axes, week_bins, panel_colors, panel_titles):
        cols  = slice(lo-1, hi)
        X     = amps[:, cols]                      # (T, ncols)
        Y     = np.repeat(y_num[:, None], X.shape[1], axis=1)
        masks = {True: is_correct[:, cols], False: ~is_correct[:, cols]}
        styles = {True: dict(marker='o', edgecolors="none", linewidths=max(0.5, point_size/12), alpha=1.0),
                  False: dict(marker='x', linewidths=max(1.2, point_size/8), alpha=0.6)}

        for ok in (True, False):
            m = masks[ok]
            if m.any():
                ax.scatter(
                    X[m], jitter(Y)[m],
                    s=point_size, c=[color], zorder=2 if ok else 3,
                    **styles[ok], label=("Correct" if ok else "Incorrect")
                )

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_xlim(0, 3)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.8, alpha=0.6)

    # Y-axis labels/ticks (left panel only; sharey=True keeps alignment)
    axes[0].set_ylabel(y_label)
    yticks  = y_num[::y_tick_stride] if y_tick_stride > 1 else y_num
    ylabels = [d.strftime("%Y-%m-%d") for d in (np.array(dates_dt)[::y_tick_stride] if y_tick_stride > 1 else dates_dt)]
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(ylabels)
    axes[0].yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    if invert_y:
        axes[0].invert_yaxis()

    fig.suptitle(super_title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_filename:
        fig.savefig(output_filename, bbox_inches="tight", dpi=300)
    else:
        plt.show()