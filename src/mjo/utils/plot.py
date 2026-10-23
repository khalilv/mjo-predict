import math
import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from typing import Sequence, Tuple, List, Union, Optional, Literal, Dict, Any
import matplotlib.patches as mpatches


# =============================================================================
# Shared helpers
# =============================================================================

def _set_title(ax, title, fontsize_title, fontweight_title, fontweight_title_label, font_family):
    """Set an axes title, bolding only a leading '(a)'-style label when fontweight_title_label is 'bold'."""
    label_match = re.match(r'^(\([a-zA-Z0-9]+\))\s*(.*)$', title)
    if label_match and fontweight_title_label == 'bold':
        title_str = f"$\\mathbf{{{label_match.group(1)}}}$ {label_match.group(2)}"
        title_kwargs = {'fontsize': fontsize_title}
    else:
        title_str = title
        title_kwargs = {'fontsize': fontsize_title, 'fontweight': fontweight_title}
    if font_family:
        title_kwargs['family'] = font_family
    ax.set_title(title_str, **title_kwargs)


def _save_and_close(output_filename):
    """Save the current figure at 600 dpi and close it, if a filename is given."""
    if output_filename:
        plt.savefig(output_filename, dpi=600, bbox_inches='tight')
        plt.close()


def correlation_scatter_plot(pred_rmm1, gt_rmm1, pred_rmm2, gt_rmm2, pred_amplitude, gt_amplitude, pred_label = None, gt_label = None, output_filename = None):
    """Scatter predicted vs ground truth RMM1, RMM2 and amplitude (shows the plot if no output_filename)."""
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
        plt.savefig(output_filename, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# Phase space
# =============================================================================

def _add_region_labels(xlim, ylim, frac, fontsize_region_label, font_family):
    """Label the four MJO regions at frac of the axis limits."""
    x_pos = xlim[1] * frac
    y_pos = ylim[1] * frac
    font_props = {'fontsize': fontsize_region_label, 'ha': 'center', 'va': 'center'}
    if font_family:
        font_props['fontfamily'] = font_family

    plt.text(0, -y_pos, "Indian Ocean", **font_props)
    plt.text(x_pos, 0, "Maritime Continent", rotation=-90, **font_props)
    plt.text(0, y_pos, "Western Pacific", **font_props)
    plt.text(-x_pos, 0, "Western Hem.& Africa", rotation=90, **font_props)


def _draw_phase_wedges(xlim, ylim, lw, fontsize_phase_label, font_family):
    """Draw the eight phase boundary lines and P1-P8 labels."""
    # Scale radius based on xlim/ylim (use average of the two)
    max_lim = (xlim[1] + ylim[1]) / 2
    r = math.sqrt(2 * max_lim * max_lim)
    phase_labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

    for i in range(8):
        angle = (5 + i) * np.pi / 4
        x = np.cos(angle)
        y = np.sin(angle)
        plt.plot([0, r * x], [0, r * y], color='lightgray', lw=lw, zorder=0)

    font_props = {'fontsize': fontsize_phase_label, 'ha': 'center', 'va': 'center', 'color': 'gray'}
    if font_family:
        font_props['fontfamily'] = font_family

    for i, label in enumerate(phase_labels):
        angle = (4.5 + i) * np.pi / 4
        x = max_lim * np.cos(angle)
        y = max_lim * np.sin(angle)
        plt.text(x, y, label, **font_props)


def _phase_space_axis_labels_and_title(ax, title, fontsize_title, fontsize_axis_label, fontweight_title, fontweight_title_label, font_family):
    """Set RMM1/RMM2 axis labels and the title, drawing a leading '(a)' label separately if fontweight_title_label is set."""
    font_props = {'fontsize': fontsize_axis_label}
    if font_family:
        font_props['fontfamily'] = font_family
    plt.xlabel("RMM1", **font_props)
    plt.ylabel("RMM2", **font_props)

    if not title:
        return
    if title.startswith('(') and ')' in title[:5] and fontweight_title_label:
        close_paren = title.index(')')
        label_part = title[:close_paren+1]
        rest_part = title[close_paren+1:].strip()
        plt.title(label_part + ' ' + rest_part, fontsize=fontsize_title,
                 fontweight=fontweight_title, pad=10)
        # Replace the title with separately weighted label and text
        ax.title.set_text('')
        ax.text(0.5, 1.02, label_part, transform=ax.transAxes,
               fontsize=fontsize_title, fontweight=fontweight_title_label,
               ha='center', va='bottom')
        ax.text(0.5 + len(label_part)*0.012, 1.02, rest_part, transform=ax.transAxes,
               fontsize=fontsize_title, fontweight=fontweight_title,
               ha='left', va='bottom')
    else:
        plt.title(title, fontsize=fontsize_title, fontweight=fontweight_title, pad=10)


def _finish_phase_space(xlim, ylim, legend_loc, fontsize_legend, output_filename, legend_order=None):
    """Apply limits and equal aspect, add a de-duplicated (optionally ordered) legend, then save."""
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect('equal')

    # prevent duplicate legend entries
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    if legend_order:
        # Listed labels first, the rest in plotting order
        ordered = {lbl: unique[lbl] for lbl in legend_order if lbl in unique}
        for lbl, h in unique.items():
            ordered.setdefault(lbl, h)
        unique = ordered
    plt.legend(unique.values(), unique.keys(), loc=legend_loc, fontsize=fontsize_legend)

    plt.tight_layout()
    _save_and_close(output_filename)


def phase_space_plot(
    trajectories: List[Dict[str, Any]],
    title: Optional[str] = None,
    output_filename: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (8, 8),
    xlim: Tuple[float, float] = (-4, 4),
    ylim: Tuple[float, float] = (-4, 4),
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
    legend_order: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot RMM1 vs RMM2 phase-space trajectories.

    Each trajectory dict needs 'rmm1', 'rmm2', 'label', 'color' and optionally 'linestyle',
    'alpha' (0.85) and 'marker_start' (True); trajectories are drawn in order. legend_order
    lists labels to show first in the legend (others follow in plotting order).
    """
    if font_family:
        plt.rcParams['font.family'] = font_family

    fig = plt.figure(figsize=figsize)

    ax = plt.gca()
    # Draw a unit circle in light gray
    unit_circle = mpatches.Circle((0, 0), 1, edgecolor='lightgray', facecolor='none', linewidth=0.8, linestyle='-', zorder=0)
    ax.add_patch(unit_circle)

    for traj in trajectories:
        rmm1 = traj['rmm1']
        rmm2 = traj['rmm2']
        color = traj['color']
        plt.plot(rmm1, rmm2, color=color, linestyle=traj.get('linestyle', '-'), alpha=traj.get('alpha', 0.85), label=traj['label'])
        if traj.get('marker_start', True) and len(rmm1) > 0:
            plt.plot(rmm1[0], rmm2[0], color=color, marker='o')
        for lt in range(marker_interval, len(rmm1), marker_interval):
            plt.plot(rmm1[lt], rmm2[lt], color=color, marker='.')

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    _draw_phase_wedges(xlim, ylim, 0.8, fontsize_phase_label, font_family)
    _add_region_labels(xlim, ylim, 0.875, fontsize_region_label, font_family)
    _phase_space_axis_labels_and_title(ax, title, fontsize_title, fontsize_axis_label, fontweight_title, fontweight_title_label, font_family)
    _finish_phase_space(xlim, ylim, legend_loc, fontsize_legend, output_filename, legend_order)

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
    """Plot per-phase composite trajectories; each composite maps init phase (1-8) to (rmm1, rmm2) means."""
    if font_family:
        plt.rcParams['font.family'] = font_family

    def _plot_composite(composite, color, label, **line_kwargs):
        """Draw every phase trajectory of one composite with start and interval markers."""
        for _, (rmm1, rmm2) in composite.items():
            plt.plot(rmm1, rmm2, color=color, label=label, linewidth=2, **line_kwargs)
            plt.plot(rmm1[0], rmm2[0], color=color, marker='o')  # start
            for lt in range(marker_interval, len(rmm1), marker_interval):
                plt.plot(rmm1[lt], rmm2[lt], color=color, marker='.')

    assert len(labels) == len(pred_composites), 'Number of labels must match number of prediction sources'
    fig = plt.figure(figsize=figsize)

    ax = plt.gca()
    # Draw a unit circle in light gray
    unit_circle = mpatches.Circle((0, 0), 1, edgecolor='lightgray', facecolor='none', linewidth=1.5, linestyle='-', zorder=0)
    ax.add_patch(unit_circle)

    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(labels)))
    for i, label in enumerate(labels):
        _plot_composite(pred_composites[i], colors[i], label, alpha=0.8)
    _plot_composite(gt_composites, 'black', gt_label, linestyle='--', alpha=0.75)
    if bom_composites is not None:
        _plot_composite(bom_composites, 'lightgray', 'BoM', linestyle='--', alpha=0.75)

    plt.axhline(0, color='lightgray', linewidth=1.5, zorder=0)
    plt.axvline(0, color='lightgray', linewidth=1.5, zorder=0)
    _draw_phase_wedges(xlim, ylim, 1.5, fontsize_phase_label, font_family)
    _add_region_labels(xlim, ylim, 0.857, fontsize_region_label, font_family)  # 1.5/1.75
    _phase_space_axis_labels_and_title(ax, title, fontsize_title, fontsize_axis_label, fontweight_title, fontweight_title_label, font_family)
    _finish_phase_space(xlim, ylim, legend_loc, fontsize_legend, output_filename)

    return fig


# =============================================================================
# Lead-time line plots
# =============================================================================

def _plot_lead_time_lines(ax, lead_times, data, labels, colors, linestyles, linewidth, significance):
    """Plot one line per model; models with a significance mask are faded where not significant."""
    nonsig_alpha = 0.25
    for i, label in enumerate(labels):
        if significance is not None and i < len(significance):
            # Full line at low alpha as base (no label; legend uses a proxy)
            ax.plot(lead_times[i], data[i], color=colors[i], linestyle=linestyles[i], linewidth=linewidth, alpha=nonsig_alpha)
            # Overdraw significant segments at full opacity
            if significance[i].any():
                y_sig = np.where(significance[i], data[i], np.nan)
                ax.plot(lead_times[i], y_sig, color=colors[i], linestyle=linestyles[i], linewidth=linewidth, alpha=1.0)
            # Full-opacity legend proxy
            ax.plot([], [], color=colors[i], linestyle=linestyles[i], linewidth=linewidth, alpha=1.0, label=label)
        else:
            ax.plot(lead_times[i], data[i], color=colors[i], linestyle=linestyles[i], label=label, linewidth=linewidth)


def _style_lead_time_axis(
    ax, *, title, x_label, y_label, xlim, ylim, show_legend, legend_loc,
    fontsize_title, fontsize_axis_label, fontsize_tick_label, fontsize_legend,
    fontweight_title, fontweight_title_label, font_family,
):
    """Apply labels, title, limits, 10-day x tick labels and legend to a lead-time axis."""
    if font_family:
        plt.rcParams['font.family'] = font_family

    label_kwargs = {'fontsize': fontsize_axis_label}
    if font_family:
        label_kwargs['family'] = font_family
    if x_label:
        ax.set_xlabel(x_label, **label_kwargs)
    if y_label:
        ax.set_ylabel(y_label, **label_kwargs)
    if title:
        _set_title(ax, title, fontsize_title, fontweight_title, fontweight_title_label, font_family)

    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    # Tick every day, label every 10 days
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x % 10 == 0 else ""))
    ax.tick_params(axis='both', labelsize=fontsize_tick_label)

    if show_legend:
        ax.legend(loc=legend_loc, fontsize=fontsize_legend)


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
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    linewidth: float = 2,
    legend_loc: str = 'best',
    # BMSE-specific parameters
    bmsea: Optional[Sequence[np.ndarray]] = None,
    bmsep: Optional[Sequence[np.ndarray]] = None,
    combined: bool = False,
    # Significance coloring: True at lead times where the difference vs baseline is significant
    significance: Optional[List[np.ndarray]] = None,
) -> plt.Axes:
    """Plot BCOR or BMSE versus lead time on one axis (BMSE as given in metric_data unless separate bmsea/bmsep with combined=False)."""
    # Defaults based on metric type
    if y_label is None:
        if metric_type == 'bcor':
            y_label = 'Bivariate correlation'
        elif metric_type == 'bmse':
            y_label = 'Bivariate mean squared error'
    if ylim is None:
        if metric_type == 'bcor':
            ylim = (0.3, 1.0)
        elif metric_type == 'bmse':
            ylim = (0, 2.0)
    if threshold is None and metric_type == 'bcor':
        threshold = 0.5
    if threshold_style is None:
        threshold_style = {'color': 'gray', 'linestyle': '--', 'linewidth': 1.5}

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    if colors is None:
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(labels)))
    if linestyles is None:
        linestyles = ['-'] * len(labels)

    if metric_type == 'bcor' or (metric_type == 'bmse' and (combined or (bmsea is None and bmsep is None))):
        _plot_lead_time_lines(ax, lead_times, metric_data, labels, colors, linestyles, linewidth, significance)
    elif metric_type == 'bmse':
        assert bmsea is not None and bmsep is not None, 'Must provide bmsea and bmsep for non-combined BMSE plot'
        for i, label in enumerate(labels):
            ax.plot(lead_times[i], bmsea[i], color=colors[i], linestyle=linestyles[i], label=f'{label} (Amplitude)', linewidth=linewidth)
            ax.plot(lead_times[i], bmsep[i], color=colors[i], linestyle='--', label=f'{label} (Phase)', linewidth=linewidth)

    if threshold is not None:
        ax.axhline(threshold, **threshold_style)

    _style_lead_time_axis(
        ax, title=title, x_label=x_label, y_label=y_label, xlim=xlim, ylim=ylim,
        show_legend=show_legend, legend_loc=legend_loc,
        fontsize_title=fontsize_title, fontsize_axis_label=fontsize_axis_label,
        fontsize_tick_label=fontsize_tick_label, fontsize_legend=fontsize_legend,
        fontweight_title=fontweight_title, fontweight_title_label=fontweight_title_label,
        font_family=font_family,
    )
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
    bcor_significance: Optional[List[np.ndarray]] = None,
    bmse_significance: Optional[List[np.ndarray]] = None,
    **plot_kwargs
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot BCOR (left) and BMSE (right) versus lead time; plot_kwargs go to lead_time_skill_subplot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if combined_bmse:
        bmse_combined = [bmsea[i] + bmsep[i] for i in range(len(bmsea))]
    else:
        bmse_combined = None

    lead_time_skill_subplot(
        lead_times=lead_times,
        metric_data=correlations,
        labels=labels,
        metric_type='bcor',
        ax=ax1,
        title=titles[0],
        significance=bcor_significance,
        **plot_kwargs
    )
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
        significance=bmse_significance,
        **plot_kwargs
    )

    plt.tight_layout()
    _save_and_close(output_filename)
    return fig, (ax1, ax2)


def lead_time_bmse_decomp_plot(
    lead_times: Sequence[np.ndarray],
    bmsea: Sequence[np.ndarray],
    bmsep: Sequence[np.ndarray],
    labels: Sequence[str],
    output_filename: Optional[str] = None,
    *,
    titles: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
    figsize: Tuple[float, float] = (16, 5),
    ylim_amplitude: Optional[Tuple[float, float]] = None,
    ylim_phase: Optional[Tuple[float, float]] = None,
    amplitude_significance: Optional[List[np.ndarray]] = None,
    phase_significance: Optional[List[np.ndarray]] = None,
    **plot_kwargs
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot the BMSE amplitude-error (BMSEa, left) and phase-error (BMSEp, right) components versus lead time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    lead_time_skill_subplot(
        lead_times=lead_times,
        metric_data=bmsea,
        labels=labels,
        metric_type='bmse',
        ax=ax1,
        title=titles[0],
        combined=True,
        ylim=ylim_amplitude,
        significance=amplitude_significance,
        **plot_kwargs
    )
    lead_time_skill_subplot(
        lead_times=lead_times,
        metric_data=bmsep,
        labels=labels,
        metric_type='bmse',
        ax=ax2,
        title=titles[1],
        combined=True,
        ylim=ylim_phase,
        significance=phase_significance,
        **plot_kwargs
    )

    plt.tight_layout()
    _save_and_close(output_filename)
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
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    linewidth: float = 2,
    legend_loc: str = 'best',
    significance: Optional[List[np.ndarray]] = None,
) -> plt.Axes:
    """Plot amplitude or phase bias versus lead time on one axis."""
    if y_label is None:
        if bias_type == 'amplitude':
            y_label = 'Amplitude bias'
        elif bias_type == 'phase':
            y_label = 'Phase bias (radians)'
    if ylim is None:
        ylim = (-1, 0.5)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    if colors is None:
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(labels)))
    if linestyles is None:
        linestyles = ['-'] * len(labels)

    _plot_lead_time_lines(ax, lead_times, bias_data, labels, colors, linestyles, linewidth, significance)

    if show_zero_line:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=0)

    _style_lead_time_axis(
        ax, title=title, x_label=x_label, y_label=y_label, xlim=xlim, ylim=ylim,
        show_legend=show_legend, legend_loc=legend_loc,
        fontsize_title=fontsize_title, fontsize_axis_label=fontsize_axis_label,
        fontsize_tick_label=fontsize_tick_label, fontsize_legend=fontsize_legend,
        fontweight_title=fontweight_title, fontweight_title_label=fontweight_title_label,
        font_family=font_family,
    )
    return ax


def _pop_bias_panel_kwargs(plot_kwargs):
    """Split amplitude- and phase-panel kwargs (ylim_*, y_label_*) out of plot_kwargs."""
    amplitude_kwargs = {'ylim': plot_kwargs.pop('ylim_amplitude', None), 'y_label': plot_kwargs.pop('y_label_amplitude', None)}
    phase_kwargs = {'ylim': plot_kwargs.pop('ylim_phase', None), 'y_label': plot_kwargs.pop('y_label_phase', None)}
    return amplitude_kwargs, phase_kwargs


def lead_time_bias_plot(
    lead_times: Sequence[np.ndarray],
    amplitude_bias: Sequence[np.ndarray],
    phase_bias: Sequence[np.ndarray],
    labels: Sequence[str],
    output_filename: Optional[str] = None,
    *,
    titles: Optional[Tuple[Optional[str], Optional[str]]] = (None, None),
    figsize: Tuple[float, float] = (16, 5),
    amplitude_significance: Optional[List[np.ndarray]] = None,
    phase_significance: Optional[List[np.ndarray]] = None,
    **plot_kwargs
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot amplitude (left) and phase (right) bias versus lead time; accepts ylim_/y_label_ amplitude/phase kwargs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    amplitude_kwargs, phase_kwargs = _pop_bias_panel_kwargs(plot_kwargs)

    lead_time_bias_subplot(
        lead_times=lead_times,
        bias_data=amplitude_bias,
        labels=labels,
        bias_type='amplitude',
        ax=ax1,
        title=titles[0],
        significance=amplitude_significance,
        **amplitude_kwargs,
        **plot_kwargs
    )
    lead_time_bias_subplot(
        lead_times=lead_times,
        bias_data=phase_bias,
        labels=labels,
        bias_type='phase',
        ax=ax2,
        title=titles[1],
        significance=phase_significance,
        **phase_kwargs,
        **plot_kwargs
    )

    plt.tight_layout()
    _save_and_close(output_filename)
    return fig, (ax1, ax2)


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
    """Plot a 2x2 grid of amplitude/phase bias (columns) for low/high-amplitude forecasts (rows)."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    amplitude_kwargs, phase_kwargs = _pop_bias_panel_kwargs(plot_kwargs)

    panels = [
        (axes[0, 0], low_amp_amplitude_bias, 'amplitude', amplitude_kwargs),
        (axes[0, 1], low_amp_phase_bias, 'phase', phase_kwargs),
        (axes[1, 0], high_amp_amplitude_bias, 'amplitude', amplitude_kwargs),
        (axes[1, 1], high_amp_phase_bias, 'phase', phase_kwargs),
    ]
    for (ax, bias_data, bias_type, panel_kwargs), title in zip(panels, titles):
        lead_time_bias_subplot(
            lead_times=lead_times,
            bias_data=bias_data,
            labels=labels,
            bias_type=bias_type,
            ax=ax,
            title=title,
            **panel_kwargs,
            **plot_kwargs
        )

    plt.tight_layout()
    _save_and_close(output_filename)
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
    low_amp_bcor_significance: Optional[List[np.ndarray]] = None,
    low_amp_bmse_significance: Optional[List[np.ndarray]] = None,
    high_amp_bcor_significance: Optional[List[np.ndarray]] = None,
    high_amp_bmse_significance: Optional[List[np.ndarray]] = None,
    **plot_kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot a 2x2 grid of BCOR/BMSE (columns) for low/high-amplitude forecasts (rows)."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    rows = [
        (low_amp_correlations, low_amp_bmsea, low_amp_bmsep, low_amp_bcor_significance, low_amp_bmse_significance),
        (high_amp_correlations, high_amp_bmsea, high_amp_bmsep, high_amp_bcor_significance, high_amp_bmse_significance),
    ]
    for r, (correlations, bmsea, bmsep, bcor_sig, bmse_sig) in enumerate(rows):
        bmse_combined = [bmsea[i] + bmsep[i] for i in range(len(bmsea))] if combined_bmse else None
        lead_time_skill_subplot(
            lead_times=lead_times,
            metric_data=correlations,
            labels=labels,
            metric_type='bcor',
            ax=axes[r, 0],
            title=titles[2 * r],
            significance=bcor_sig,
            **plot_kwargs
        )
        lead_time_skill_subplot(
            lead_times=lead_times,
            metric_data=bmse_combined if combined_bmse else bmsea,
            labels=labels,
            metric_type='bmse',
            ax=axes[r, 1],
            title=titles[2 * r + 1],
            bmsea=bmsea if not combined_bmse else None,
            bmsep=bmsep if not combined_bmse else None,
            combined=combined_bmse,
            significance=bmse_sig,
            **plot_kwargs
        )

    plt.tight_layout()
    _save_and_close(output_filename)
    return fig, axes


# =============================================================================
# Heatmaps and EOFs
# =============================================================================

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
    """Contour plot of BCOR by initialization month (x) and lead time (y), with a threshold contour."""
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
        _set_title(ax, title, fontsize_title, fontweight_title, fontweight_title_label, font_family)

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
        plt.savefig(output_filename, dpi=600, bbox_inches='tight')
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
    # Bootstrap crossing CI parameters
    crossing_ci_lower: Optional[np.ndarray] = None,        # (N, L_i) lower CI for crossing index
    crossing_ci_upper: Optional[np.ndarray] = None,        # (N, L_i) upper CI for crossing index
    fuxi_crossing_ci: Optional[Tuple[float, float]] = None,  # (lower, upper) for FuXi crossing
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
    """Plot one lookback (x) by lead time (y) BCOR heatmap per model, with an optional narrow FuXi panel in the middle.

    Each lookback column gets a bar where BCOR first drops below threshold, and optional bootstrap CI tails
    from crossing_ci_lower/upper (per model and lookback) and fuxi_crossing_ci.
    """
    def first_crossing_yindex(vals: np.ndarray, thr: float = 0.5) -> float:
        """Return the interpolated first index where vals drop below thr (nan if never)."""
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

    # Narrow middle panel for FuXi
    if has_fuxi:
        width = 0.15
        widths = [1] * (num_panels // 2) + [width] + [1] * (num_panels - 1 - num_panels // 2)
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

    if has_fuxi:
        middle = num_panels // 2
        model_axes = [i for i in range(num_panels) if i != middle]
        fuxi_axis = middle
    else:
        model_axes = list(range(num_panels))
        fuxi_axis = None

    def _crossing_to_y(val, T):
        """Convert a crossing index to a y-axis position."""
        y = np.floor(val) - 0.5
        y = max(-0.5, min(T - 0.5, y))
        if val > 0:
            y += 1
        return y

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

        _set_title(ax, lab, fontsize_title, fontweight_title, fontweight_title_label, font_family)

        has_crossing_ci = crossing_ci_lower is not None and crossing_ci_upper is not None

        for c in range(L_i):
            ycross = first_crossing_yindex(A[:, c], thr=threshold)

            # Draw threshold bar only if the point estimate actually crosses
            if not np.isnan(ycross):
                y_bar = _crossing_to_y(ycross, T_i)
                ax.hlines(y_bar, c - bar_half_width, c + bar_half_width, colors="k", linewidth=1.5)

            # Draw CI tails even if point estimate doesn't cross
            if has_crossing_ci:
                ci_lo = crossing_ci_lower[current_model - 1, c]
                ci_hi = crossing_ci_upper[current_model - 1, c]
                if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
                    y_ci_lo = _crossing_to_y(ci_lo, T_i)
                    y_ci_hi = _crossing_to_y(ci_hi, T_i)
                    if abs(y_ci_hi - y_ci_lo) > 0.01:
                        ax.vlines(c, y_ci_lo, y_ci_hi, colors="k", linewidth=1.0)
                        cap_width = bar_half_width * 0.4
                        # Skip caps at plot edges so they don't look clipped
                        if y_ci_lo > -0.4:
                            ax.hlines(y_ci_lo, c - cap_width, c + cap_width, colors="k", linewidth=1.0)
                        if y_ci_hi < T_i - 0.6:
                            ax.hlines(y_ci_hi, c - cap_width, c + cap_width, colors="k", linewidth=1.0)

        bar_proxy = Line2D([0], [0], color="k", lw=1.3, label=legend_label_template.format(threshold=threshold))
        # Legend only on the rightmost model axis
        show_legend = (i == (model_axes[-1]))
        if show_legend:
            ax.legend(handles=[bar_proxy], loc=legend_loc, frameon=True, fancybox=True, framealpha=0.95, fontsize=fontsize_legend)

        ax.set_xlim(-0.5, L_i - 0.5)
        ax.set_ylim(-0.5, T_i - 0.5)

    # Optional FuXi panel in the middle
    if has_fuxi:
        ax = axes[fuxi_axis]
        T_i = fuxi_correlations.shape[0]
        fuxi_A = fuxi_correlations.reshape(-1, 1)
        im = ax.imshow(
            fuxi_A, origin="lower", aspect="auto", cmap=cmap, norm=norm
        )
        last_im = im

        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")

        _set_title(ax, fuxi_label, fontsize_title, fontweight_title, fontweight_title_label, font_family)
        ax.set_xlabel("")

        ycross = first_crossing_yindex(fuxi_A[:, 0], thr=threshold)
        if not np.isnan(ycross):
            y_bar = _crossing_to_y(ycross, T_i)
            ax.hlines(y_bar, -bar_half_width, bar_half_width, colors="k", linewidth=1.5)

        if fuxi_crossing_ci is not None:
            ci_lo, ci_hi = fuxi_crossing_ci
            if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
                y_ci_lo = _crossing_to_y(ci_lo, T_i)
                y_ci_hi = _crossing_to_y(ci_hi, T_i)
                if abs(y_ci_hi - y_ci_lo) > 0.01:
                    ax.vlines(0, y_ci_lo, y_ci_hi, colors="k", linewidth=1.0)
                    cap_width = bar_half_width * 0.4
                    if y_ci_lo > -0.4:
                        ax.hlines(y_ci_lo, -cap_width, cap_width, colors="k", linewidth=1.0)
                    if y_ci_hi < T_i - 0.6:
                        ax.hlines(y_ci_hi, -cap_width, cap_width, colors="k", linewidth=1.0)

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, T_i - 0.5)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.9, pad=0.02, boundaries=bounds, ticks=bounds, spacing="proportional")
        cbar.set_label(colorbar_label, fontsize=fontsize_colorbar_label)
        cbar.ax.tick_params(labelsize=fontsize_tick_label)
        cbar.ax.invert_yaxis()

    if output_filename:
        fig.savefig(output_filename, dpi=600, bbox_inches="tight")

    return fig, list(axes)


def eof_explained_variance_plot(explained_variance, output_filename=None):
    """Bar plot of explained variance (%) per EOF (shows the plot if no output_filename)."""
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(explained_variance)), explained_variance * 100)
    plt.title("Explained Variance of Principal Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def eof_pattern_plot(eof_vector, longitudes, variable_labels, eof_number, output_filename=None):
    """Line plot of one EOF spatial pattern per variable, eof_vector shaped (n_variables, n_longitudes)."""
    plt.figure(figsize=(10, 4))
    for i in range(len(variable_labels)):
        plt.plot(longitudes, eof_vector[i], label=variable_labels[i])

    plt.title(f"EOF{eof_number}")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Eigenvector")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()