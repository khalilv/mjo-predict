import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from typing import Sequence, Tuple, List, Union

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

def phase_space_plot(pred_rmm1s, gt_rmm1, pred_rmm2s, gt_rmm2, labels, gt_label, abm_rmm1=None, abm_rmm2=None, title=None, output_filename=None):

    def _add_region_labels():
        plt.text(0, -3.75, "Indian Ocean", ha='center', va='center', fontsize=9)
        plt.text(3.75, 0, "Maritime Continent", ha='center', va='center', fontsize=9, rotation=-90)
        plt.text(0, 3.75, "Western Pacific", ha='center', va='center', fontsize=9)
        plt.text(-3.75, 0, "Western Hem.& Africa", ha='center', va='center', fontsize=9, rotation=90)


    def _draw_phase_wedges():
        r = math.sqrt(32)
        labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

        for i in range(8):
            angle = (5 + i) * np.pi / 4  
            x = np.cos(angle)
            y = np.sin(angle)
            plt.plot([0, r * x], [0, r * y], color='lightgray', lw=0.8, zorder=0)

        for i, label in enumerate(labels):
            angle = (4.5 + i) * np.pi / 4
            x = 4 * np.cos(angle)
            y = 4 * np.sin(angle)
            plt.text(x, y, label, ha='center', va='center', fontsize=9, color='gray')

    assert len(labels) == len(pred_rmm1s), 'Number of labels must match number of prediction sources'
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, label in enumerate(labels):
        plt.plot(pred_rmm1s[i], pred_rmm2s[i], color=colors[i], alpha=0.8, label=label)
        plt.plot(pred_rmm1s[i][0], pred_rmm2s[i][0], color=colors[i], marker='o')  # start
        for lt in range(5, len(pred_rmm1s[i]), 5):
            plt.plot(pred_rmm1s[i][lt], pred_rmm2s[i][lt], color=colors[i], marker='.')

    plt.plot(gt_rmm1, gt_rmm2, color='black', linestyle='--', alpha=0.75, label=gt_label)
    plt.plot(gt_rmm1[0], gt_rmm2[0], color='black', marker='o')  # start
    for lt in range(5, len(gt_rmm1), 5):
            plt.plot(gt_rmm1[lt], gt_rmm2[lt], color='black', marker='.')
    
    if abm_rmm1 is not None and abm_rmm2 is not None:
        plt.plot(abm_rmm1, abm_rmm2, color='lightgray', linestyle='--', alpha=0.75, label='ABM')
        plt.plot(abm_rmm1[0], abm_rmm2[0], color='lightgray', marker='o')  # start
        for lt in range(5, len(gt_rmm1), 5):
                plt.plot(abm_rmm1[lt], abm_rmm2[lt], color='lightgray', marker='.')

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    _draw_phase_wedges()
    _add_region_labels()

    plt.xlabel("RMM1")
    plt.ylabel("RMM2")
    plt.title(title if title else f"Phase space diagram")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal')

    # prevent duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def phase_space_composite_plot(pred_composites, gt_composites, labels, gt_label, abm_composites=None, title=None, output_filename=None):

    def _add_region_labels():
        plt.text(0, -1.75, "Indian Ocean", ha='center', va='center', fontsize=9)
        plt.text(1.75, 0, "Maritime Continent", ha='center', va='center', fontsize=9, rotation=-90)
        plt.text(0, 1.75, "Western Pacific", ha='center', va='center', fontsize=9)
        plt.text(-1.75, 0, "Western Hem.& Africa", ha='center', va='center', fontsize=9, rotation=90)


    def _draw_phase_wedges():
        r = math.sqrt(8)
        labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

        for i in range(8):
            angle = (5 + i) * np.pi / 4  
            x = np.cos(angle)
            y = np.sin(angle)
            plt.plot([0, r * x], [0, r * y], color='lightgray', lw=0.8, zorder=0)

        for i, label in enumerate(labels):
            angle = (4.5 + i) * np.pi / 4
            x = 2 * np.cos(angle)
            y = 2 * np.sin(angle)
            plt.text(x, y, label, ha='center', va='center', fontsize=9, color='gray')

    assert len(labels) == len(pred_composites), 'Number of labels must match number of prediction sources'
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, label in enumerate(labels):
        for _, (rmm1, rmm2) in pred_composites[i].items():
            plt.plot(rmm1, rmm2, color=colors[i], alpha=0.8, label=label)
            plt.plot(rmm1[0], rmm2[0], color=colors[i], marker='o')  # start
            for lt in range(5, len(rmm1), 5):
                plt.plot(rmm1[lt], rmm2[lt], color=colors[i], marker='.')

    for _, (rmm1, rmm2) in gt_composites.items():
        plt.plot(rmm1, rmm2, color='black', linestyle='--', alpha=0.75, label=gt_label)
        plt.plot(rmm1[0], rmm2[0], color='black', marker='o')  # start
        for lt in range(5, len(rmm1), 5):
                plt.plot(rmm1[lt], rmm2[lt], color='black', marker='.')
    
    if abm_composites is not None:
        for _, (rmm1, rmm2) in abm_composites.items():
            plt.plot(rmm1, rmm2, color='lightgray', linestyle='--', alpha=0.75, label='ABM')
            plt.plot(rmm1[0], rmm2[0], color='lightgray', marker='o')  # start
            for lt in range(5, len(rmm1), 5):
                    plt.plot(rmm1[lt], rmm2[lt], color='lightgray', marker='.')

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    _draw_phase_wedges()
    _add_region_labels()

    plt.xlabel("RMM1")
    plt.ylabel("RMM2")
    plt.title(title if title else f"Phase space composite diagram")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal')

    # prevent duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def bivariate_correlation_vs_lead_time_plot(lead_times, correlations, labels, output_filename=None):
    assert len(correlations) == len(labels), 'Number of labels must match number of correlation sources'
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(labels):
        plt.plot(lead_times[i], correlations[i], color=colors[i], label=label)
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1) 
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Bivariate Correlation')
    plt.title('Bivariate Correlation vs Lead Time')
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))   # grid every day
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x % 10 == 0 else ""))  # label every 10 days
    plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5)
    plt.grid(which='major', axis='y')
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
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Bivariate Mean Squared Error')
    plt.title('Bivariate Mean Squared Error vs Lead Time')
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))   # grid every day
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x % 10 == 0 else ""))  # label every 10 days
    plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5)
    plt.grid(which='major', axis='y')
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

def bivariate_correlation_by_month_plot(bcorr_dict, label, title, output_filename=None, levels=np.linspace(0.1, 1.0, 10), threshold=0.5, cmap="RdBu_r"):
    months = sorted(bcorr_dict.keys())
    lead_times = len(next(iter(bcorr_dict.values())))  # assume all months have same lead time length

    # build metric matrix of shape (lead_time, 12)
    metric_array = np.stack([bcorr_dict[m] for m in months], axis=-1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Contourf plot
    c = ax.contourf(months, np.arange(1, lead_times + 1), metric_array, levels=levels, cmap=cmap)

    # Contour line at threshold
    cs = ax.contour(months, np.arange(1, lead_times + 1), metric_array, levels=[threshold], colors=['blue', 'red'])
    ax.clabel(cs, fmt='%.1f')

    # Month labels
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    ax.set_ylabel("Lead time (days)")
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(c, orientation='horizontal', pad=0.1)
    cbar.set_label(label)

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def bivariate_mse_vs_phase_plot(bmse_dict, labels, output_filename, show_error = False, stacked = False, figsize = (24, 6), sharey = True):
    """
    Plot 3 subplots (Weeks 1-2, 3-4, 5-6) showing BMSE vs phase for multiple data sources.

    Parameters
    ----------
    bmse_dict : list of dict
        Each element corresponds to one data source (match to `labels`).
        Each dict maps `phase` -> array of shape (42, 2), where the last dim is
        [BMSEA, BMSEP]. We compute BMSE = BMSEA + BMSEP.
        Phase keys can be int or str (e.g., 1..8 or "1".."8").
    labels : list of str
        Names of the data sources. Must match len(bmse_dict).
    output_filename : str
        Path to save the resulting figure (e.g., "bmse_vs_phase.png").
    show_error : bool, default True
        Show y-error bars (std across leads within the week bin) for BMSE totals.
        (Ignored for stacked components—error bars represent totals only.)
    stacked : bool, default False
        If True, bars are stacked to show BMSEA and BMSEP composition.
        If False, bars show only the total BMSE.
    figsize : (int, int), default (12, 4)
        Figure size.
    dpi : int, default 150
        Dots per inch for saved figure.
    sharey : bool, default True
        Share y-axis across subplots for easier comparison.

    Returns
    -------
    df_agg : pd.DataFrame
        Tidy aggregated data with columns:
        ['label', 'phase', 'bin_index', 'bin_name',
         'bmse_mean', 'bmse_std', 'bmsea_mean', 'bmsep_mean'].
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.

    Notes
    -----
    - Lead indexing (42 leads) is split into 3 bins:
        Weeks 1-2: leads 1-14  -> indices [0:14]
        Weeks 3-4: leads 15-28 -> indices [14:28]
        Weeks 5-6: leads 29-42 -> indices [28:42]
    - Assumes the trailing axis order is [BMSEA, BMSEP].
      BMSE is computed as BMSEA + BMSEP, so order does not affect the total.
    """
    if len(bmse_dict) != len(labels):
        raise ValueError("len(bmse_dict) must match len(labels).")

    # Define week bins as slices over the 42-lead axis
    bins = [
        (slice(0, 14), "Weeks 1-2 (Leads 1-14)"),
        (slice(14, 28), "Weeks 3-4 (Leads 15-28)"),
        (slice(28, 42), "Weeks 5-6 (Leads 29-42)"),
    ]

    # Determine phases across all sources (union), coerce to ints, then sort 1..8
    phases_all = set()
    for src in bmse_dict:
        for ph in src.keys():
            try:
                phases_all.add(int(ph))
            except Exception:
                # If not coercible, keep as-is; but plotting expects ints 1..8
                raise ValueError(f"Phase key '{ph}' is not an int or int-like string.")
    phases = sorted(phases_all)  # expected [1..8], but will honor whatever is present
    n_phases = len(phases)
    n_sources = len(labels)

    # Aggregate into a tidy DataFrame
    rows = []
    for label, src in zip(labels, bmse_dict):
        for ph in phases:
            if ph not in src and str(ph) not in src:
                # Missing phase for this source; fill NaNs
                arr = np.full((42, 2), np.nan, dtype=float)
            else:
                arr = src.get(ph, src.get(str(ph)))
                if not isinstance(arr, np.ndarray) or arr.shape != (42, 2):
                    raise ValueError(
                        f"For label '{label}', phase '{ph}', expected array shape (42,2); got {getattr(arr, 'shape', None)}"
                    )

            bmsea = arr[:, 0]
            bmsep = arr[:, 1]
            bmse = bmsea + bmsep

            for b_idx, (slc, b_name) in enumerate(bins):
                chunk = bmse[slc]
                chunk_a = bmsea[slc]
                chunk_p = bmsep[slc]

                rows.append({
                    "label": label,
                    "phase": ph,
                    "bin_index": b_idx,
                    "bin_name": b_name,
                    "bmse_mean": np.nanmean(chunk),
                    "bmse_std": np.nanstd(chunk, ddof=1) if np.sum(~np.isnan(chunk)) > 1 else np.nan,
                    "bmsea_mean": np.nanmean(chunk_a),
                    "bmsep_mean": np.nanmean(chunk_p),
                })

    df_agg = pd.DataFrame(rows)

    # Start plotting
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=300, sharey=sharey)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])  # safety for edge cases

    # Bar layout
    x = np.arange(n_phases)  # positions for phases
    total_group_width = 0.8
    bar_width = total_group_width / max(1, n_sources)
    # Leftmost offset so groups are centered on each phase tick
    offsets = (np.arange(n_sources) - (n_sources - 1) / 2.0) * bar_width

    # Shared y-limits across subplots (based on totals)
    if sharey:
        ymax = 0.0
        for b_idx in range(3):
            df_bin = df_agg[df_agg["bin_index"] == b_idx]
            ymax = max(ymax, np.nanmax(df_bin["bmse_mean"].values))
        if np.isfinite(ymax):
            y_max_lim = (np.ceil(ymax * 20) / 20) if ymax > 0 else 1.0  # round up a bit
        else:
            y_max_lim = None
    else:
        y_max_lim = None

    # Draw subplots
    for b_idx, ax in enumerate(axes):
        df_bin = df_agg[df_agg["bin_index"] == b_idx]

        for s_idx, label in enumerate(labels):
            df_ls = df_bin[df_bin["label"] == label].set_index("phase").reindex(phases)

            heights_total = df_ls["bmse_mean"].values
            yerr = df_ls["bmse_std"].values if show_error and not stacked else None

            xpos = x + offsets[s_idx]

            if stacked:
                # Draw BMSEA and stack BMSEP on top
                a_part = df_ls["bmsea_mean"].values
                p_part = df_ls["bmsep_mean"].values

                ax.bar(xpos, a_part, width=bar_width, label=(label if b_idx == 0 else None))
                ax.bar(xpos, p_part, width=bar_width, bottom=a_part)
                # Error bars on totals if requested (visually okay on stacks)
                if show_error and yerr is not None:
                    ax.errorbar(xpos, a_part + p_part, yerr=yerr, fmt='none', capsize=3, linewidth=1)
            else:
                ax.bar(xpos, heights_total, width=bar_width, label=(label if b_idx == 0 else None), yerr=yerr, capsize=3)

        # Axis cosmetics
        ax.set_title(df_ls["bin_name"].iloc[0] if not df_ls.empty else ["Weeks 1-2","Weeks 3-4","Weeks 5-6"][b_idx])
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in phases])
        ax.set_xlabel("Phase")
        ax.grid(axis="y", alpha=0.3)
        if y_max_lim is not None:
            ax.set_ylim(0, y_max_lim)

        if b_idx == 0:
            ax.set_ylabel("BMSE")

    # Single legend (labels only need to appear once; we added them on b_idx==0)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    fig.tight_layout(rect=(0, 0, 0.92, 1))  # leave space on right for legend
    if output_filename:
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def bivariate_correlation_vs_phase_plot(bcorr_dict, labels, output_filename, show_error = False, figsize = (24, 6), sharey =True):
    """
    Plot 3 subplots (Weeks 1-2, 3-4, 5-6) showing BCorr vs phase for multiple data sources.

    Parameters
    ----------
    bcorr_dict : list of dict
        Each element corresponds to one data source (match to `labels`).
        Each dict maps `phase` -> array of shape (42,), correlation per lead.
    labels : list of str
        Names of the data sources. Must match len(bcorr_dict).
    output_filename : str
        Path to save the resulting figure (e.g., "bcorr_vs_phase.png").
    show_error : bool, default True
        Show y-error bars (std across leads within the week bin).
    figsize : (int, int), default (12, 4)
        Figure size.
    dpi : int, default 150
        Dots per inch for saved figure.
    sharey : bool, default True
        Share y-axis across subplots for easier comparison.

    Returns
    -------
    df_agg : pd.DataFrame
        Tidy aggregated data with columns:
        ['label', 'phase', 'bin_index', 'bin_name', 'bcorr_mean', 'bcorr_std'].
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.
    """
    if len(bcorr_dict) != len(labels):
        raise ValueError("len(bcorr_dict) must match len(labels).")

    # Week bins over the 42 leads (1-indexed leads → these 0-indexed slices)
    bins = [
        (slice(0, 14), "Weeks 1-2 (Leads 1-14)"),
        (slice(14, 28), "Weeks 3-4 (Leads 15-28)"),
        (slice(28, 42), "Weeks 5-6 (Leads 29-42)"),
    ]

    # Union of phases across all sources; force int and sort
    phases_all = set()
    for src in bcorr_dict:
        for ph in src.keys():
            try:
                phases_all.add(int(ph))
            except Exception:
                raise ValueError(f"Phase key '{ph}' is not an int or int-like string.")
    phases = sorted(phases_all)
    n_phases = len(phases)
    n_sources = len(labels)

    # Aggregate into tidy frame
    rows = []
    for label, src in zip(labels, bcorr_dict):
        for ph in phases:
            arr = src.get(ph, src.get(str(ph)))
            if not isinstance(arr, np.ndarray) or arr.shape != (42,):
                # Missing phase or wrong shape → fill NaNs so it plots as empty
                arr = np.full((42,), np.nan, dtype=float)

            for b_idx, (slc, b_name) in enumerate(bins):
                chunk = arr[slc]
                rows.append({
                    "label": label,
                    "phase": ph,
                    "bin_index": b_idx,
                    "bin_name": b_name,
                    "bcorr_mean": np.nanmean(chunk),
                    "bcorr_std": np.nanstd(chunk, ddof=1) if np.sum(~np.isnan(chunk)) > 1 else np.nan,
                })

    df_agg = pd.DataFrame(rows)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=300, sharey=sharey)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Bar layout
    x = np.arange(n_phases)
    total_group_width = 0.8
    bar_width = total_group_width / max(1, n_sources)
    offsets = (np.arange(n_sources) - (n_sources - 1) / 2.0) * bar_width

    # Shared y-limits across subplots
    if sharey and not df_agg.empty:
        ymin = np.nanmin(df_agg["bcorr_mean"].values)
        ymax = np.nanmax(df_agg["bcorr_mean"].values)
        if np.isfinite(ymin) and np.isfinite(ymax):
            pad = 0.04 * max(1e-6, (ymax - ymin))
            y_limits = (ymin - pad, ymax + pad)
        else:
            y_limits = None
    else:
        y_limits = None

    # Draw
    for b_idx, ax in enumerate(axes):
        df_bin = df_agg[df_agg["bin_index"] == b_idx]

        for s_idx, label in enumerate(labels):
            df_ls = df_bin[df_bin["label"] == label].set_index("phase").reindex(phases)

            heights = df_ls["bcorr_mean"].values
            yerr = df_ls["bcorr_std"].values if show_error else None
            xpos = x + offsets[s_idx]

            ax.bar(xpos, heights, width=bar_width, label=(label if b_idx == 0 else None),
                   yerr=yerr, capsize=3)

        title = df_bin["bin_name"].iloc[0] if not df_bin.empty else ["Weeks 1-2","Weeks 3-4","Weeks 5-6"][b_idx]
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in phases])
        ax.set_xlabel("Phase")
        ax.grid(axis="y", alpha=0.3)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        if b_idx == 0:
            ax.set_ylabel("BCorr")

    # Single legend on the right
    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    fig.tight_layout(rect=(0, 0, 0.92, 1))
    if output_filename:
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def bivariate_correlation_vs_lead_time_heatmap(
    lead_times: Sequence[Sequence[Union[int, float]]],     # (N, T_i)
    lookbacks: Sequence[Sequence[Union[int, float]]],      # (N, L_i)
    correlations: Sequence[np.ndarray],                    # (N, L_i, T_i)
    labels: Sequence[str],                                 # (N,)
    output_filename: str = None,
    *,
    threshold: float = 0.5,
    bar_half_width: float = 0.35,  # half the horizontal length of each bar (x-index units)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    N heatmaps (one per model) with x=lookback (categories), y=lead time (categories).
    Shared Blues colorbar. For each lookback column, draw a short horizontal black bar
    at the first lead-time *index* where correlation falls below `threshold`
    (bar placed at the *lower edge* of that day cell). Y ticks are placed on the
    *upper edge* of cells so the max day label sits exactly at the top border.
    """
    N = len(labels)
    if not (len(lead_times) == len(lookbacks) == len(correlations) == N):
        raise ValueError("lead_times, lookbacks, correlations, and labels must all have length N.")

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

    fig, axes = plt.subplots(1, N, figsize=(5.6 * N, 4.3), constrained_layout=True)
    if N == 1:
        axes = [axes]

    vmin, vmax = 0.0, 1.0
    cmap = "Greens"
    last_im = None

    for i, (ax, lab) in enumerate(zip(axes, labels)):
        arr = np.asarray(correlations[i], dtype=float)   # (L, T)
        leads = np.asarray(lead_times[i])
        looks = np.asarray(lookbacks[i])

        if arr.ndim != 2:
            raise ValueError(f"correlations[{i}] must be 2D, got {arr.shape}.")
        L_i, T_i = arr.shape
        if len(leads) != T_i:
            raise ValueError(f"lead_times[{i}] length {len(leads)} != correlations[{i}].shape[1] {T_i}.")
        if len(looks) != L_i:
            raise ValueError(f"lookbacks[{i}] length {len(looks)} != correlations[{i}].shape[0] {L_i}.")

        # sort lookbacks ascending (x-axis)
        try:
            order_looks = np.argsort(looks.astype(float))
        except Exception:
            order_looks = np.argsort(looks.astype(str))
        looks_sorted = looks[order_looks]
        arr = arr[order_looks, :]

        # ensure lead_times ascending (y-axis)
        try:
            order_leads = np.argsort(leads.astype(float))
        except Exception:
            order_leads = np.argsort(leads.astype(str))
        leads_sorted = leads[order_leads]
        arr = arr[:, order_leads]

        # transpose so x = lookback (columns), y = lead time (rows)
        A = arr.T  # shape (T, L)

        # heatmap (imshow uses cell centers at integer indices)
        im = ax.imshow(A, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        last_im = im

        # x ticks centered on lookbacks
        ax.set_xticks(np.arange(L_i))
        ax.set_xticklabels([str(x) for x in looks_sorted])

        # y ticks on the UPPER edge of selected rows (so top label sits on the top border)
        step = 3
        yt_idx = np.arange(0, T_i, step)          # 0,3,6,... in index space
        ax.set_yticks(yt_idx + 0.5)               # put tick on upper edge
        ax.set_yticklabels([str(leads_sorted[j]) for j in yt_idx])

        ax.set_xlabel("Lookback Window (days)")
        if i == 0:
            ax.set_ylabel("Lead Time (days)")
        ax.set_title(lab)

        # place bars on the LOWER edge of the day cell (index - 0.5)
        for c in range(L_i):
            ycross = first_crossing_yindex(A[:, c], thr=threshold)
            if np.isnan(ycross):
                continue
            y_low_edge = np.floor(ycross) - 0.5
            # clip to image bounds [-0.5, T_i - 0.5]
            y_low_edge = max(-0.5, min(T_i - 0.5, y_low_edge))
            ax.hlines(y_low_edge, c - bar_half_width, c + bar_half_width, colors="k", linewidth=1.3)

        # legend (top-right, boxed) on every subplot
        bar_proxy = Line2D([0], [0], color="k", lw=1.3, label=f"{threshold} skill crossing")
        ax.legend(handles=[bar_proxy], loc="upper right", frameon=True, fancybox=True, framealpha=0.95)

        # tighten to exact image edges so ticks at -0.5 and T_i-0.5 align with borders
        ax.set_xlim(-0.5, L_i - 0.5)
        ax.set_ylim(-0.5, T_i - 0.5)

    # shared colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.9, pad=0.02)
        cbar.set_label("Bivariate Correlation")

    if output_filename:
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")

    return fig, axes