import math
import numpy as np
from matplotlib import pyplot as plt

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
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def bivariate_mse_vs_lead_time_plot(lead_times, bmsea, bmsep, labels, output_filename=None):
    assert len(bmsea) == len(labels), 'Number of labels must match number of bmsea sources'
    assert len(bmsep) == len(labels), 'Number of labels must match number of bmsep sources'

    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(labels):
        plt.plot(lead_times[i], bmsea[i], color=colors[i], linestyle='-', label=f'{label} (Amplitude)')
        plt.plot(lead_times[i], bmsep[i], color=colors[i], linestyle='--', label=f'{label} (Phase)')
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Bivariate Mean Squared Error')
    plt.title('Bivariate Mean Squared Error vs Lead Time')
    plt.legend()
    plt.tight_layout()
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
