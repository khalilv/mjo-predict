#!/bin/bash
# Reproduce the paper figures that only need the data shipped in data/ (Figs 2-5, 7, 8 and the
# single-case phase-space plot). Run from the repository root after unzipping data/*.zip.
# Figures are written to plots/. Figs 2-5 run a 1,000-sample bootstrap and take a while.
# Usage: bash scripts/make_figures.sh

# module <- config, run sequentially
JOBS=(
    "mjo.utils.analysis.lead_time_errors|configs/analysis/lead_time_errors.yaml"      # Figs 2, 4
    "mjo.utils.analysis.lead_time_bias|configs/analysis/lead_time_bias.yaml"          # Fig 3
    "mjo.utils.analysis.lead_time_errors|configs/analysis/lead_time_errors_BoM.yaml"  # Fig 5
    "mjo.utils.analysis.phase_space_composite|configs/analysis/phase_space_composite.yaml"  # Fig 7
    "mjo.utils.analysis.monthly_errors|configs/analysis/monthly_errors.yaml"          # Fig 8
    "mjo.utils.analysis.phase_space|configs/analysis/phase_space.yaml"                # single case
)

failures=0
for job in "${JOBS[@]}"; do
    module_name="${job%%|*}"
    config="${job##*|}"
    echo "[$(date '+%F %T')] START  ${module_name} <- ${config}"
    if python -m "${module_name}" --config "${config}"; then
        echo "[$(date '+%F %T')] DONE   ${config}"
    else
        echo "[$(date '+%F %T')] FAILED ${config}"
        failures=$((failures + 1))
    fi
done
exit "$failures"
