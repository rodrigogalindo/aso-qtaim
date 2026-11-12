#!/usr/bin/env bash
# rev01 analysis: every QTAIM-dependent step with QTAIM_REV=rev01 (data from <replica>/qtaim_bp6_rev01), outputs in
# local_analysis/rev01/{qtaim_data,figures,tables,structures} and article/rev01/tables; then the rev01 figure sync
# and the PDF build. The MD analyses (md_structure, md_pucker_strands, md_nastruct, ff_consistency) do not depend on
# the wavefunction and are not rerun; their tables are read from local_analysis/tables.
#
#   local_analysis/rev01/run_rev01_analysis.sh            # analysis + sync + build main/si
#   NOBUILD=1 local_analysis/rev01/run_rev01_analysis.sh  # analysis only
# Test overrides (passed through to qtaim_local.py): QTAIM_QMDIR, QTAIM_OUT, QTAIM_ARTICLE.
set -euo pipefail
LA="$(cd "$(dirname "$0")/.." && pwd)"
PY=~/software/miniforge3/envs/qtaim/bin/python
export QTAIM_REV=rev01
cd "$LA"
step() { echo; echo "=== $(date '+%T') $*"; "$PY" "$@"; }
step make_figures.py          # collect + group charges / rho_b vs control: tables/summary_<ctx>.csv, frame_status.csv
step make_dq_maps.py          # per-atom dq PDBs + tables/atom_dq_<ctx>.csv
step make_si_figures.py       # rho-distance fits, MD/nastruct SI figures
step npa_groups.py            # NPA group charges from the rev01 g16 logs
step rho_decomposition.py
step binding_energy.py        # MM/GBSA (MD) vs rev01 rho_b; writes article/rev01/tables/tabS_binding.tex
step make_atom_dn.py          # per-atom populations (ESI S7); rev01 cache in rev01/qtaim_data
step make_ewg_table.py        # discussion image (not an article figure)
step make_latex_tables.py     # article/rev01/tables/tabS_*.tex
[[ -n ${NOBUILD:-} || -n ${QTAIM_ARTICLE:-} ]] && { echo; echo "analysis done (no sync/build)"; exit 0; }
cd "$LA/../article/rev01"
./sync_figures.sh
./build.sh all < /dev/null
