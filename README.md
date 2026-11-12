# QTAIM charges and base-pair electron densities of chemically modified antisense oligonucleotides

Data and analysis scripts for

> R. Galindo-Murillo and F. Cortés-Guzmán, *QTAIM charges and base-pair electron densities of chemically
> modified antisense oligonucleotides from MD-sampled quantum chemistry*, Phys. Chem. Chem. Phys., submitted (2026).

## What is here

The 3725 MD snapshots of the central base pair are included, together with every number derived from them.

| directory / file | content |
|---|---|
| `systems.csv` | one row per MD replica (147): context (A, G:C, U), modification label, QM fragment charge, snapshot counts (planned, computed, passing QC, Watson–Crick-paired) |
| `snapshots/<replica>/frame_NNNNN.com` | Gaussian 16 input for each snapshot: the exact fragment geometry (Å), charge, multiplicity and route |
| `snapshots/<replica>/frame_NNNNN.atommap.csv` | maps each Gaussian atom index to the Amber topology index, residue, atom name and role (`core`, `linkO`, H `cap`) |
| `snapshots/frames.csv` | snapshot index: replica, frame, MD time (ps), fragment charge, residues, topology file |
| `topologies/*.topo` | Amber topologies (solute only) of the 49 duplexes, used to assign atoms to residues and groups |
| `qtaim/<replica>_atoms.csv` | per-atom QTAIM results for every snapshot that passes QC: charge *q*, *L*(Ω), *K*(Ω), dipole, group |
| `qtaim/<replica>_bcps.csv` | bond critical points: ρ<sub>b</sub>, ∇²ρ<sub>b</sub>, ellipticity, *V*, *G*, *K* = −*H* (`Kcp`) |
| `qtaim/<replica>_groups.csv` | group charges per snapshot (linkage, sugar, base, cap) per residue |
| `npa/<replica>_npa.csv` | NPA charges per atom (from the same Gaussian runs, `pop=nbo`), indexed by Gaussian atom index |
| `results/` | the tables behind the figures and tables of the article and ESI (pooled means, differences vs control, SEM, *z*) |
| `scripts/` | the scripts that produced `qtaim/` and `results/` (see *Scripts* below) |

Replica directory names (e.g. `set1_sys02_v3_seq2_1`) are internal identifiers; `systems.csv` gives the
context and modification of each. Within a context, every modification is compared with its own control:
A = rA:dT duplex, pair A6:dT17; G:C = pair G6:dC15; U = rU:dA duplex, pair U6:dA17 (terminal-pair restraints).

## Level of theory

* Fragment: modified nucleotide 6 with its 5′-phosphate and the Watson–Crick partner, H-capped (see the article, Methods).
* Gaussian 16 A.03: `#p wb97xd/def2TZVP scf=tight nosymm pop=(nbo) scrf=(smd,solvent=water) output=wfx`
  (route as in each `.com` file; `%Mem`/`%CPU` lines are machine specific).
* AIMAll 19.10.12: `aimqb -nogui -bim=auto -cpconn=basic frame_NNNNN.wfx`.
* QC: a snapshot is used only if |Σq − fragment charge| ≤ 0.01 e and max|*L*(Ω)| < 5×10⁻³ (3691 of 3725),
  and only if the imino H-bond has a bond critical point with N···N ≤ 3.5 Å (3600 snapshots in the analysis).
  Two snapshots stopped with a Gaussian SMD error (`Inv3`) and have no QTAIM or NPA data.

## Files that are not included

The wavefunctions (`.wfx`, ~12 MB each, ~45 GB in total) and the AIMAll outputs (`.sum`, ~1.1 MB each,
~4 GB; `.sumviz`, ~60 GB) are not included. Everything used in the article was parsed from the `.sum` files into
`qtaim/`, and both can be regenerated exactly from the `.com` files:

```bash
g16 frame_00001.com                                     # writes frame_00001.wfx
aimqb.ish -nogui -bim=auto -cpconn=basic frame_00001.wfx   # writes frame_00001.sum
```

They are available from the authors on request.

## Scripts

Python 3 with numpy and pandas (matplotlib is imported by some modules). The scripts are the versions used for the article. They expect
the original project directory layout and are provided to document the analysis; the parsed data in `qtaim/`
can be used directly.

* `collect_qtaim.py`, `collect_qtaim_safe.py`: parse AIMAll `.sum` files into the `qtaim/*.csv` tables (QC, atom → group assignment from the topology).
* `compare_systems.py`, `make_preliminary.py`, `qtaim_local.py`: loading, paired-frame selection, statistics
  (SEM across replica means when ≥ 2 replicas have ≥ 10 frames, otherwise block SEM; SEM of a difference = √(SEM₁² + SEM₂²)).
* `make_atom_dn.py`, `npa_groups.py`, `rho_decomposition.py`, `binding_energy.py`, `make_latex_tables.py`: result
  tables; `text_numbers.py`: numbers quoted in the text. The `results/summary_<context>.csv` tables are written by
  `qtaim_local.compare`. Plotting scripts are not included.
* `run_rev01_analysis.sh`: order in which the steps were run (it also lists the plotting steps, which are not included).
* `make_gaussian_input_from_pdb.py`, `extract_central_bp.py`: fragment extraction and capping.

## Licence

Data: CC BY 4.0. Code: MIT. See `LICENSE`.

## Contact

Fernando Cortés-Guzmán (corresponding author), Universidad Nacional Autónoma de México, fercor@unam.mx.
