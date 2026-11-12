#!/usr/bin/env python3
"""
Extract the central base pair (default: residue 6 + its Watson-Crick partner)
from an AMBER trajectory and write one Gaussian single-point input per frame.

Works directly on prmtop + NetCDF by topology atom index, so duplicate atom
names inside modified residues (e.g. two "C2" in MRU) are not a problem.

Fragment construction (link-atom capping):
  * all atoms of the selected residues, INCLUDING their own 5'-phosphate
    (where every set1 backbone modification lives);
  * if a selected P is bonded to an atom outside the fragment (the O3' of the
    5'-neighbour), that O is pulled into the fragment;
  * every remaining bond that crosses the fragment boundary is capped with an
    H placed along the bond vector (O-H 0.96, N-H 1.01, C-H 1.09, S-H 1.34 A).
  Result per nucleotide: HO-P(X)(Y)-O5'-sugar-base ... 3'-OH.
  Net charge = sum of the (integer) AMBER residue charges of the selected
  residues; the caps are chemically neutral.

Usage (from a replica dir, e.g. control01/1):
  extract_central_bp.py --topo ../nowater.topo --traj nowater.nc \
      --residues 6,17 --frames 1:2500:50 --outdir qtaim_bp6
"""
import argparse
import csv
import json
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 15: "P", 16: "S",
        17: "Cl", 19: "K", 35: "Br"}
CAP_LEN = {"O": 0.96, "N": 1.01, "C": 1.09, "S": 1.34}


def read_prmtop(path):
    """Minimal fixed-width prmtop reader -> dict FLAG -> list of values."""
    data, flag, fmt = {}, None, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("%FLAG"):
                flag = line.split()[1]
                data[flag] = []
            elif line.startswith("%FORMAT"):
                m = re.match(r"%FORMAT\((\d+)([aIE])(\d+)", line.strip())
                n, kind, width = int(m.group(1)), m.group(2), int(m.group(3))
                fmt = (kind, width)
            elif line.startswith("%"):
                continue
            elif flag is not None:
                kind, width = fmt
                s = line.rstrip("\n")
                for i in range(0, len(s), width):
                    tok = s[i:i + width]
                    if not tok.strip() and kind != "a":
                        continue
                    if kind == "a":
                        data[flag].append(tok.strip())
                    elif kind == "I":
                        data[flag].append(int(tok))
                    else:
                        data[flag].append(float(tok))
    return data


class Topology:
    def __init__(self, path):
        d = read_prmtop(path)
        self.natom = d["POINTERS"][0]
        self.names = d["ATOM_NAME"][: self.natom]
        self.charges = np.array(d["CHARGE"][: self.natom]) / 18.2223
        self.z = d["ATOMIC_NUMBER"][: self.natom]
        labels = d["RESIDUE_LABEL"]
        ptrs = d["RESIDUE_POINTER"] + [self.natom + 1]
        self.resid = np.zeros(self.natom, int)
        self.resname = [""] * self.natom
        for r in range(len(labels)):
            for a in range(ptrs[r] - 1, ptrs[r + 1] - 1):
                self.resid[a] = r + 1
                self.resname[a] = labels[r]
        self.bonds = []
        for key in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
            b = d[key]
            for i in range(0, len(b), 3):
                self.bonds.append((b[i] // 3, b[i + 1] // 3))
        self.nbr = [[] for _ in range(self.natom)]
        for i, j in self.bonds:
            self.nbr[i].append(j)
            self.nbr[j].append(i)

    def elem(self, i):
        return ELEM.get(self.z[i], "X")


def build_fragment(top, residues, charge_override=None):
    core = [i for i in range(top.natom) if top.resid[i] in residues]
    frag = set(core)
    link_o = []
    # pull in the 5'-neighbour O3' bonded to each selected P
    for i in core:
        if top.elem(i) == "P":
            for j in top.nbr[i]:
                if j not in frag and top.elem(j) == "O":
                    frag.add(j)
                    link_o.append(j)
    caps = []  # (inside atom, outside atom it replaces)
    for i in sorted(frag):
        for j in top.nbr[i]:
            if j not in frag:
                caps.append((i, j))
    for i, _ in caps:
        if top.elem(i) not in CAP_LEN:
            sys.exit(f"[ERROR] no cap length for element {top.elem(i)} (atom {i + 1})")
    res_q = {r: float(top.charges[top.resid == r].sum()) for r in residues}
    for r, q in res_q.items():
        if abs(q - round(q)) > 0.02:
            sys.exit(f"[ERROR] residue {r} AMBER charge {q:.4f} is not an integer")
    charge = int(sum(round(q) for q in res_q.values()))
    if charge_override is not None:
        charge = charge_override
    order = sorted(frag)
    nelec = sum(top.z[i] for i in order) + len(caps) - charge
    if nelec % 2:
        sys.exit(f"[ERROR] odd electron count {nelec} for charge {charge}. The AMBER residue "
                 "charges do not match the chemistry (e.g. neutral methylphosphonate or "
                 "guanidine linkages carry -1 in some libraries); pass --charge explicitly.")
    return order, set(link_o), caps, charge, res_q, nelec


def write_frame(outdir, tag, top, order, link_o, caps, charge, xyz, time_ps,
                frame, args, res_q, nelec):
    work = outdir / tag
    com = work / f"{tag}.com"
    if com.exists() and not args.force:
        return False
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in order:
        role = "linkO" if i in link_o else "core"
        rows.append(dict(elem=top.elem(i), top_index=i + 1, resid=int(top.resid[i]),
                         resname=top.resname[i], atom_name=top.names[i], role=role,
                         xyz=xyz[i]))
    for i, j in caps:
        v = xyz[j] - xyz[i]
        h = xyz[i] + CAP_LEN[top.elem(i)] * v / np.linalg.norm(v)
        rows.append(dict(elem="H", top_index=0, resid=int(top.resid[i]),
                         resname=top.resname[i], atom_name=f"HCAP_{top.names[i]}",
                         role=f"cap(replaces {top.names[j]}:{top.resid[j]})", xyz=h))
    with open(work / f"{tag}.atommap.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gauss_index", "label", "element", "top_index", "resid", "resname",
                    "atom_name", "role", "x", "y", "z"])
        for k, r in enumerate(rows, 1):
            w.writerow([k, f"{r['elem']}{k}", r["elem"], r["top_index"], r["resid"],
                        r["resname"], r["atom_name"], r["role"],
                        *[f"{c:.4f}" for c in r["xyz"]]])
    route = f"#p {args.method}/{args.basis} scf=tight nosymm"
    if args.nbo:
        route += " pop=(nbo)"
    if args.solv == "smd":
        route += f" scrf=(smd,solvent={args.smd_solvent})"
    route += " output=wfx"
    with open(com, "w") as fh:
        fh.write(f"%Mem={args.mem}\n%CPU={args.cpu}\n{route}\n\n")
        fh.write(f"{tag} frame {frame} t={time_ps:.1f}ps residues {args.residues} q={charge}\n\n")
        fh.write(f"{charge} 1\n")
        for r in rows:
            x, y, z = r["xyz"]
            fh.write(f"{r['elem']:2s} {x:14.6f} {y:14.6f} {z:14.6f}\n")
        fh.write(f"\n{tag}.wfx\n\n")
    meta = dict(tag=tag, frame=frame, time_ps=float(time_ps), charge=charge, mult=1,
                n_electrons=int(nelec), residues=args.residues,
                residue_amber_charges={str(k): round(v, 4) for k, v in res_q.items()},
                resnames={str(r): top.resname[int(np.where(top.resid == r)[0][0])]
                          for r in res_q},
                natoms=len(rows), ncaps=len(caps), method=args.method, basis=args.basis,
                solv=args.solv, topo=str(Path(args.topo).resolve()),
                traj=str(Path(args.traj).resolve()),
                created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                host=socket.gethostname())
    (work / "meta.json").write_text(json.dumps(meta, indent=1))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topo", default="../nowater.topo")
    ap.add_argument("--traj", default="nowater.nc")
    ap.add_argument("--residues", default="6,17",
                    help="1-based residue numbers (seq1/seq2: 6,17; seq3/404D: 6,15)")
    ap.add_argument("--frames", default="1:2500:50",
                    help="start:stop:step, 1-based inclusive frame numbers in the NetCDF")
    ap.add_argument("--outdir", default="qtaim_bp6")
    ap.add_argument("--method", default="wb97xd")
    ap.add_argument("--basis", default="def2SVP")
    ap.add_argument("--solv", choices=["gas", "smd"], default="gas")
    ap.add_argument("--smd-solvent", default="water")
    ap.add_argument("--no-nbo", dest="nbo", action="store_false")
    ap.add_argument("--mem", default="64GB")
    ap.add_argument("--cpu", default="0-47", help="Gaussian %%CPU list (physical cores)")
    ap.add_argument("--charge", type=int, default=None,
                    help="net fragment charge; default = sum of AMBER residue charges")
    ap.add_argument("--force", action="store_true", help="overwrite existing .com files")
    args = ap.parse_args()

    residues = [int(x) for x in args.residues.split(",")]
    top = Topology(args.topo)
    order, link_o, caps, charge, res_q, nelec = build_fragment(top, residues, args.charge)

    start, stop, step = (int(x) for x in args.frames.split(":"))
    nc = Dataset(args.traj)
    nframes = nc.dimensions["frame"].size
    if nc.dimensions["atom"].size != top.natom:
        sys.exit(f"[ERROR] traj has {nc.dimensions['atom'].size} atoms, topo {top.natom}")
    stop = min(stop, nframes)

    names = ", ".join(f"{r}:{top.resname[int(np.where(top.resid == r)[0][0])]}"
                      f"(q={res_q[r]:+.2f})" for r in residues)
    print(f"[INFO] residues {names}; fragment {len(order)} atoms + {len(caps)} caps "
          f"= {len(order) + len(caps)}; charge {charge}; electrons {nelec}")
    print(f"[INFO] frames {start}..{stop} step {step} of {nframes}")

    outdir = Path(args.outdir)
    nnew = 0
    for f in range(start, stop + 1, step):
        xyz = np.asarray(nc.variables["coordinates"][f - 1], float)
        t = float(nc.variables["time"][f - 1]) if "time" in nc.variables else float("nan")
        tag = f"frame_{f:05d}"
        nnew += write_frame(outdir, tag, top, order, link_o, caps, charge, xyz, t, f,
                            args, res_q, nelec)
    nc.close()
    print(f"[OK] wrote {nnew} new frame inputs under {outdir}/")


if __name__ == "__main__":
    main()
