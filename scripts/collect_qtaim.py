#!/usr/bin/env python3
"""
Collect AIMAll results of every frame under one or more qtaim_bp6 dirs into CSVs.

  collect_qtaim.py control01/1/qtaim_bp6 [more dirs ...] --out results/control01_1

Writes <out>_atoms.csv, <out>_groups.csv and <out>_bcps.csv (long format, one
row per frame x atom / group / bond critical point).

Groups (per nucleotide, from the topology graph, so they work for any
backbone modification):
  linkage : P and everything bonded to it except O5'/O3' (i.e. OP1/OP2, S, CH3,
            N-SO2CH3, guanidine...)  <- where set1 modifications live
  sugar   : ribose/deoxyribose incl. O5', O3', 2'-substituent and their H
  base    : nucleobase (everything beyond the glycosidic N)
  cap     : link O3' of the neighbour + H link atoms (not part of the nucleotide)
  nucleotide = linkage + sugar + base
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_central_bp import Topology  # noqa: E402

FLOAT = r"[-+]?\d+\.\d+E[-+]\d+"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EFS_ROOT = "/mnt/efs/fs1/work/20251224-nbo-qtaim-studies-of-ASOs"


def local_path(p):
    """meta.json stores absolute EFS paths; on another computer map them onto this project root."""
    p = str(p)
    if not Path(p).exists() and p.startswith(EFS_ROOT):
        p = str(PROJECT_ROOT) + p[len(EFS_ROOT):]
    return p


def parse_sum(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    atoms = {}
    for k, line in enumerate(lines):
        if line.startswith("Atom A          q(A)"):
            for row in lines[k + 2:]:
                if row.startswith("-----"):
                    break
                p = row.split()
                atoms[p[0]] = dict(q=float(p[1]), L=float(p[2]), K=float(p[3]),
                                   mu_intra=float(p[5]))
            break
    bcps = []
    k = 0
    while k < len(lines):
        if lines[k].startswith("CP#"):
            m = re.match(r"\s*Type = \((3,[-+]?\d)\)\s+(\w+)\s*(.*)", lines[k + 1])
            if m and m.group(2) == "BCP":
                a, b = m.group(3).split()[:2]
                cp = dict(A=a, B=b)
                for row in lines[k + 2:k + 40]:
                    if row.startswith("CP#") or not row.strip():
                        break
                    s = row.strip()
                    for key, name in (("Rho =", "rho"), ("DelSqRho =", "lap"),
                                      ("Bond Ellipticity =", "ellip"), ("V =", "V"),
                                      ("G =", "G"), ("K =", "Kcp")):
                        if s.startswith(key):
                            v = s[len(key):].split()[0]
                            cp[name] = float(v) if re.match(FLOAT, v) else float("nan")
                bcps.append(cp)
        k += 1
    return atoms, bcps


def classify(top, idx):
    """Return group name for topology atom idx (0-based) within its residue."""
    cache = top.__dict__.setdefault("_groups", {})
    r = int(top.resid[idx])
    if r not in cache:
        cache[r] = _residue_groups(top, r)
    base, link = cache[r]
    if idx in base:
        return "base"
    if idx in link:
        return "linkage"
    return "sugar"


def _residue_groups(top, r):
    same = lambda j: top.resid[j] == r  # noqa: E731
    name = lambda j: top.names[j]       # noqa: E731
    # glycosidic bond C1'-N: base = component beyond N
    c1 = next(j for j in range(top.natom) if same(j) and name(j) == "C1'")
    nglyc = next(j for j in top.nbr[c1] if same(j) and top.elem(j) == "N")
    base, stack = {nglyc}, [nglyc]
    while stack:
        i = stack.pop()
        for j in top.nbr[i]:
            if same(j) and j not in base and j != c1:
                base.add(j)
                stack.append(j)
    link = set()
    ps = [j for j in range(top.natom) if same(j) and top.elem(j) == "P"]
    for p in ps:
        link.add(p)
        stack = [j for j in top.nbr[p] if same(j) and name(j) != "O5'"]
        while stack:
            i = stack.pop()
            if i in link:
                continue
            link.add(i)
            stack += [j for j in top.nbr[i] if same(j) and j not in link and j != p]
    return base, link


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("qm_dirs", nargs="+")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    atom_rows, group_rows, bcp_rows = [], [], []
    top_cache = {}
    for qm in args.qm_dirs:
        for d in sorted(Path(qm).glob("frame_*")):
            tag = d.name
            s = d / f"{tag}.sum"
            if not s.exists():          # .sum is written only when AIMAll finishes; .sumviz not needed
                continue
            meta = json.loads((d / "meta.json").read_text())
            topo = local_path(meta["topo"])
            top = top_cache.setdefault(topo, Topology(topo))
            amap = {}
            with open(d / f"{tag}.atommap.csv") as fh:
                for r in csv.DictReader(fh):
                    ti = int(r["top_index"])
                    if r["role"] == "core":
                        r["group"] = classify(top, ti - 1)
                    else:
                        r["group"] = "cap"
                    amap[r["label"]] = r
            atoms, bcps = parse_sum(s)
            if not atoms:                # incomplete .sum
                continue
            # Non-nuclear attractors (NNA, seen inside P-S and S-C bonds with def2-SVP) are extra
            # basins. Assign each to the group of the atoms its bond paths connect it to, so group
            # and nucleotide charges stay complete; individual atom charges next to it are not
            # comparable with NNA-free frames.
            nna = [lab for lab in atoms if lab.startswith("NNA")]
            for lab in nna:
                nb = [cp["B"] if cp["A"] == lab else cp["A"] for cp in bcps if lab in (cp["A"], cp["B"])]
                nb = [amap[x] for x in nb if x in amap]
                keys = {(m["resid"], m["resname"], m["group"]) for m in nb}
                rid, rn, grp = keys.pop() if len(keys) == 1 else (nb[0]["resid"] if nb else "0",
                                                                    nb[0]["resname"] if nb else "?", "nna")
                amap[lab] = dict(label=lab, element="NNA", top_index="0", resid=rid, resname=rn,
                                 atom_name="NNA", role="nna", group=grp)
            # QC: total charge must match the fragment charge and every |L(A)| < 1e-3
            sumq = sum(a["q"] for a in atoms.values())
            maxl = max(abs(a["L"]) for a in atoms.values())
            if abs(sumq - meta["charge"]) > 0.01 or maxl > 5e-3:   # same policy as run_frames.sh
                print(f"[QC-SKIP] {d}: sum q = {sumq:.3f} (expected {meta['charge']}), "
                      f"max |L| = {maxl:.1e}")
                continue
            if len(atoms) != len(amap):
                print(f"[WARN] {d}: {len(atoms)} basins in .sum vs {len(amap)} in map")
            t_ns = meta["time_ps"] / 1000.0
            base = dict(qm_dir=str(qm), frame=meta["frame"], time_ns=t_ns, n_nna=len(nna), max_absL=maxl)
            sums = {}
            for lab, a in atoms.items():
                m = amap[lab]
                atom_rows.append({**base, "label": lab, "resid": m["resid"],
                                  "resname": m["resname"], "atom_name": m["atom_name"],
                                  "group": m["group"], "top_index": m["top_index"], **a})
                key = (m["resid"], m["resname"], m["group"])
                sums[key] = sums.get(key, 0.0) + a["q"]
                if m["group"] != "cap":
                    k2 = (m["resid"], m["resname"], "nucleotide")
                    sums[k2] = sums.get(k2, 0.0) + a["q"]
            for (rid, rn, g), q in sorted(sums.items()):
                group_rows.append({**base, "resid": rid, "resname": rn, "group": g, "q": q})
            group_rows.append({**base, "resid": "all", "resname": "all", "group": "total",
                               "q": sum(a["q"] for a in atoms.values())})
            for cp in bcps:
                ma, mb = amap[cp["A"]], amap[cp["B"]]
                # canonical order (AIMAll lists the two atoms in either order)
                if (int(ma["resid"]), ma["atom_name"]) > (int(mb["resid"]), mb["atom_name"]):
                    ma, mb = mb, ma
                    cp = {**cp, "A": cp["B"], "B": cp["A"]}
                inter = ma["resid"] != mb["resid"] and "cap" not in (ma["group"], mb["group"])
                name = (f"{ma['resname']}{ma['resid']}:{ma['atom_name']}-"
                        f"{mb['resname']}{mb['resid']}:{mb['atom_name']}")
                bcp_rows.append({**base, "bcp": name, "A": cp["A"], "B": cp["B"],
                                 "groupA": ma["group"], "groupB": mb["group"],
                                 "inter_residue": inter,
                                 **{k: cp.get(k) for k in ("rho", "lap", "ellip", "V", "G", "Kcp")}})
    if not atom_rows:
        sys.exit("[ERROR] no finished frames found")
    for suffix, rows in (("atoms", atom_rows), ("groups", group_rows), ("bcps", bcp_rows)):
        out = Path(f"{args.out}_{suffix}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
