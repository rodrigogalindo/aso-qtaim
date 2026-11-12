#!/usr/bin/env python3
"""
Per-atom delta-q maps (modified - control) as PDBs for ChimeraX/VMD.

  ~/software/miniforge3/envs/qtaim/bin/python local_analysis/make_dq_maps.py

For every modified system of a context with data, and the control itself:
  structures/<ctx>_<system>_dq.pdb   B-factor = mean q(mod) - mean q(V3 control of the context) / e
                                     occupancy = 1 matched atom, 0 unmatched (linkage, caps)
  structures/<ctx>_control_q.pdb     B-factor = mean q of the control
  tables/atom_dq_<ctx>.csv           per matched atom: q means, SEMs, delta, z
Matching (rules 2-4 of ONBOARDING_LOCAL_ANALYSIS.md):
  residue 6: sugar + base atoms by atom name, only names unique within those groups in both
             residues (same base type within a context); linkage atoms differ by chemistry -> unmatched
  partner:   every atom of the (unchanged) partner nucleotide, by atom name
Frames with a non-nuclear attractor (n_nna > 0) and frames whose central pair is open (rule 9,
compare_systems.load) are left out of the per-atom means.
Geometry = one representative QC-passing, NNA-free frame of the modified system (replica 1,
closest to 250 ns), read from its atommap.csv; the frame directories are only read.
Note: the two amino H of the residue-6 base (H61/H62 or H21/H22) swap during MD, so their
per-name means mix both positions.
"""
import json

import numpy as np
import pandas as pd

import qtaim_local as ql
from qtaim_local import cs, mp

TMID = 250.0


def atoms_of(tag):
    sysdir = cs.SYSTEMS[tag][0]
    out = []
    for r in (1, 2, 3):
        f = mp.DATA / f"{sysdir.replace('/', '_')}_{r}_atoms.csv"
        if f.exists():
            a = pd.read_csv(f, dtype={"resid": str})
            a["replica"] = r
            out.append(a)
    a = pd.concat(out, ignore_index=True)
    # rule 9: Watson-Crick paired frames only (same selection as compare_systems.load)
    keep = set(map(tuple, cs.load(tag, sysdir)["groups"][["replica", "frame"]].drop_duplicates().values))
    a = a[[(r, f) in keep for r, f in zip(a.replica, a.frame)]]
    return a[a.n_nna == 0]


def per_atom(a, partner):
    """mean/SEM per (role, atom_name) for res-6 sugar/base (unique names) and every partner atom."""
    res6 = a[(a.resid == "6") & a.group.isin(["sugar", "base"])]
    # unique names only (rule 4): a name that occurs twice in one frame is dropped
    dup = res6.groupby(["replica", "frame", "atom_name"]).size()
    bad = set(dup[dup > 1].index.get_level_values("atom_name"))
    res6 = res6[~res6.atom_name.isin(bad)].assign(site="6")
    par = a[a.resid == str(partner)].assign(site="partner")
    rows = []
    for (site, name), g in pd.concat([res6, par]).groupby(["site", "atom_name"]):
        p = cs.pooled_stats(g[["replica", "q"]].rename(columns={"q": "v"}))
        rows.append(dict(site=site, atom_name=name, group=g.group.iloc[0], n=p["n"], mean=p["mean"], sem=p["sem"],
                         sem_kind=p["sem_kind"]))
    return pd.DataFrame(rows).set_index(["site", "atom_name"]), bad


def rep_frame(tag):
    qm = ql.ROOT / cs.SYSTEMS[tag][0] / "1" / ql.QMDIR
    g = pd.read_csv(mp.DATA / f"{cs.SYSTEMS[tag][0].replace('/', '_')}_1_groups.csv").drop_duplicates("frame")
    g = g[g.n_nna == 0]
    keep = set(cs.load(tag, cs.SYSTEMS[tag][0])["groups"].query("replica == 1").frame)
    g = g[g.frame.isin(keep)]
    fr = int(g.iloc[(g.time_ns - TMID).abs().argsort().iloc[0]].frame)
    d = qm / f"frame_{fr:05d}"
    return d, pd.read_csv(d / f"frame_{fr:05d}.atommap.csv"), json.loads((d / "meta.json").read_text())


def write_pdb(path, amap, meta, partner, values, remark):
    """values: {(site, atom_name): number}; unmatched atoms get B = 0, occupancy 0."""
    with open(path, "w") as fh:
        fh.write(f"REMARK  {remark}\n")
        fh.write(f"REMARK  geometry: {meta['traj']} frame {meta['frame']} t={meta['time_ps'] / 1000:.1f} ns\n")
        for k, r in amap.iterrows():
            cap = r.role.startswith("cap")
            site = "6" if r.resid == 6 else "partner" if r.resid == partner else None
            v = None if cap or site is None else values.get((site, r.atom_name))
            rec = "HETATM" if cap else "ATOM  "
            resn = "CAP" if cap else r.resname[:3]
            chain = "A" if r.resid <= 11 else "B"
            fh.write(f"{rec}{k + 1:5d} {r.atom_name[:4]:<4s} {resn:>3s} {chain}{r.resid:4d}    "
                     f"{r.x:8.3f}{r.y:8.3f}{r.z:8.3f}{1.0 if v is not None else 0.0:6.2f}"
                     f"{(v if v is not None else 0.0):6.3f}          {r.element[:2]:>2s}\n")
        fh.write("END\n")


def main():
    for ctx in ql.CONTEXTS:
        tags = [t for t, v in cs.SYSTEMS.items() if v[4] == ctx
                and (ql.ROOT / v[0] / "1" / ql.QMDIR).is_dir()
                and (mp.DATA / f"{v[0].replace('/', '_')}_1_atoms.csv").exists()]
        ctrl = ql.CTRL[ctx]
        if ctrl not in tags:
            continue
        partner = ql.PARTNER[ctx]
        means = {}

        def means_of(tag):
            if tag not in means:
                means[tag] = per_atom(atoms_of(tag), partner)
            return means[tag]

        c, _ = means_of(ctrl)
        d, amap, meta = rep_frame(ctrl)
        write_pdb(ql.STRUCT / f"{ctx}_control_q.pdb", amap, meta, partner, c["mean"].to_dict(),
                  f"QTAIM mean q (e) of {ctx} control, all replicas, NNA-free frames, in B-factor")
        # same control choice as the group tables: *_std only with >= MIN_CTRL_FRAMES paired frames
        rows = []
        for t in tags:
            if t == ctrl:
                continue
            lab = cs.SYSTEMS[t][1]
            ct = ctrl                        # the V3 control of the context
            m, dropped = means_of(t)
            j = m.join(means_of(ct)[0], lsuffix="_mod", rsuffix="_ctrl", how="inner")
            j["dq"] = j.mean_mod - j.mean_ctrl
            j["sem_diff"] = np.hypot(j.sem_mod, j.sem_ctrl)
            j["z"] = j.dq / j.sem_diff
            j["system"], j["control"] = lab, cs.SYSTEMS[ct][1]
            j["ff_identical_to_control"] = ql.ff_identical(t)
            rows.append(j.reset_index())
            d, amap, meta = rep_frame(t)
            fn = lab.replace("'", "").replace(" ", "").replace("(", "_").replace(")", "")
            write_pdb(ql.STRUCT / f"{ctx}_{fn}_dq.pdb", amap, meta, partner, j["dq"].to_dict(),
                      f"QTAIM delta q = {lab} - {cs.SYSTEMS[ct][1]} (e) in B-factor; occupancy 0 = not matched (linkage, caps)")
            top = j.reindex(j.dq.abs().sort_values(ascending=False).index).head(4)
            print(f"[{ctx}] {lab} vs {cs.SYSTEMS[ct][1]}: {len(j)} atoms matched, dropped dup names {sorted(dropped) or '-'}; "
                  f"frame {meta['frame']}; largest: " + ", ".join(f"{s}:{n} {v:+.4f} (z {z:+.1f})"
                                                                  for (s, n), v, z in zip(top.index, top.dq, top.z)))
        if rows:
            pd.concat(rows).round(6).to_csv(ql.TAB / f"atom_dq_{ctx}.csv", index=False)


if __name__ == "__main__":
    main()
