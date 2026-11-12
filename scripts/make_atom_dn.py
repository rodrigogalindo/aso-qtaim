#!/usr/bin/env python3
"""
Per-atom QTAIM population changes of the modified nucleotide (residue 6), one bar panel per system,
in the style of Fig. 10.4 (b, d, f, h) of Matta & Boyd (eds.), Quantum Biochemistry (Wiley, 2010).

  ~/software/miniforge3/envs/qtaim/bin/python local_analysis/make_atom_dn.py

Per context:
  figures/figS_atom_dN_<ctx>.{png,pdf}   first panel: N(Omega) of the control; then one panel per
                                         modification: dN = N(mod) - N(control) per atom position
  tables/atom_dN_<ctx>.csv               per position: means, SEMs, dN, z
Atom positions ("slots") are those of the control nucleotide plus the bridging O3' of residue 5.
Atoms of a modified residue are put on a slot by connectivity in one representative frame (by
top_index, so names reused in the linkage and the base cannot be confused, and H5'1/H5'2 are assigned
by chirality, since the naming is not consistent between the residue libraries). Where the element
changes (O -> S, N, C or F) the bar is hatched and shows -dq, i.e. the change of N - Z; atoms with no
counterpart in the control (methyl, Ms, guanidine, MOE, OMe, cEt bridge atoms) are left out.
Frames: QC-passing, NNA-free, Watson-Crick-paired (make_dq_maps.atoms_of); SEM across replica means.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import qtaim_local as ql
from qtaim_local import cs
import make_dq_maps as dm

Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16}
PRO_R_SIGN = None   # set from the control in main()
O3P = "O3'(5)"          # bridging O3' of the 5'-neighbour
SUGAR = ["O5'", "C5'", "H5'", "H5''", "C4'", "H4'", "O4'", "C1'", "H1'", "C2'", "H2'", "O2'", "HO2'",
         "C3'", "H3'", "O3'"]
TEX = {"'": "′"}


def nice(s):
    return s.replace("''", "″").replace("'", "′")


def slots_of(amap, groups):
    """{top_index: (slot, group, element)} for residue 6 (+ the residue-5 O3') of one frame;
    groups: {top_index: linkage|sugar|base} from the collected atom table."""
    r = amap[(amap.resid == 6) & ~amap.role.str.startswith("cap")].reset_index(drop=True)
    r["role"] = r.top_index.map(groups)
    X = r[["x", "y", "z"]].to_numpy()
    name = list(r.atom_name)
    idx = {n: i for i, n in enumerate(name)}

    def nbrs(i, cut=2.1):
        d = np.linalg.norm(X - X[i], axis=1)
        return [j for j in np.argsort(d)[1:6] if d[j] < (cut if r.element[j] != "H" and r.element[i] != "H" else 1.25)]

    out = {}
    o3 = amap[(amap.resid == 5) & (amap.atom_name == "O3'")]
    out[int(o3.top_index.iloc[0])] = (O3P, "linkage", "O")
    sugar_i = {i for i, g in enumerate(r.role) if g == "sugar"}
    # linkage: P and the two non-bridging positions
    p = next(i for i, n in enumerate(name) if n == "P" and r.role[i] == "linkage")
    out[int(r.top_index[p])] = ("P", "linkage", "P")
    nb = [j for j in nbrs(p) if r.role[j] == "linkage"]
    if len(nb) != 2:
        raise RuntimeError(f"expected 2 non-bridging P substituents: {[name[j] for j in nb]}")
    # pro-R / pro-S position by chirality at P (library names are not consistent); the sign that
    # OP1 has in the standard residue is PRO_R_SIGN
    xo3 = o3[["x", "y", "z"]].to_numpy()[0]
    chi = lambda j: np.sign(np.dot(np.cross(xo3 - X[p], X[idx["O5'"]] - X[p]), X[j] - X[p]))
    global PRO_R_SIGN
    if PRO_R_SIGN is None:                       # first call = the control of the first context
        PRO_R_SIGN = chi(idx["OP1"])
    got = {j: ("OP1" if chi(j) == PRO_R_SIGN else "OP2") for j in nb}
    if len(set(got.values())) != 2:
        raise RuntimeError("both non-bridging substituents on the same side")
    for j, s in got.items():
        out[int(r.top_index[j])] = (s, "linkage", r.element[j])
    print("   linkage:", ", ".join(f"{name[j]}->{s}" for j, s in sorted(got.items(), key=lambda t: t[1])))
    # sugar: canonical names, plus the 2'-substituent and the H5' pair by geometry
    for n in SUGAR:
        if n in idx and idx[n] in sugar_i and n not in ("O2'", "H5'", "H5''"):
            out[int(r.top_index[idx[n]])] = (n, "sugar", r.element[idx[n]])
    c2 = idx["C2'"]
    x2 = [j for j in nbrs(c2) if r.element[j] in ("O", "F") and j in sugar_i]
    if len(x2) == 1:
        out[int(r.top_index[x2[0]])] = ("O2'", "sugar", r.element[x2[0]])
    c5 = idx["C5'"]
    v1, v2 = X[idx["O5'"]] - X[c5], X[idx["C4'"]] - X[c5]
    for j in nbrs(c5):
        if r.element[j] == "H":
            s = "H5'" if np.dot(np.cross(v1, v2), X[j] - X[c5]) > 0 else "H5''"
            out[int(r.top_index[j])] = (s, "sugar", "H")
    # base: names are the same in every residue of a context
    for i, g in enumerate(r.role):
        if g == "base":
            out[int(r.top_index[i])] = (name[i], "base", r.element[i])
    return out


# residue-6 amino H and the partner acceptor of its Watson-Crick H-bond; which amino H is H-bonded
# differs between residue libraries, so both are re-assigned per frame by distance
AMINO = {"A_v3": (("H61", "H62"), "O4"), "GC_v3": (("H21", "H22"), "O2")}
HB, FREE = "hb", "free"   # amino H in the Watson-Crick H-bond / the other one


def amino_hb(tag, a):
    ctx = cs.SYSTEMS[tag][4]
    if ctx not in AMINO:
        return
    (h1, h2), acc = AMINO[ctx]
    partner = ql.PARTNER[ctx]
    hmask = (a.resid == "6") & a.atom_name.isin([h1, h2]) & (a.group == "base")
    for (qd, fr), g in a[hmask].groupby(["qm_dir", "frame"]):
        m = pd.read_csv(Path(qd) / f"frame_{fr:05d}" / f"frame_{fr:05d}.atommap.csv")
        xa = m[(m.resid == partner) & (m.atom_name == acc)][["x", "y", "z"]].to_numpy()[0]
        hh = m[m.top_index.isin(g.top_index)]
        d = np.linalg.norm(hh[["x", "y", "z"]].to_numpy() - xa, axis=1)
        near = int(hh.top_index.iloc[int(np.argmin(d))])
        for i, t in zip(g.index, g.top_index):
            a.loc[i, "slot"] = f"{h1[:-1]} {HB if t == near else FREE}"


def per_slot(tag):
    print(" ", cs.SYSTEMS[tag][1])
    a = dm.atoms_of(tag)
    _, amap, _ = dm.rep_frame(tag)
    g6 = a[a.resid == "6"].drop_duplicates("top_index")
    sl = slots_of(amap, dict(zip(g6.top_index, g6.group)))
    a = a[a.top_index.isin(sl)].copy()
    a["slot"] = a.top_index.map(lambda t: sl[t][0])
    amino_hb(tag, a)
    a["grp"] = a.top_index.map(lambda t: sl[t][1])
    a["el"] = a.top_index.map(lambda t: sl[t][2])
    rows = []
    for (s, grp, el), g in a.groupby(["slot", "grp", "el"]):
        p = cs.pooled_stats(g[["replica", "q"]].rename(columns={"q": "v"}))
        rows.append(dict(slot=s, group=grp, element=el, name=g.atom_name.iloc[0], n=p["n"], q=p["mean"],
                         sem=p["sem"], sem_kind=p["sem_kind"]))
    df = pd.DataFrame(rows).set_index("slot")
    if df.index.duplicated().any():
        raise RuntimeError(f"{tag}: duplicated slots {df.index[df.index.duplicated()].tolist()}")
    return df


def order_of(ctrl):
    base = [s for s in ctrl.index if ctrl.loc[s, "group"] == "base"]
    base = [s for s in ctrl.sort_values("name").index if s in base]
    _, amap, _ = dm.rep_frame(CT)
    seq = list(amap[amap.resid == 6].atom_name)   # base atoms in topology order
    base.sort(key=lambda s: seq.index(ctrl.loc[s, "name"]))
    return [O3P, "P", "OP1", "OP2"] + [s for s in SUGAR if s in ctrl.index] + base


def panel(ax, xs, vals, errs, cols, hatch, labels, title, ylabel, spans, zoom=None, fs=1.0):
    ax.axhline(0, color="k", lw=0.8)
    for g, (a, b), c in spans:
        ax.axvspan(a - 0.5, b + 0.5, color=c, alpha=0.35, lw=0, zorder=0)
    ax.bar(xs, vals, 0.8, color=cols, edgecolor="k", lw=0.5, hatch=None, zorder=2)
    for x, v, h in zip(xs, vals, hatch):
        if h and np.isfinite(v):
            ax.bar(x, v, 0.8, color="none", edgecolor="k", lw=0.5, hatch="////", zorder=3)
    ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor="k", elinewidth=0.6 * fs, capsize=1.5 * fs, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5 * fs)
    ax.set_xlim(-0.7, len(xs) - 0.3)
    ax.set_ylabel(ylabel, fontsize=8 * fs, labelpad=1)
    ax.tick_params(axis="y", labelsize=7 * fs)
    ax.tick_params(axis="x", pad=1, length=2)
    ax.grid(axis="y", ls=":", lw=0.5, zorder=0)
    ax.set_title(title, loc="left", fontsize=9 * fs, fontweight="bold", pad=2)
    lo = np.nanmin(np.minimum(vals - np.nan_to_num(errs), 0))
    hi = np.nanmax(np.maximum(vals + np.nan_to_num(errs), 0))
    if zoom is not None:                         # scale to the bars not in `zoom`, label the clipped ones
        keep = ~np.isin(xs, zoom)
        lo = min(np.nanmin(vals[keep] - np.nan_to_num(errs[keep])), 0)
        hi = max(np.nanmax(vals[keep] + np.nan_to_num(errs[keep])), 0)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    if zoom is not None:
        for x in zoom:
            v = vals[x]
            if np.isfinite(v) and not (lo - pad <= v <= hi + pad):
                y = hi + pad if v > 0 else lo - pad
                ax.text(x, y - 0.03 * (hi - lo) * np.sign(v), f"{v:+.2f}", rotation=90, fontsize=6 * fs,
                        ha="center", va="top" if v > 0 else "bottom", color="w", fontweight="bold", zorder=5)


CTX_NAME = {"U_v3": "U context (U6:dA17)", "A_v3": "A context (A6:dT17)", "GC_v3": "G:C context (G6:dC15)"}
SPAN_COL = {"linkage": "#d9e6f2", "sugar": "#f2ead9", "base": "#e3f2d9"}


def main():
    global CT
    for ctx in ql.CONTEXTS:
        data = ql.load_context(ctx)
        if not data:
            continue
        CT = ql.CTRL[ctx]
        tags = list(data)
        cache = ql.mp.DATA / f"atom_dN_stats_{ctx}.pkl"     # per-slot stats; delete (or --recompute) to redo
        if cache.exists() and "--recompute" not in sys.argv:
            stats = pd.read_pickle(cache)
        else:
            stats = {t: per_slot(t) for t in tags}
            pd.to_pickle(stats, cache)
        ctrl = stats[CT]
        order = order_of(ctrl)
        xs = np.arange(len(order))
        grp = [ctrl.loc[s, "group"] for s in order]
        spans = []
        for g in ("linkage", "sugar", "base"):
            ii = [i for i, gg in enumerate(grp) if gg == g]
            spans.append((g, (min(ii), max(ii)), SPAN_COL[g]))
        rows = None
        for zoomed in (False, True):
            r = draw(ctx, tags, stats, ctrl, order, xs, spans, zoomed, f"figS_atom_dN_{ctx}{'_zoom' if zoomed else ''}")
            rows = rows or r
        pd.DataFrame(rows).round(6).to_csv(ql.TAB / f"atom_dN_{ctx}.csv", index=False)
        # SI version: landscape pages, zoomed, no title (the caption carries it); control on each page
        bb = [t for t in tags[1:] if cs.SYSTEMS[t][0].startswith("set1/")]
        rest = [t for t in tags[1:] if t not in bb]
        for k, pg in enumerate(([CT] + bb, [CT] + rest), 1):
            draw(ctx, pg, stats, ctrl, order, xs, spans, True, f"figS_atom_dN_{ctx}_land{k}", landscape=True)


def trimmed(png, pad=10):
    from PIL import Image, ImageChops
    im = Image.open(png).convert("RGB")
    bb = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    return np.asarray(im.crop((max(bb[0] - pad, 0), max(bb[1] - pad, 0), bb[2] + pad, bb[3] + pad)))


def draw(ctx, tags, stats, ctrl, order, xs, spans, zoomed, fname, landscape=False):
    if True:
        ncol = 2
        nrow = int(np.ceil(len(tags) / ncol))
        fs = 0.82 if landscape else 1.0
        size = (10.6, 6.95 * max(nrow, 3) / 5) if landscape else (14, 2.35 * nrow)
        if landscape:                                # labelled control nucleotide as a key, left column
            fig = plt.figure(figsize=size)
            gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[0.52] + [1] * ncol)
            axes = np.array([[fig.add_subplot(gs[r, c + 1]) for c in range(ncol)] for r in range(nrow)])
            key = fig.add_subplot(gs[:, 0])
            key.imshow(trimmed(ql.HERE / "chimerax" / "png" / f"labels_{ctx}.png"))
            key.axis("off")
            key.set_title(f"atom key: {CTX_NAME[ctx].split(' (')[0]} control, nucleotide 6", fontsize=8 * fs,
                          loc="left", fontweight="bold")
        else:
            fig, axes = plt.subplots(nrow, ncol, figsize=size, squeeze=False)
        rows = []
        for k, t in enumerate(tags):
            ax = axes[k // ncol][k % ncol]
            lab = cs.SYSTEMS[t][1]
            m = stats[t]
            if t == CT:
                N = np.array([Z[ctrl.loc[s, "element"]] - ctrl.loc[s, "q"] for s in order])
                panel(ax, xs, N, ctrl.loc[order, "sem"].to_numpy(), ["0.45"] * len(xs), [False] * len(xs),
                      [nice(s) for s in order], lab, "N(Ω) / e", spans, fs=fs)
                for s, v in zip(order, N):
                    rows.append(dict(context=ctx, system=lab, slot=s, group=ctrl.loc[s, "group"],
                                     element=ctrl.loc[s, "element"], N_ctrl=v, q_ctrl=ctrl.loc[s, "q"],
                                     sem_ctrl=ctrl.loc[s, "sem"]))
                continue
            vals, errs, cols, hat, labels = [], [], [], [], []
            for s in order:
                c = ctrl.loc[s]
                if s not in m.index:
                    vals.append(np.nan), errs.append(np.nan), cols.append("w"), hat.append(False)
                    labels.append(f"({nice(s)})")
                    rows.append(dict(context=ctx, system=lab, slot=s, group=c["group"], element=c["element"],
                                     present=False))
                    continue
                x = m.loc[s]
                dn = -(x["q"] - c["q"])                      # = N(mod) - N(ctrl) when the element is the same
                se = float(np.hypot(x["sem"], c["sem"]))
                z = dn / se if se > 0 else np.nan
                sub = x["element"] != c["element"]
                vals.append(dn), errs.append(se), hat.append(sub)
                cols.append("0.25" if abs(z) >= 3 else "0.8")
                labels.append(f"{nice(s)}→{x["element"]}" if sub else nice(s))
                rows.append(dict(context=ctx, system=lab, slot=s, group=c["group"], element=c["element"],
                                 present=True, atom_name=x["name"], element_mod=x["element"], n_mod=x["n"],
                                 n_ctrl=c["n"], q_mod=x["q"], q_ctrl=c["q"], sem_mod=x["sem"], sem_ctrl=c["sem"],
                                 dN=dn, sem_diff=se, z=z, substituted=sub))
            big = [i for i, (h, sl) in enumerate(zip(hat, order)) if h or sl == "P"] if zoomed else None
            panel(ax, xs, np.array(vals), np.array(errs), cols, hat, labels, lab, "ΔN(Ω) / e", spans, big, fs=fs)
        for k in range(len(tags), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        handles = [Patch(fc="0.25", ec="k", label="|z| ≥ 3"), Patch(fc="0.8", ec="k", label="|z| < 3"),
                   Patch(fc="w", ec="k", hatch="////", label="element replaced: −Δq (change of N − Z)")]
        handles += [Patch(fc=c, label=g) for g, c in SPAN_COL.items()]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8 * fs, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        if not landscape:
            fig.suptitle(f"{CTX_NAME[ctx]}: "
                         "QTAIM atomic populations of nucleotide 6 (control) and their changes on modification "
                         "(mean over replicas ± SEM)", fontsize=10)
        fig.tight_layout(rect=(0, 0.035 if landscape else 0.025, 1, 1 if landscape else 0.985), h_pad=0.4, w_pad=0.8)
        for ext in ("png", "pdf"):
            fig.savefig(ql.FIG / f"{fname}.{ext}", dpi=200)
        plt.close(fig)
        print(f"[{ctx}] {len(tags)} panels -> figures/{fname}.png")
        return rows


if __name__ == "__main__":
    main()
