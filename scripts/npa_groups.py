#!/usr/bin/env python3
"""
NPA (natural population analysis) group charges from the existing g16 logs (pop=nbo), for the ESI comparison
with QTAIM (user, 2026-10-14). No new calculation: reads <replica>/qtaim_bp6/frame_*/frame_*.log.

  ~/software/miniforge3/envs/qtaim/bin/python local_analysis/npa_groups.py

Same frames as the QTAIM analysis (QC-passing, Watson-Crick-paired: the frames in compare_systems.load),
same atom groups (taken per atom from the collected QTAIM atoms table of the same frame), same statistics
(qtaim_local.compare).
Writes
  tables/summary_npa_<ctx>.csv        pooled NPA group charges and differences vs control
  tables/npa_vs_qtaim.csv             QTAIM and NPA differences side by side
  figures/figS_npa_vs_qtaim.png|pdf   NPA vs QTAIM change, per group, three contexts
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qtaim_local as ql  # noqa: E402
from qtaim_local import cs, mp  # noqa: E402
from make_figures import CTX_COL, CTX_ORDER, CTX_SHORT, NEUTRAL  # noqa: E402

ROOT = ql.ROOT


def npa_charges(log):
    """{gauss_index: natural charge} from the last 'Summary of Natural Population Analysis' block."""
    txt = log.read_text(errors="replace")
    i = txt.rfind("Summary of Natural Population Analysis")
    if i < 0:
        return None
    out = {}
    for line in txt[i:].splitlines()[6:]:
        if line.startswith(" ====") or line.strip().startswith("* Total"):
            break
        p = line.split()
        if len(p) >= 3 and p[1].isdigit():
            out[int(p[1])] = float(p[2])
    return out


_ATOMS = {}


def atom_groups(qm_dir, frame):
    """label -> (resid, group) for one frame, from the collected QTAIM atoms table."""
    tag = str(Path(qm_dir).parent.relative_to(ROOT)).replace("/", "_")
    if tag not in _ATOMS:
        a = pd.read_csv(ql.mp.DATA / f"{tag}_atoms.csv", usecols=["frame", "label", "resid", "group"],
                        dtype={"resid": str})
        _ATOMS[tag] = {f: dict(zip(g.label, zip(g.resid, g.group))) for f, g in a.groupby("frame")}
    return _ATOMS[tag][frame]


def npa_groups(d):
    rows = []
    for (qm, fr, rep), _ in d["groups"].groupby(["qm_dir", "frame", "replica"]):
        fd = Path(qm) / f"frame_{fr:05d}"
        q = npa_charges(fd / f"{fd.name}.log")
        amap = pd.read_csv(fd / f"{fd.name}.atommap.csv")
        grp = atom_groups(qm, fr)
        if q is None or len(q) != len(amap):
            print(f"[WARN] {fd}: NPA block missing or atom count differs; frame skipped")
            continue
        sums = {}
        for gi, lab in zip(amap.gauss_index, amap.label):
            rid, g = grp[lab]
            sums[(rid, g)] = sums.get((rid, g), 0.0) + q[gi]
            if g != "cap":
                sums[(rid, "nucleotide")] = sums.get((rid, "nucleotide"), 0.0) + q[gi]
        rows += [dict(qm_dir=qm, frame=fr, replica=rep, resid=r, group=g, q=v) for (r, g), v in sums.items()]
    return pd.DataFrame(rows)


def main():
    out = []
    for ctx in ql.CONTEXTS:
        data = ql.load_context(ctx)
        npa = {}
        for tag, d in data.items():
            npa[tag] = {**d, "groups": npa_groups(d)}
            print(f"[{ctx}] {cs.SYSTEMS[tag][1]}: {npa[tag]['groups'][['qm_dir', 'frame']].drop_duplicates().shape[0]} frames")
        s = ql.compare(npa, "q", ql.GROUPS)
        s.to_csv(ql.TAB / f"summary_npa_{ctx}.csv", index=False, float_format="%.6g")
        qt = pd.read_csv(ql.TAB / f"summary_{ctx}.csv")
        m = s.merge(qt[["quantity", "system", "diff", "sem_diff", "z"]], on=["quantity", "system"],
                    suffixes=("_npa", "_qtaim"))
        out.append(m)
    m = pd.concat(out, ignore_index=True)
    m.to_csv(ql.TAB / "npa_vs_qtaim.csv", index=False, float_format="%.6g")
    figure(m)


def figure(m):
    m = m[m.diff_npa.notna()]
    titles = ["res. 6 linkage", "res. 6 sugar", "res. 6 base", "res. 6 nucleotide", "partner nucleotide"]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.9), facecolor="white", gridspec_kw=dict(wspace=0.45))
    for ax, k, t in zip(axes, ql.GROUPS, titles):
        q = ql.qname("q", k)
        d = m[m.quantity == q]
        if k[1] in ("linkage", "nucleotide") and k[0] == "6":
            d = d[~d.chem.isin(NEUTRAL)]          # neutral linkages: formal-charge shift, listed in the text
        mp.style(ax)
        ax.set_facecolor("white")
        lim = 1.1 * max(d.diff_qtaim.abs().max(), d.diff_npa.abs().max())
        ax.plot([-lim, lim], [-lim, lim], color=mp.GRID, lw=0.8, zorder=1)
        ax.axhline(0, color=mp.INK2, lw=0.5)
        ax.axvline(0, color=mp.INK2, lw=0.5)
        for c in CTX_ORDER:
            e = d[d.context == c]
            ax.scatter(e.diff_qtaim, e.diff_npa, s=9, color=CTX_COL[c], edgecolor="none", zorder=3)
        r = np.corrcoef(d.diff_qtaim, d.diff_npa)[0, 1]
        sl = np.polyfit(d.diff_qtaim, d.diff_npa, 1)[0]
        ax.text(0.04, 0.96, f"r = {r:.2f}\nslope {sl:.2f}", transform=ax.transAxes, fontsize=6, va="top",
                color=mp.INK)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(t, fontsize=7.5, loc="left", color=mp.INK)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Δq NPA / e", fontsize=7.5)
    axes[2].set_xlabel("Δq QTAIM / e", fontsize=7.5)
    h = [plt.Line2D([], [], marker="o", ls="", ms=4, color=CTX_COL[c]) for c in CTX_ORDER]
    fig.legend(h, [CTX_SHORT[c] for c in CTX_ORDER], loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(ql.FIG / f"figS_npa_vs_qtaim.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
