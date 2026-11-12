#!/usr/bin/env python3
"""
Preliminary analysis of the central-base-pair QTAIM runs.

  ~/miniforge3/envs/qtaim/bin/python analysis/make_preliminary.py control02/1 [control03/1 ...]

For each replica dir it (re)collects <replica>/qtaim_bp6 into analysis/data/,
then writes to analysis/figures/:
  <tag>_timeseries.png     charges (with fixed AMBER reference), H-bond rho, D...A distance vs time
  <tag>_rho_vs_distance.png rho at H-bond BCPs vs H...A distance (+ exponential fit)
  <tag>_qtaim_vs_amber.png  per-group QTAIM charge distribution vs the fixed AMBER value
and analysis/data/<tag>_summary.csv (mean/sd/min/max of every quantity).
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "central_bp"))
from collect_qtaim import classify, local_path  # noqa: E402
from extract_central_bp import Topology  # noqa: E402

PY = sys.executable
FIG, DATA = ROOT / "analysis" / "figures", ROOT / "analysis" / "data"
# dataviz reference palette, categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUPS = ["linkage", "sugar", "base"]


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def title(ax, text, pad=6):
    ax.set_title(text, loc="left", fontsize=10, color=INK, pad=pad)


def legend_top(ax, extra=None, ncol=4):
    h, l = ax.get_legend_handles_labels()
    if extra:
        h, l = h + [extra[0]], l + [extra[1]]
    ax.legend(h, l, fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=ncol, borderaxespad=0.2, handlelength=1.8)


def label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=7.5, color=INK)


def collect(rep):
    tag = rep.replace("/", "_")
    out = subprocess.run([PY, str(ROOT / "lib/central_bp/collect_qtaim.py"), str(ROOT / rep / "qtaim_bp6"),
                          "--out", str(DATA / tag)], check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "QC-SKIP" in line or "WARN" in line:
            print(line)
    return tag


def hbond_table(rep, bcps):
    """Relabel inter-strand BCPs as donor-H...acceptor and add H...A / D...A distances."""
    qm = ROOT / rep / "qtaim_bp6"
    meta = json.loads(next(qm.glob("frame_*/meta.json")).read_text())
    top = Topology(local_path(meta["topo"]))
    rows = []
    for (frame, t), grp in bcps[bcps.inter_residue.astype(str) == "True"].groupby(["frame", "time_ns"]):
        amap = pd.read_csv(qm / f"frame_{frame:05d}" / f"frame_{frame:05d}.atommap.csv").set_index("label")
        for _, r in grp.iterrows():
            a, b = amap.loc[r.A], amap.loc[r.B]
            h, acc = (a, b) if a.element == "H" else (b, a) if b.element == "H" else (None, None)
            if h is None:
                continue
            don_i = next(j for j in top.nbr[int(h.top_index) - 1])
            don = amap[amap.top_index == don_i + 1].iloc[0]
            lab = (f"{don.resname}{don.resid}:{don.atom_name}–H···"
                   f"{acc.resname}{acc.resid}:{acc.atom_name}")
            key = f"{don.resid}:{don.atom_name}–H···{acc.resid}:{acc.atom_name}"  # residue names may contain digits
            xyz = lambda m: np.array([m.x, m.y, m.z], float)  # noqa: E731
            rows.append(dict(frame=frame, time_ns=t, hbond=lab, key=key, rho=r.rho, lap=r.lap,
                             d_HA=np.linalg.norm(xyz(h) - xyz(acc)),
                             d_DA=np.linalg.norm(xyz(don) - xyz(acc))))
    return pd.DataFrame(rows), top


def amber_reference(top, residues):
    ref = {}
    for r in residues:
        idx = [i for i in range(top.natom) if top.resid[i] == r]
        name = f"{top.resname[idx[0]]}{r}"
        for g in GROUPS:
            ref[(r, g)] = sum(top.charges[i] for i in idx if classify(top, i) == g)
        ref[(r, "nucleotide")] = sum(top.charges[i] for i in idx)
        ref[(r, "name")] = name
    return ref


def fig_timeseries(tag, groups, hb, ref, residues):
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 13), sharex=True, facecolor=SURF,
                             gridspec_kw=dict(hspace=0.45))
    ax = axes[0]; style(ax)
    for k, r in enumerate(residues):
        d = groups[(groups.resid == str(r)) & (groups.group == "nucleotide")].sort_values("time_ns")
        ax.plot(d.time_ns, d.q, color=SERIES[k], lw=2, marker="o", ms=4, label=f"{ref[(r, 'name')]} QTAIM")
        ax.axhline(ref[(r, "nucleotide")], color=SERIES[k], lw=1, ls="--")
        label_end(ax, d.time_ns.iloc[-1], d.q.iloc[-1], ref[(r, "name")])
    amber = (plt.Line2D([], [], color=INK2, lw=1, ls="--"), "AMBER (fixed, same colour)")
    ax.set_ylabel("q(nucleotide) / e", fontsize=9, color=INK)
    legend_top(ax, amber)
    title(ax, "a  Nucleotide charge: QTAIM (per frame) vs AMBER (fixed)", pad=22)

    ax = axes[1]; style(ax)
    r6 = residues[0]
    for k, g in enumerate(GROUPS):
        d = groups[(groups.resid == str(r6)) & (groups.group == g)].sort_values("time_ns")
        ax.plot(d.time_ns, d.q, color=SERIES[k], lw=2, marker="o", ms=4, label=g)
        ax.axhline(ref[(r6, g)], color=SERIES[k], lw=1, ls="--")
        label_end(ax, d.time_ns.iloc[-1], d.q.iloc[-1], g)
    ax.set_ylabel("q(group) / e", fontsize=9, color=INK)
    legend_top(ax, amber)
    title(ax, f"b  {ref[(r6, 'name')]} charge by group (QTAIM and RESP partition atoms differently: compare trends, not offsets)", pad=22)

    order = hb.groupby("hbond").rho.mean().sort_values(ascending=False).index
    for ax, col, ylab, ttl in ((axes[2], "rho", "ρ(BCP) / a.u.", "c  Base-pair H-bonds: electron density at the BCP"),
                               (axes[3], "d_DA", "D···A / Å", "d  Base-pair H-bonds: donor···acceptor distance")):
        style(ax)
        for k, h in enumerate(order):
            d = hb[hb.hbond == h].sort_values("time_ns")
            ax.plot(d.time_ns, d[col], color=SERIES[k], lw=2, marker="o", ms=4, label=h)
            label_end(ax, d.time_ns.iloc[-1], d[col].iloc[-1], h.split("···")[0].split(":")[1] + "···" + h.split("···")[1].split(":")[1])
        ax.set_ylabel(ylab, fontsize=9, color=INK)
        legend_top(ax, ncol=3)
        title(ax, ttl, pad=22)
    axes[3].set_xlabel("MD time / ns", fontsize=9, color=INK)
    fig.suptitle(f"{tag}: central base pair, ωB97X-D/def2-SVP + QTAIM", x=0.06, ha="left",
                 fontsize=12, color=INK)
    fig.savefig(FIG / f"{tag}_timeseries.png", dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def fig_rho_distance(tag, hb):
    fig, ax = plt.subplots(figsize=(7, 4.8), facecolor=SURF)
    style(ax)
    order = hb.groupby("hbond").rho.mean().sort_values(ascending=False).index
    for k, h in enumerate(order):
        d = hb[hb.hbond == h]
        ax.scatter(d.d_HA, d.rho, s=36, color=SERIES[k], edgecolor=SURF, lw=1.5, label=h, zorder=3)
    # fit each acceptor element separately: N acceptors sit above O acceptors at equal H...A
    hb = hb.assign(acc=hb.hbond.str.split("···").str[1].str.split(":").str[1].str[0])
    fits, lines = {}, []
    for (acc, d), ls in zip(sorted(hb.groupby("acc")), ("--", ":", "-.")):
        if len(d) < 5 or d.d_HA.max() - d.d_HA.min() < 0.2:
            continue
        b, lna = np.polyfit(d.d_HA, np.log(d.rho), 1)
        r2 = np.corrcoef(d.d_HA, np.log(d.rho))[0, 1] ** 2
        xs = np.linspace(d.d_HA.min() - 0.05, d.d_HA.max() + 0.05, 100)
        ax.plot(xs, np.exp(lna + b * xs), color=INK2, lw=1.2, ls=ls, zorder=2)
        fits[acc] = (b, np.exp(lna), r2, len(d))
        lines.append(f"{acc} acceptors ({ls}): ρ = {np.exp(lna):.2f}·exp({b:.2f}·d), R² = {r2:.3f}, n = {len(d)}")
    ax.text(0.98, 0.95, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax.set_xlabel("H···A distance / Å", fontsize=9, color=INK)
    ax.set_ylabel("ρ(BCP) / a.u.", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    title(ax, f"{tag}: H-bond density vs geometry (all frames)")
    fig.savefig(FIG / f"{tag}_rho_vs_distance.png", dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    return fits


def fig_qtaim_vs_amber(tag, groups, ref, residues):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURF)
    style(ax)
    cats = [(r, g) for r in residues for g in GROUPS + ["nucleotide"]]
    rng = np.random.default_rng(0)
    for x, (r, g) in enumerate(cats):
        d = groups[(groups.resid == str(r)) & (groups.group == g)].q.values
        c = SERIES[residues.index(r)]
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(d)), d, s=16, color=c, alpha=0.8, edgecolor="none", zorder=3)
        ax.plot([x - 0.3, x + 0.3], [ref[(r, g)]] * 2, color=INK, lw=2, zorder=4)
        ax.annotate(f"sd {d.std():.3f}", (x, d.max()), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{ref[(r, 'name')]}\n{g}" for r, g in cats], fontsize=7.5, color=INK2)
    ax.set_ylabel("charge / e", fontsize=9, color=INK)
    fig.text(0.07, -0.04, "Dots: QTAIM charge, one per frame. Black bar: fixed AMBER (RESP) charge used in the MD.\n"
             "The offset between them comes from the different atom partitioning (QTAIM basins vs ESP fit);"
             " the spread (sd) is the frame-to-frame variation.", fontsize=7.5, color=INK2, va="top")
    title(ax, f"{tag}: per-group charges over time: QTAIM spread vs the single AMBER value")
    fig.savefig(FIG / f"{tag}_qtaim_vs_amber.png", dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for rep in sys.argv[1:] or ["control02/1"]:
        tag = collect(rep)
        groups = pd.read_csv(DATA / f"{tag}_groups.csv", dtype={"resid": str})
        bcps = pd.read_csv(DATA / f"{tag}_bcps.csv")
        hb, top = hbond_table(rep, bcps)
        hb.to_csv(DATA / f"{tag}_hbonds.csv", index=False)
        residues = sorted(int(r) for r in groups[groups.group == "nucleotide"].resid.unique())
        residues.sort(key=lambda r: r != 6)
        ref = amber_reference(top, residues)
        fig_timeseries(tag, groups, hb, ref, residues)
        fits = fig_rho_distance(tag, hb)
        fig_qtaim_vs_amber(tag, groups, ref, residues)
        rows = []
        for r in residues:
            for g in GROUPS + ["nucleotide"]:
                q = groups[(groups.resid == str(r)) & (groups.group == g)].q
                rows.append(dict(quantity=f"q {ref[(r, 'name')]} {g}", n=len(q), mean=q.mean(), sd=q.std(),
                                 min=q.min(), max=q.max(), amber=ref[(r, g)]))
        for h, d in hb.groupby("hbond"):
            for col in ("rho", "d_HA", "d_DA"):
                rows.append(dict(quantity=f"{col} {h}", n=len(d), mean=d[col].mean(), sd=d[col].std(),
                                 min=d[col].min(), max=d[col].max(), amber=np.nan))
        pd.DataFrame(rows).round(4).to_csv(DATA / f"{tag}_summary.csv", index=False)
        print(f"[OK] {tag}: {groups.frame.nunique()} frames, {groups.time_ns.min():.0f}-{groups.time_ns.max():.0f} ns;"
              "; " + "; ".join(f"{k}-acceptor rho = {v[1]:.2f} exp({v[0]:.2f} d), R2 = {v[2]:.3f} (n={v[3]})"
                               for k, v in fits.items()))


if __name__ == "__main__":
    main()
