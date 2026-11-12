#!/usr/bin/env python3
"""
Control vs modified comparison of the central-base-pair QTAIM results.

  ~/miniforge3/envs/qtaim/bin/python analysis/compare_systems.py

Uses every system in SYSTEMS that has finished frames (QC-passing only), grouped
by sequence context (the H-bonds differ between A:T and G:C pairs). Per context
it writes to analysis/figures/:
  compare_<ctx>_distributions.png  per-system distributions of base-pair H-bond rho and
                                   of the residue-6 / partner charges (control = grey)
  compare_<ctx>_timeseries.png     small multiples: each modified system's time series
                                   with the control overlaid in grey
and analysis/data/compare_<ctx>_summary.csv.

Replicas: every replica <sys>/<r> (r = 1..3) with QTAIM data is pooled. The standard error
is taken across replica means when >= 2 replicas have >= 10 frames (independent MD runs);
otherwise it falls back to the block SEM of the single replica. Figures: dots = frames,
small diamonds = replica means, bar = pooled mean. Time series show replica 1.

Paired frames only (default): a frame counts only if the central base pair is Watson-Crick
paired, i.e. its imino H-bond (T:N3-H...A:N1 or G:N1-H...C:N3) has a bond critical point and
N...N <= 3.5 A. All control frames pass.
Frames where the pair has opened (seen for some 2'-OMe / MOE replicas) would otherwise mix a
structural change into the charge differences. --all-frames disables the filter. The summary
CSV gives the fraction of frames kept (frac_paired) per system.

Colour = linkage chemistry (fixed, never re-assigned); filled marker = Rp, open = Sp.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_preliminary as mp  # noqa: E402

GREY = "#8a8985"
# chemistry -> fixed categorical slot (dataviz reference palette order)
CHEM = {"PS": mp.SERIES[0], "PS2": mp.SERIES[1], "MsPA": mp.SERIES[2], "MP": mp.SERIES[3],
        "guan": mp.SERIES[4], "2F": mp.SERIES[5], "2OMe": mp.SERIES[6], "MOE": mp.SERIES[7],
        "cEt": "#4a4a4a"}
# residue codes and names follow ~/work/residues-library/22-0e84.docx (modXNA library):
# F1x = phosphate + 2'-F, 21x = phosphate + 2'-O-methyl (DPO OME), 51x = phosphate + MOE (BPO MOE)
# tag: (system dir without replica, label, chemistry or None for control, stereo, context)
#
# FORCE FIELD (user decision 2026-10-05): the main results use only V3 = standard OL3/OL21 + OPC
# (lib/rebuild_std recipe; every V3 topology passes lib/build_check/gate_v3.py). V3 systems live in
# *_v3 directories (the first V3 rebuilds are *_std directories, reached through *_v3 symlinks) and
# are compared with control0x_v3 in the contexts "A_v3", "GC_v3", "U_v3".
# The older builds (V1 = original controls + set2/set3 originals, V2 = set1 + control0x_std) are WRONG
# and are not analysed (user, 2026-10-05); their registry entries below are kept only as a record and
# are never run (main() loops over the V3 contexts only).
SET1 = (("sys01", "PS2", "PS2", ""), ("sys02", "Rp-PS", "PS", "Rp"), ("sys03", "Sp-PS", "PS", "Sp"),
        ("sys06", "Rp-MsPA", "MsPA", "Rp"), ("sys07", "Sp-MsPA", "MsPA", "Sp"), ("sys04_q0", "Rp-MP", "MP", "Rp"),
        ("sys05_q0", "Sp-MP", "MP", "Sp"), ("sys08_q0", "Rp-guan", "guan", "Rp"), ("sys09_q0", "Sp-guan", "guan", "Sp"))
SUGAR = (("2'-F", "2F"), ("2'-OMe", "2OMe"), ("2'-MOE", "MOE"))
# set3 combinations (labels from the P-stereo audit; sys01 is Sp in seq2 but Rp in seq3)
SET3 = {"seq2": (("sys01", "Sp-PS+2'-F", "2F", "Sp"), ("sys02", "Rp-PS+2'-OMe", "2OMe", "Rp"),
                 ("sys03", "Rp-PS+2'-MOE", "MOE", "Rp"), ("sys04", "cEt", "cEt", ""),
                 ("sys05", "Sp-MsPA+2'-MOE", "MOE", "Sp")),
        "seq3": (("sys01", "Rp-PS+2'-F", "2F", "Rp"), ("sys02", "Rp-PS+2'-OMe", "2OMe", "Rp"),
                 ("sys03", "Rp-PS+2'-MOE", "MOE", "Rp"), ("sys04", "cEt", "cEt", ""),
                 ("sys05", "Sp-MsPA+2'-MOE", "MOE", "Sp"))}
SYSTEMS = {}
# --- V3 (main results) ---
# seq1 (U_v3) uses ONLY the restrained runs (user 2026-10-09: all 13 seq1 systems re-run with flat-bottom
# restraints on the terminal U:A pairs, <system>/seq1_rst and control01_v3_rst); unrestrained seq1 = stability only.
for ctx, seq, ctrl, sdir in (("A_v3", "seq2", "control02_v3", "seq2"), ("GC_v3", "seq3", "control03_v3", "seq3"),
                             ("U_v3", "seq1", "control01_v3_rst", "seq1_rst")):
    SYSTEMS[ctrl] = (ctrl, "control (PO)", None, "", ctx)
    for sysn, lab, chem, st in SET1:
        SYSTEMS[f"set1_{sysn}_v3_{seq}"] = (f"set1/{sysn}_v3/{sdir}", lab, chem, st, ctx)
    for (lab, chem), sysn in zip(SUGAR, ("sys01", "sys02", "sys03")):
        SYSTEMS[f"set2_{sysn}_v3_{seq}"] = (f"set2/{sysn}_v3/{sdir}", lab, chem, "", ctx)
    for sysn, lab, chem, st in SET3.get(seq, ()):
        SYSTEMS[f"set3_{sysn}_v3_{seq}"] = (f"set3/{sysn}_v3/{seq}", lab, chem, st, ctx)
# --- legacy V1/V2: NOT USED (wrong parameters; record only) ---
# U context legacy control is control01_std (V2, MD stopped 2026-10-05); set2 seq1 exists only as V3.
for ctx, seq, ctrl in (("A", "seq2", "control02"), ("GC", "seq3", "control03"), ("U", "seq1", "control01_std")):
    SYSTEMS[ctrl] = (ctrl, "control (PO)", None, "", ctx)
    for sysn, lab, chem, st in SET1:
        SYSTEMS[f"set1_{sysn}_{seq}"] = (f"set1/{sysn}/{seq}", lab, chem, st, ctx)
    if ctx != "U":
        for (lab, chem), sysn in zip(SUGAR, ("sys01", "sys02", "sys03")):
            SYSTEMS[f"set2_{sysn}_{seq}"] = (f"set2/{sysn}/{seq}", lab, chem, "", ctx)
CTX_TITLE = {"A": "A context: A6:DT17 (rA11:dT11) [legacy FF, SI]", "GC": "404D context: G6:DC15 [legacy FF, SI]",
             "U": "U context: U6:DA17 (rU11:dA11) [legacy FF, SI]",
             "A_v3": "A context: A6:DT17 (rA11:dT11), V3 force field", "GC_v3": "404D context: G6:DC15, V3 force field",
             "U_v3": "U context: U6:DA17 (rU11:dA11), V3 force field, terminal-pair restraints"}
ALL_FRAMES = "--all-frames" in sys.argv
IMINO = re.compile(r"^\d+:N3–H···6:N1$|^6:N1–H···\d+:N3$|^6:N3–H···\d+:N1$")  # A:T, G:C, U:A imino H-bond
PAIRED_DA = 3.5
QTY = [("linkage", 6), ("sugar", 6), ("base", 6), ("nucleotide", 6), ("nucleotide", None)]


def colour(chem):
    return GREY if chem is None else CHEM[chem]


def hkey(label):
    """'PRA6:N6–H···DT17:O4' -> '6:N6–H···17:O4' (residue names differ between systems)."""
    return re.sub(r"[A-Z]+(\d+):", r"\1:", label)


def block_sem(x, nblocks=5):
    x = np.asarray(x, float)
    if len(x) < 2 * nblocks:
        return np.nan
    means = [b.mean() for b in np.array_split(x, nblocks)]
    return np.std(means, ddof=1) / np.sqrt(nblocks)


def load(tag, sysdir):
    gs, hbs = [], []
    for r in (1, 2, 3):
        rep = f"{sysdir}/{r}"
        if not any((mp.ROOT / rep / "qtaim_bp6").glob("frame_*/frame_*.sum")):
            continue
        t = mp.collect(rep)
        g = pd.read_csv(mp.DATA / f"{t}_groups.csv", dtype={"resid": str})
        if g.empty:
            continue
        hb, _ = mp.hbond_table(rep, pd.read_csv(mp.DATA / f"{t}_bcps.csv"))
        g["replica"], hb["replica"] = r, r
        gs.append(g); hbs.append(hb)
    if not gs:
        return None
    g, hb = pd.concat(gs, ignore_index=True), pd.concat(hbs, ignore_index=True)
    if "key" not in hb:
        hb["key"] = hb.hbond.map(hkey)
    im = hb[hb.key.str.match(IMINO) & (hb.d_DA <= PAIRED_DA)]
    paired = set(zip(im.replica, im.frame))
    frames = g[["replica", "frame"]].drop_duplicates()
    ok = pd.Series([(a, b) in paired for a, b in frames.values], index=frames.index)
    frac = ok.mean()
    if not ALL_FRAMES:
        keep = set(map(tuple, frames[ok].values))
        g = g[[(a, b) in keep for a, b in zip(g.replica, g.frame)]]
        hb = hb[[(a, b) in keep for a, b in zip(hb.replica, hb.frame)]]
    partner = sorted(int(r) for r in g[g.group == "nucleotide"].resid.unique() if r != "6")[0]
    return dict(groups=g, hb=hb, partner=partner, replicas=sorted(g.replica.unique()), frac_paired=frac)


def pooled_stats(df):
    """df with columns replica, v -> n, nrep, mean, sd, sem, sem_kind, replica means."""
    v = df.v.values
    rm = df.groupby("replica").v.agg(["mean", "count"])
    good = rm[rm["count"] >= 10]
    if len(good) >= 2:
        sem, kind = good["mean"].std(ddof=1) / np.sqrt(len(good)), f"replicas({len(good)})"
    else:
        sem, kind = block_sem(v), "block"
    return dict(n=len(v), nrep=len(rm), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                sem=sem, sem_kind=kind, rep_means=rm["mean"].to_dict())


def marker_kw(chem, stereo, size=16):
    c = colour(chem)
    face = "white" if stereo == "Sp" else c
    return dict(s=size, facecolor=face, edgecolor=c, linewidth=1.2)


def fig_distributions(ctx, data, order):
    ctrl = order[0]
    hkeys = [k for k in data[ctrl]["hb"].groupby("key").rho.mean().sort_values(ascending=False).index]
    ncol = max(len(hkeys), len(QTY))
    fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 8), facecolor=mp.SURF,
                             gridspec_kw=dict(hspace=0.55, wspace=0.35))
    rng = np.random.default_rng(1)
    rows = []

    def strip(ax, vals_by_sys, ylab, name):
        mp.style(ax)
        base = pooled_stats(vals_by_sys[ctrl])
        for x, tag in enumerate(order):
            df = vals_by_sys.get(tag)
            if df is None or len(df) == 0:
                continue
            _, lab, chem, st, _ = SYSTEMS[tag]
            v = df.v.values
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, alpha=0.6, zorder=3, **marker_kw(chem, st, 12))
            p = pooled_stats(df)
            for k, (r, m) in enumerate(sorted(p["rep_means"].items())):
                ax.scatter(x + 0.34, m, marker="D", s=22, color=mp.INK2, edgecolor=mp.SURF, lw=1, zorder=5)
            ax.plot([x - 0.28, x + 0.28], [p["mean"]] * 2, color=mp.INK, lw=2, zorder=4)
            d = p["mean"] - base["mean"]
            se = np.sqrt(p["sem"] ** 2 + base["sem"] ** 2) if tag != ctrl else np.nan
            rows.append(dict(quantity=name, system=lab, n=p["n"], replicas=p["nrep"], mean=p["mean"], sd=p["sd"],
                             sem=p["sem"], sem_kind=p["sem_kind"], diff_vs_control=d, frac_paired=round(data[tag]["frac_paired"], 3),
                             z=(d / se if se and np.isfinite(se) and se > 0 else np.nan)))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([SYSTEMS[t][1] for t in order], rotation=45, ha="right", fontsize=7.5, color=mp.INK2)
        ax.set_ylabel(ylab, fontsize=8.5, color=mp.INK)

    for j, k in enumerate(hkeys):
        vals = {t: data[t]["hb"].loc[data[t]["hb"].key == k, ["replica", "rho"]].rename(columns={"rho": "v"})
                for t in order}
        strip(axes[0, j], vals, "ρ(BCP) / a.u.", f"rho {k}")
        mp.title(axes[0, j], k.replace("–H···", "–H···"))
    for j in range(len(hkeys), ncol):
        axes[0, j].axis("off")
    for j, (grp, rid) in enumerate(QTY):
        vals = {}
        for t in order:
            g = data[t]["groups"]
            r = str(rid if rid else data[t]["partner"])
            vals[t] = g[(g.resid == r) & (g.group == grp)][["replica", "q"]].rename(columns={"q": "v"})
        who = "residue 6" if rid else "partner"
        strip(axes[1, j], vals, "q / e", f"q {who} {grp}")
        mp.title(axes[1, j], f"{who}: {grp}")
    for j in range(len(QTY), ncol):
        axes[1, j].axis("off")
    fig.suptitle(f"{CTX_TITLE[ctx]}: control (grey) vs modifications at residue 6", x=0.02,
                 ha="left", fontsize=12, color=mp.INK)
    fig.text(0.02, -0.02, "Dots: one QTAIM frame each (0-500 ns, every 10 ns, all replicas); diamonds: replica means; bar: pooled mean.\n Colour = chemistry (backbone or sugar),"
             " filled = Rp, open = Sp. Top row: electron density at the base-pair H-bond critical points;"
             " bottom row: QTAIM charges.", fontsize=8, color=mp.INK2)
    fig.savefig(mp.FIG / f"compare_{ctx}_distributions.png", dpi=150, bbox_inches="tight", facecolor=mp.SURF)
    plt.close(fig)
    return rows


def fig_timeseries(ctx, data, order):
    ctrl, mods = order[0], order[1:]
    if not mods:
        return
    hk = data[ctrl]["hb"].groupby("key").rho.mean().sort_values(ascending=False).index[0]
    r1 = lambda df: df[df.replica == 1]  # noqa: E731  time series: replica 1 only
    rows_q = [("ρ strongest WC H-bond / a.u.", lambda d: r1(d["hb"])[r1(d["hb"]).key == hk][["time_ns", "rho"]].rename(columns={"rho": "y"})),
              ("q residue-6 linkage / e", lambda d: r1(d["groups"])[(r1(d["groups"]).resid == "6") & (r1(d["groups"]).group == "linkage")][["time_ns", "q"]].rename(columns={"q": "y"})),
              ("q residue-6 nucleotide / e", lambda d: r1(d["groups"])[(r1(d["groups"]).resid == "6") & (r1(d["groups"]).group == "nucleotide")][["time_ns", "q"]].rename(columns={"q": "y"}))]
    fig, axes = plt.subplots(len(rows_q), len(mods), figsize=(3.2 * len(mods), 2.6 * len(rows_q)), sharex=True,
                             sharey="row", facecolor=mp.SURF, squeeze=False, gridspec_kw=dict(hspace=0.35, wspace=0.12))
    for i, (ylab, get) in enumerate(rows_q):
        c = get(data[ctrl]).sort_values("time_ns")
        for j, t in enumerate(mods):
            ax = axes[i, j]; mp.style(ax)
            _, lab, chem, st, _ = SYSTEMS[t]
            ax.plot(c.time_ns, c.y, color=GREY, lw=1.5, label="control")
            d = get(data[t]).sort_values("time_ns")
            ax.plot(d.time_ns, d.y, color=colour(chem), lw=2, ls="-" if st != "Sp" else "--", label=lab)
            if i == 0:
                mp.title(ax, lab)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8.5, color=mp.INK)
            if i == len(rows_q) - 1:
                ax.set_xlabel("MD time / ns", fontsize=8.5, color=mp.INK)
    handles = [plt.Line2D([], [], color=GREY, lw=1.5), plt.Line2D([], [], color=mp.INK2, lw=2),
               plt.Line2D([], [], color=mp.INK2, lw=2, ls="--")]
    fig.legend(handles, ["control", "modification, Rp (colour = chemistry)", "modification, Sp"],
               fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.99), ncol=3)
    fig.suptitle(f"{CTX_TITLE[ctx]}: time series, each modification vs control (grey); strongest H-bond = {hk}",
                 x=0.02, ha="left", fontsize=11, color=mp.INK, y=1.02)
    fig.savefig(mp.FIG / f"compare_{ctx}_timeseries.png", dpi=150, bbox_inches="tight", facecolor=mp.SURF)
    plt.close(fig)


def main():
    mp.FIG.mkdir(parents=True, exist_ok=True)
    # V3 only (user, 2026-10-05): V1/V2 parameter sets are wrong and are not compared at all.
    for ctx in ("A_v3", "GC_v3", "U_v3"):
        tags = [t for t, v in SYSTEMS.items() if v[4] == ctx]
        data = {}
        for t in tags:
            d = load(t, SYSTEMS[t][0])
            if d is not None:
                data[t] = d
        order = [t for t in tags if t in data]
        if not order or SYSTEMS[order[0]][2] is not None:
            print(f"[{ctx}] control not available yet; skipping")
            continue
        rows = fig_distributions(ctx, data, order)
        fig_timeseries(ctx, data, order)
        pd.DataFrame(rows).round(5).to_csv(mp.DATA / f"compare_{ctx}_summary.csv", index=False)
        print(f"[{ctx}] systems with data: " + ", ".join(
            f"{SYSTEMS[t][1]} ({len(data[t]['groups'][['replica', 'frame']].drop_duplicates())} fr, paired {data[t]['frac_paired']:.0%}, rep {[int(r) for r in data[t]['replicas']]})"
            for t in order))


if __name__ == "__main__":
    main()
