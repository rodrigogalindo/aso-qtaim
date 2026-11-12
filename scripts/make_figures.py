#!/usr/bin/env python3
"""
Report figures and tables (local analysis).

  ~/software/miniforge3/envs/qtaim/bin/python local_analysis/make_figures.py

Writes
  (V3 contexts only: A_v3, GC_v3, U_v3)
  figures/fig1_dq_<ctx>.png|pdf      forest plot: delta q vs control per system, by group
  figures/fig2_drho_<ctx>.png|pdf    forest plot: delta rho at the Watson-Crick H-bond BCPs
  tables/summary_<ctx>.csv           mean, sd, SEM, delta, z, replicas, SEM kind, H-bond coverage
  tables/frame_status.csv            frames planned / finished / used / excluded / marginalL / NNA
A context only gets forest plots once a modified system has data.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qtaim_local as ql  # noqa: E402
from qtaim_local import cs, mp  # noqa: E402

import plotfont  # noqa: E402
plotfont.apply()
plt.rcParams.update({"font.size": 8})
NEUTRAL = {"MP", "guan"}  # neutral linkages: fragment charge -1
# per figure (user, 2026-10-14): bars for the charge changes, points (forest plot) for the rho changes
STYLE = {"dq": "bars", "drho": "points"}


def bar(ax, i, d, e, chem, stereo):
    """Horizontal bar from 0 to d (solid: Rp or no stereo, light: Sp) with black +-1 SE (thick) and +-2 SE
    (thin) error bars at its end."""
    c = cs.colour(chem)
    ax.barh(i, d, height=0.64, color=c, alpha=0.38 if stereo == "Sp" else 1.0, edgecolor=c, linewidth=0.8, zorder=2)
    if np.isfinite(e):
        ax.plot([d - 2 * e, d + 2 * e], [i, i], color=mp.INK, lw=0.8, solid_capstyle="butt", zorder=3)
        ax.plot([d - e, d + e], [i, i], color=mp.INK, lw=2.2, solid_capstyle="butt", zorder=4)
CTX_LONG = {"A_v3": "A context (A6:dT17, rA11:dT11)", "GC_v3": "G:C context (G6:dC15, 404D)",
            "U_v3": "U context (U6:dA17, rU11:dA11)"}

def row_label(r):
    if not np.isfinite(r.sem_diff):
        return f"{r.system}  (n={int(r.n)}, no SEM)"
    ff = "" if r.ff_identical_to_control else " §"   # safety flag: should never appear for V3 builds
    if r.sem_kind == "block":
        return f"{r.system}{ff}  (1 rep*)"
    nrep = int(r.sem_kind.split("(")[1].rstrip(")"))
    return f"{r.system}{ff}  ({nrep} rep{'†' if nrep == 2 else ''})"


def scale_tick(scale):
    """xlim always spans at least one tick step either side of zero (step is given in plotted units)."""
    return 1.0


def span(df, quantities, scale):
    """(lo, hi) of the in-scale differences +- 2 SE (neutral linkages excluded), in plotted units."""
    mods = df[df["diff"].notna()]
    inr = mods[mods.quantity.isin(quantities) & ~mods.chem.isin(NEUTRAL)]
    e2 = inr.sem_diff.fillna(0) * 2            # rows without an SEM (< 10 frames) count with their mean only
    return ((inr["diff"] - e2) * scale).min(), ((inr["diff"] + e2) * scale).max()


def forest(df, quantities, titles, xlabel, scale, path, suptitle, note, step, lohi=None, style="points"):
    """lohi: common (lo, hi) data range shared by the contexts, so that their figures have the same x axis."""
    mods = df[df["diff"].notna()]
    systems = [s for s in ql.ORDER if s in set(mods.system)]
    if not systems:
        return False
    ny = len(systems)
    fig, axes = plt.subplots(1, len(quantities), figsize=(7.0, max(0.42 * ny + 1.5, 1.9)), sharey=True, sharex=True,
                             facecolor=mp.SURF, gridspec_kw=dict(wspace=0.42))
    axes = np.atleast_1d(axes)
    # x range from the in-scale points; neutral-linkage systems (fragment charge -1) shift the
    # linkage/nucleotide charge by ~+0.9 e and are written at the panel edge instead
    lo, hi = lohi if lohi is not None else span(df, quantities, scale)
    pad = 0.22 * (max(hi, 0, step) - min(lo, 0, -step))  # keeps outer tick labels clear of the panel edges
    xlim = (min(lo, 0, -step * scale_tick(scale)) - pad, max(hi, 0, step * scale_tick(scale)) + pad)
    for ax, q, t in zip(axes, quantities, titles):
        mp.style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=mp.GRID, lw=0.6)
        ax.axvline(0, color=mp.INK2, lw=0.9, zorder=1)
        for i, s in enumerate(systems):
            r = mods[(mods.quantity == q) & (mods.system == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            c = cs.colour(r.chem)
            d, e = r["diff"] * scale, r.sem_diff * scale
            if style == "bars":
                if not xlim[0] <= d <= xlim[1]:   # neutral linkage: bar to the edge, value written
                    bar(ax, i, xlim[1] - 0.02 * (xlim[1] - xlim[0]), np.nan, r.chem, r.stereo)
                    ax.annotate(f"{d:+.2f} ►", (xlim[1], i), xytext=(-2, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=6.5, color="white" if r.stereo != "Sp" else mp.INK,
                                zorder=5)
                else:
                    bar(ax, i, d, e, r.chem, r.stereo)
                continue
            if not np.isfinite(e):   # too few frames for an SEM: marker only
                ax.scatter([d], [i], zorder=4, **cs.marker_kw(r.chem, r.stereo, 30))
                continue
            if not xlim[0] <= d <= xlim[1]:
                ax.annotate(f"{d:+.2f} ►", (xlim[1], i), ha="right", va="center", fontsize=7, color=mp.INK2)
                continue
            ax.plot([d - 2 * e, d + 2 * e], [i, i], color=c, lw=1.0, alpha=0.55, solid_capstyle="round", zorder=2)
            ax.plot([d - e, d + e], [i, i], color=c, lw=2.6, solid_capstyle="round", zorder=3)
            ax.scatter([d], [i], zorder=4, **cs.marker_kw(r.chem, r.stereo, 30))
        ax.set_title(t, loc="left", fontsize=8.5, color=mp.INK, pad=5)
        ax.tick_params(axis="y", length=0)
        st = step
        while (xlim[1] - xlim[0]) / st > (3.2 if len(quantities) >= 5 else 5):   # fewer ticks in narrow panels
            st *= 2
        ax.xaxis.set_major_locator(MultipleLocator(st))
        if style == "bars":   # unlabelled gridlines halfway between the labelled ticks
            ax.xaxis.set_minor_locator(MultipleLocator(st / 2))
            ax.grid(axis="x", which="minor", color=mp.GRID, lw=0.5)
            ax.tick_params(axis="x", which="minor", length=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else f"{v:+g}".replace("-", "−")))
    labs = [row_label(mods[mods.system == s].iloc[0]) for s in systems]
    axes[0].set_xlim(*xlim)
    axes[0].set_yticks(range(ny))
    axes[0].set_yticklabels(labs, fontsize=8, color=mp.INK)
    axes[0].set_ylim(ny - 0.4, -0.6)
    axes[len(axes) // 2].set_xlabel(xlabel, fontsize=8.5, color=mp.INK, labelpad=6)
    # article version first: no title/footnote (the LaTeX caption carries them), white background
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
    art = path.with_name(path.name + "_article")
    for ext in ("png", "pdf"):
        fig.savefig(art.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.patch.set_facecolor(mp.SURF)
    for ax in axes:
        ax.set_facecolor(mp.SURF)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=9.5, color=mp.INK, y=1.0 if ny > 2 else 1.08)
    fig.canvas.draw()
    bottom = min(a.get_tightbbox().transformed(fig.transFigure.inverted()).y0 for a in axes)
    fig.text(0.01, bottom - 0.03, note, fontsize=6.8, color=mp.INK2, va="top", wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor=mp.SURF)
    plt.close(fig)
    return True


NOTE = ("Marker = difference of pooled means (modified − control); thick bar ±1 SEM, thin bar ±2 SEM of the difference. "
        "Colour = chemistry; filled = Rp, open = Sp. "
        "► off scale (neutral linkage, fragment charge −1). SEM across replica means; * one replica: block SEM (5 blocks) within it; † only 2 replicas: SEM may be unrealistically small. "
        "§ force field differs from the control outside residue 6 (should not occur).")


# ---- combined figures (user, 2026-10-14): the three contexts side by side in one figure per quantity ----
CTX_ORDER = ["U_v3", "A_v3", "GC_v3"]                      # seq1, seq2, seq3
CTX_SHORT = {"U_v3": "U context (U6:dA17)", "A_v3": "A context (A6:dT17)", "GC_v3": "G:C context (G6:dC15)"}
CTX_COL = {"U_v3": "#E69F00", "A_v3": "#0072B2", "GC_v3": "#009E73"}   # Okabe-Ito, colour-blind safe
# set3/sys01 is Sp-PS + 2'-F in the A context but Rp-PS + 2'-F in the G:C context: one row
ROW_OF = {"Rp-PS+2'-F": "Sp-PS+2'-F"}
ROW_LABEL = {"Sp-PS+2'-F": "PS+2'-F‡"}
# Watson-Crick H-bonds matched by position in the pair: imino, major groove, minor groove
HB_COLS = [("imino N–H···N", {"U_v3": "6:N3–H···17:N1", "A_v3": "17:N3–H···6:N1", "GC_v3": "6:N1–H···15:N3"}),
           ("major groove N–H···O", {"U_v3": "17:N6–H···6:O4", "A_v3": "6:N6–H···17:O4", "GC_v3": "15:N4–H···6:O6"}),
           ("minor groove X–H···O2", {"U_v3": "17:C2–H···6:O2", "A_v3": "6:C2–H···17:O2", "GC_v3": "6:N2–H···15:O2"})]


def combined(summaries, cols, titles, xlabel, scale, path, step, style):
    """cols: list of {ctx: quantity}; rows = modifications, three sub-rows (U, A, G:C) per modification."""
    ctxs = [c for c in CTX_ORDER if c in summaries]
    mods = {c: summaries[c][summaries[c]["diff"].notna()].assign(row=lambda d: d.system.replace(ROW_OF)) for c in ctxs}
    rows = [s for s in ql.ORDER if s not in ROW_OF and any(s in set(m.row) for m in mods.values())]
    ny, off = len(rows), {c: (k - (len(ctxs) - 1) / 2) * 0.27 for k, c in enumerate(ctxs)}
    fig, axes = plt.subplots(1, len(cols), figsize=(7.0, 0.36 * ny + 1.3), sharey=True, sharex=True,
                             facecolor="white", gridspec_kw=dict(wspace=0.2))
    axes = np.atleast_1d(axes)
    # x range: everything +- 2 SE except the formal-charge shift of the neutral linkages (linkage, nucleotide 6)
    v = []
    for c in ctxs:
        for col in cols:
            r = mods[c][mods[c].quantity == col[c]]
            if col[c] in (ql.qname("q", ("6", "linkage")), ql.qname("q", ("6", "nucleotide"))):
                r = r[~r.chem.isin(NEUTRAL)]
            e2 = r.sem_diff.fillna(0) * 2
            v += list((r["diff"] - e2) * scale) + list((r["diff"] + e2) * scale)
    lo, hi = min(v), max(v)
    pad = 0.12 * (max(hi, 0, step) - min(lo, 0, -step))
    xlim = (min(lo, 0, -step) - pad, max(hi, 0, step) + pad)
    for ax, col, t in zip(axes, cols, titles):
        mp.style(ax)
        ax.set_facecolor("white")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=mp.GRID, lw=0.6)
        ax.axvline(0, color=mp.INK2, lw=0.9, zorder=1)
        for i in range(1, ny, 2):          # light bands on every other modification
            ax.axhspan(i - 0.5, i + 0.5, color="#f3f2ef", zorder=0, lw=0)
        for i, rname in enumerate(rows):
            for c in ctxs:
                r = mods[c][(mods[c].quantity == col[c]) & (mods[c].row == rname)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y, k = i + off[c], CTX_COL[c]
                d, e = r["diff"] * scale, r.sem_diff * scale
                sp = r.stereo == "Sp"
                if not xlim[0] <= d <= xlim[1]:            # neutral linkage: to the edge, value written
                    edge = xlim[1] - 0.01 * (xlim[1] - xlim[0])
                    if style == "bars":
                        ax.barh(y, edge, height=0.25, color=k, alpha=0.4 if sp else 1.0, edgecolor=k, lw=0.6, zorder=2)
                    ax.annotate(f"{d:+.2f}►", (xlim[1], y), xytext=(-1.5, 0), textcoords="offset points", ha="right",
                                va="center", fontsize=5.2, color=mp.INK if (sp or style != "bars") else "white", zorder=5)
                    continue
                if style == "bars":
                    ax.barh(y, d, height=0.25, color=k, alpha=0.4 if sp else 1.0, edgecolor=k, lw=0.6, zorder=2)
                    if np.isfinite(e):
                        ax.plot([d - 2 * e, d + 2 * e], [y, y], color=mp.INK, lw=0.6, solid_capstyle="butt", zorder=3)
                        ax.plot([d - e, d + e], [y, y], color=mp.INK, lw=1.6, solid_capstyle="butt", zorder=4)
                else:
                    if np.isfinite(e):
                        ax.plot([d - 2 * e, d + 2 * e], [y, y], color=k, lw=0.8, alpha=0.6, solid_capstyle="round", zorder=2)
                        ax.plot([d - e, d + e], [y, y], color=k, lw=2.0, solid_capstyle="round", zorder=3)
                    ax.scatter([d], [y], s=14, facecolor="white" if sp else k, edgecolor=k, linewidth=1.0, zorder=4)
        ax.set_title(t, loc="left", fontsize=8, color=mp.INK, pad=4)
        ax.tick_params(axis="y", length=0)
        st = step
        while (xlim[1] - xlim[0]) / st > (3.2 if len(cols) >= 5 else 5):
            st *= 2
        ax.xaxis.set_major_locator(MultipleLocator(st))
        ax.xaxis.set_minor_locator(MultipleLocator(st / 2))
        ax.grid(axis="x", which="minor", color=mp.GRID, lw=0.5)
        ax.tick_params(axis="x", which="minor", length=2)
        ax.tick_params(axis="x", which="major", labelsize=6.8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else f"{v:+g}".replace("-", "−")))
    axes[0].set_xlim(*xlim)
    axes[0].set_yticks(range(ny))
    axes[0].set_yticklabels([ROW_LABEL.get(r, r).replace("+", " + ") for r in rows], fontsize=7.5, color=mp.INK)
    axes[0].set_ylim(ny - 0.5, -0.5)
    axes[len(axes) // 2].set_xlabel(xlabel, fontsize=8, color=mp.INK, labelpad=5)
    if style == "bars":
        h = [plt.Rectangle((0, 0), 1, 1, color=CTX_COL[c]) for c in ctxs]
    else:
        h = [plt.Line2D([], [], marker="o", ls="", ms=5, color=CTX_COL[c]) for c in ctxs]
    top = axes[0].get_position().y1
    fig.legend(h, [CTX_SHORT[c] for c in ctxs], loc="lower center", ncol=len(ctxs), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, top + 0.22 / (0.36 * ny + 1.3)))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    status, summaries = [], {}
    for ctx in ql.CONTEXTS:
        data = ql.load_context(ctx)
        st = ql.frame_status(ctx)
        if st.empty:
            print(f"[{ctx}] no QTAIM frames yet; skipping")
            continue
        allsys = data or ql.load_all(ctx)          # paired-frame counts do not need the control
        paired = {(cs.SYSTEMS[t][1], int(r)): n for t, d in allsys.items()
                  for r, n in d["groups"][["replica", "frame"]].drop_duplicates().groupby("replica").size().items()}
        st["frames_paired"] = [paired.get((a, b), 0) for a, b in zip(st.system, st.replica)]
        status.append(st)
        if not data:
            print(f"[{ctx}] control {ql.CTRL[ctx]} has no QTAIM data yet; skipping figures")
            continue
        q = ql.compare(data, "q", ql.GROUPS)
        h = ql.compare(data, "rho", ql.WC[ctx])
        # H-bond coverage: fraction of a system's QC-passing frames in which that BCP exists
        nfr = {cs.SYSTEMS[t][1]: len(d["groups"][["replica", "frame"]].drop_duplicates()) for t, d in data.items()}
        h["frames_total"] = h.system.map(nfr)
        h["coverage"] = h.n / h.frames_total
        s = pd.concat([q, h], ignore_index=True)
        s.to_csv(ql.TAB / f"summary_{ctx}.csv", index=False, float_format="%.6g")
        summaries[ctx] = s
    # second pass: one x range per figure type, shared by all contexts
    qs = [ql.qname("q", k) for k in ql.GROUPS]
    def common(fn):
        sp = [span(s, fn(ctx), sc) for ctx, s in summaries.items() for sc in [1.0 if fn is QS else 1e3]]
        sp = [x for x in sp if np.isfinite(x[0])]
        return (min(a for a, _ in sp), max(b for _, b in sp)) if sp else None
    QS = lambda ctx: qs
    HS = lambda ctx: [ql.qname("rho", k) for k in ql.WC[ctx]]
    lq, lh = common(QS), common(HS)
    for ctx, s in summaries.items():
        qs = [ql.qname("q", k) for k in ql.GROUPS]
        ok = forest(s, qs, ["res. 6 linkage", "res. 6 sugar", "res. 6 base", "res. 6 nucleotide", "partner nucleotide"],
                    "Δq vs control / e  (negative = more electrons)", 1.0, ql.FIG / f"fig1_dq_{ctx}",
                    f"{CTX_LONG[ctx]}: change in QTAIM group charge caused by the residue-6 modification", NOTE, 0.02, lq, style=STYLE["dq"])
        hs = [ql.qname("rho", k) for k in ql.WC[ctx]]
        ok2 = forest(s, hs, ql.WC[ctx], "Δρ(BCP) vs control / 10$^{-3}$ a.u.", 1e3, ql.FIG / f"fig2_drho_{ctx}",
                     f"{CTX_LONG[ctx]}: change in electron density at the Watson–Crick H-bond critical points",
                     NOTE + " Means include only frames in which the BCP exists (coverage in table).", 2, lh, style=STYLE["drho"])
        print(f"[{ctx}] systems: {sorted(set(s.system))}; figures: {'yes' if ok else 'no modified data yet'}"
              f"{'' if ok2 else ' (no H-bond figure)'}")
    if summaries:
        combined(summaries, [{c: ql.qname("q", k) for c in summaries} for k in ql.GROUPS],
                 ["res. 6 linkage", "res. 6 sugar", "res. 6 base", "res. 6 nucleotide", "partner nucleotide"],
                 "Δq vs control / e  (negative = more electrons)", 1.0, ql.FIG / "fig_dq_combined_article", 0.02,
                 STYLE["dq"])
        combined(summaries, [{c: ql.qname("rho", col[c]) for c in summaries} for _, col in HB_COLS],
                 [t for t, _ in HB_COLS], "Δρ(BCP) vs control / 10$^{-3}$ a.u.", 1e3,
                 ql.FIG / "fig_drho_combined_article", 2, STYLE["drho"])
    st = pd.concat(status, ignore_index=True)
    st.to_csv(ql.TAB / "frame_status.csv", index=False)
    print(st.groupby("context")[["frames_planned", "frames_finished", "frames_used", "frames_excluded",
                                 "frames_paired"]].sum())


if __name__ == "__main__":
    main()
