#!/usr/bin/env python3
"""
Supporting-information figures - V3 data only.

  ~/software/miniforge3/envs/qtaim/bin/python local_analysis/make_si_figures.py

  figures/figS_md_validation.png|pdf   per V3 system and replica: C3'-endo fraction of the RNA strand
                                       (residue 6 excluded), of residue 6 and of the DNA strand, and the
                                       fraction of frames with the central pair open (tables/md_pucker_strands.csv,
                                       tables/md_structure_summary.csv)
  figures/figS_rho_distance.png|pdf    rho(BCP) of the base-pair H-bonds vs H...A distance, all QC-passing
                                       paired V3 frames, fitted per donor/acceptor type (only once V3 QTAIM exists)
  tables/rho_distance_fits.csv         fit parameters rho = a exp(-b d), R^2, n
  figures/figS_nastruct.png|pdf        helical/base-pair parameters along the duplex (cpptraj nastruct, 3DNA),
                                       per V3 system (mean over replicas +- SEM), control black dashed when available
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qtaim_local as ql  # noqa: E402
from qtaim_local import cs, mp  # noqa: E402

import plotfont  # noqa: E402
plotfont.apply()
plt.rcParams.update({"font.size": 8})
LABEL = {v[0]: v[1] for v in cs.SYSTEMS.values()}   # directory -> label (V3 registry)


def label(sysd):
    if sysd.startswith("control"):
        return "control"
    return LABEL.get(sysd, LABEL.get(sysd.replace("_std/", "_v3/"), LABEL.get(sysd + "_rst", sysd)))


# contexts shown in the article (U context, seq1 rU:dA, added 2026-10-09)
ARTICLE_CTX = ("U", "A", "GC")   # seq1, seq2, seq3 (user, 2026-10-14)


def md_validation():
    d = pd.read_csv(ql.MDTAB / "md_pucker_strands.csv")
    s = pd.read_csv(ql.MDTAB / "md_structure_summary.csv")
    d = d[d.context.isin(ARTICLE_CTX)]
    if d.empty:
        return None
    per = []
    for (sysd, ctx, rep), g in d.groupby(["system", "context", "replica"]):
        o = s[(s.system == sysd) & (s.replica == rep)]
        per.append(dict(system=sysd, label=label(sysd), context=ctx, replica=rep,
                        rna=g[(g.strand == "RNA") & (g.resid != 6)].frac_north.mean(),
                        res6=g[g.resid == 6].frac_north.mean(), dna=g[g.strand == "DNA"].frac_north.mean(),
                        open6=float(o.imino_6_frac_open.iloc[0]) if len(o) else np.nan))
    per = pd.DataFrame(per)
    per["row"] = per.label + "  (" + per.context.map({"A": "A", "GC": "G:C", "U": "U"}) + ")"
    per["ctx_i"] = per.context.map({c: i for i, c in enumerate(ARTICLE_CTX)})
    rows = list(dict.fromkeys(per.sort_values(["ctx_i", "system"]).row))
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 0.32 * len(rows) + 1.2), sharey=True, facecolor="white",
                             gridspec_kw=dict(wspace=0.18))
    for ax, (col, title) in zip(axes, (("rna", "RNA strand C3′-endo\n(residue 6 excluded)"),
                                       ("res6", "residue 6 C3′-endo"), ("dna", "DNA strand C3′-endo"),
                                       ("open6", "central pair open"))):
        mp.style(ax)
        ax.set_facecolor("white")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=mp.GRID, lw=0.6)
        for i, r in enumerate(rows):
            v = per[per.row == r][col].values
            ax.scatter(v, [i] * len(v), s=16, color="#2a78d6", alpha=0.8, edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_xlim(-0.06, 1.06)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"], fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=8, color=mp.INK)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows, fontsize=7.5)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    fig.canvas.draw()
    y0 = axes[0].get_position().y0 - 0.45 / fig.get_figheight()   # 0.45 in below the panels
    fig.supxlabel("fraction of frames (0–500 ns)", fontsize=8.5, y=y0)
    for ext in ("png", "pdf"):
        fig.savefig(ql.FIG / f"figS_md_validation.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return per


def rho_distance():
    hb = []
    for ctx in ql.CONTEXTS:
        for tag, d in ql.load_all(ctx).items():   # no control needed
            h = d["hb"].copy()
            h["system"], h["context"] = tag, ctx
            hb.append(h)
    if not hb:
        print("[rho-distance] no V3 QTAIM data yet; skipped")
        return None
    hb = pd.concat(hb, ignore_index=True)
    hb["acc"] = hb.key.str.split("···").str[1].str.split(":").str[1].str[0]
    hb["don"] = hb.key.str.split(":").str[1].str[0]
    # H...A <= 3.5 A: the def2-TZVP wavefunctions (rev01) also give a few very weak, long bond paths (one at 6 A,
    # rho ~ 3e-6) that are not hydrogen bonds; rev00 had none beyond 3.5 A, so this leaves the rev00 fits unchanged
    hb = hb[(hb.rho > 0) & (hb.d_HA <= 3.5)]
    fits = []
    fig, ax = plt.subplots(figsize=(3.4, 2.8), facecolor="white")
    mp.style(ax)
    ax.set_facecolor("white")
    ax.grid(axis="x", color=mp.GRID, lw=0.6)
    for (don, acc), col, lab in ((("N", "N"), "#2a78d6", "N–H···N"), (("N", "O"), "#eb6834", "N–H···O"),
                                 (("C", "O"), "#8a8985", "C–H···O")):
        g = hb[(hb.don == don) & (hb.acc == acc)]
        if len(g) < 10:
            continue
        ax.scatter(g.d_HA, g.rho, s=3, color=col, alpha=0.25, linewidth=0, rasterized=True)
        b, lna = np.polyfit(g.d_HA, np.log(g.rho), 1)
        pred = np.exp(lna + b * g.d_HA)
        r2 = 1 - ((g.rho - pred) ** 2).sum() / ((g.rho - g.rho.mean()) ** 2).sum()
        xs = np.linspace(g.d_HA.min(), g.d_HA.max(), 50)
        ax.plot(xs, np.exp(lna + b * xs), color=col, lw=1.6, label=f"{lab}: {np.exp(lna):.2f}·e$^{{{b:.2f}d}}$, R²={r2:.3f}")
        fits.append(dict(hbond=lab, n=len(g), a=np.exp(lna), b=-b, R2=r2, d_min=g.d_HA.min(), d_max=g.d_HA.max()))
    ax.set_xlabel("H···A distance / Å", fontsize=8.5)
    ax.set_ylabel("ρ(BCP) / a.u.", fontsize=8.5)
    ax.set_yscale("log")
    ax.legend(fontsize=6.3, frameon=False, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(ql.FIG / f"figS_rho_distance.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    f = pd.DataFrame(fits)
    f.round(4).to_csv(ql.TAB / "rho_distance_fits.csv", index=False)
    return f


NA_BP = [("propeller", "propeller / °"), ("opening", "opening / °"), ("buckle", "buckle / °")]
NA_STEP = [("roll", "roll / °"), ("twist", "twist / °"), ("minor", "minor groove / Å"), ("major", "major groove / Å")]


def nastruct_profiles():
    b, st = ql.MDTAB / "nastruct_bp.csv", ql.MDTAB / "nastruct_step.csv"
    if not b.exists():
        return None
    b, st = pd.read_csv(b), pd.read_csv(st)
    chem = {v[0]: v[2] for v in cs.SYSTEMS.values()}
    ctxs = [c for c in ARTICLE_CTX if c in set(b.context)]
    fig, axes = plt.subplots(len(NA_BP) + len(NA_STEP), len(ctxs), figsize=(min(3.5 * len(ctxs), 9.0), 1.05 * 7 + 0.5),
                             sharex="col", squeeze=False, facecolor="white", gridspec_kw=dict(hspace=0.10, wspace=0.30))
    handles = {}
    for j, ctx in enumerate(ctxs):
        for i, (col, ylab) in enumerate(NA_BP + NA_STEP):
            ax = axes[i, j]
            mp.style(ax)
            ax.set_facecolor("white")
            ax.axvspan(5.5, 6.5, color="#f4d9dd", zorder=0) if i < len(NA_BP) else ax.axvspan(5, 7, color="#f4d9dd", zorder=0)
            src, xkey = (b, "pair") if i < len(NA_BP) else (st, "step_i")
            src = src[src.context == ctx]
            for sysd, g in src.groupby("system"):
                m = g.groupby(xkey)[f"{col}_mean"]
                mean, sem = m.mean(), m.std(ddof=1) / np.sqrt(m.count())
                x = mean.index.values + (0.5 if xkey == "step_i" else 0)
                ctrl = sysd.startswith("control")
                # control: black dashed (cEt's colour in the registry is a dark grey, so colour alone would not do)
                c = "#000000" if ctrl else cs.colour(chem.get(sysd, chem.get(sysd.replace("_std/", "_v3/"))))
                lab = "control" if ctrl else label(sysd)
                ln, = ax.plot(x, mean.values, color=c, lw=2.0 if ctrl else 1.4, ls="--" if ctrl else "-",
                              marker=None if ctrl else "o", ms=2.5, zorder=3 if ctrl else 2)
                ax.fill_between(x, (mean - sem).values, (mean + sem).values, color=c, alpha=0.15, linewidth=0)
                handles[lab] = ln
            ax.set_ylabel(ylab, fontsize=7)
            ax.tick_params(labelsize=6.5)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            if i == 0:
                ax.set_title({"A": "A context (A6:dT17)", "GC": "G:C context (G6:dC15)", "U": "U context (U6:dA17)"}[ctx], loc="left", fontsize=8.5)
        axes[-1, j].set_xlabel("base pair / step (RNA residue number)", fontsize=7.5)
        axes[-1, j].set_xticks(range(1, 12))
    fig.canvas.draw()
    y0 = axes[-1, 0].get_tightbbox().transformed(fig.transFigure.inverted()).y0   # below the x labels
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=min(len(handles), 6), frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, y0 - 0.005))
    for ext in ("png", "pdf"):
        fig.savefig(ql.FIG / f"figS_nastruct.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return sorted(handles)


if __name__ == "__main__":
    print("nastruct profiles:", nastruct_profiles())
    per = md_validation()
    if per is not None:
        print(per.groupby("label")[["rna", "res6", "dna", "open6"]].mean().round(3).to_string())
    f = rho_distance()
    if f is not None:
        print(f.round(4).to_string(index=False))
