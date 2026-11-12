#!/usr/bin/env python3
"""
Pairing energy of the central base pair (residue 6 vs its Watson-Crick partner) from the compute side's
MM/GBSA (igb=7) and, when complete, QM supermolecule interaction energies (user, 2026-10-18), and their
relation to the QTAIM observables. No new calculation: reads analysis/data/mmgbsa_pair_igb7_frames.csv
(and analysis/data/qmint_frames.csv if present).

  ~/software/miniforge3/envs/qtaim/bin/python local_analysis/binding_energy.py

Frames: the QTAIM frames of compare_systems.load (QC-passing, Watson-Crick-paired), joined on
(system, replica, frame); checked against the compute side's in_qtaim_set flag. Statistics as for the
QTAIM tables (compare_systems.pooled_stats: SEM across replica means; SEM of a difference = hypot).
A two-residue model without stacking: a local pairing energy, not a hybridization free energy.
Writes
  tables/binding_mmgbsa_<ctx>.csv       pooled MM/GBSA components per system and differences vs control
  tables/binding_vs_qtaim.csv           per system: MM/GBSA and QTAIM differences side by side
  tables/binding_correlations.csv       correlations of the differences (between systems) and per frame
                                        (within systems, system means removed)
  figures/fig_binding_vs_rho.png|pdf    (main text)
  ../article/tables/tabS_binding.tex   (ESI)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import qtaim_local as ql  # noqa: E402
from qtaim_local import cs, mp  # noqa: E402
from make_figures import CTX_COL, CTX_ORDER, CTX_SHORT, NEUTRAL  # noqa: E402

MMGBSA = ql.ROOT / "analysis/data/mmgbsa_pair_igb7_frames.csv"
CTXMAP = {"A_v3": "A", "GC_v3": "GC", "U_v3": "U"}
EN = ["DELTA TOTAL", "VDWAALS", "EEL", "EGB", "ESURF", "DELTA G gas", "DELTA G solv"]


def frames_qtaim(d):
    """per-frame QTAIM observables of one system: sum and per-bond rho / ellipticity of the WC H-bonds,
    group charges."""
    hb = d["hb"]
    # ellipticity is not carried by mp.hbond_table: take it from the collected BCP table of the same
    # replica (match on frame and the exact rho/laplacian values of the BCP)
    ell = []
    for (qm, rep), _ in d["groups"].groupby(["qm_dir", "replica"]):
        tag = str(ql.Path(qm).parent.relative_to(ql.ROOT)).replace("/", "_")
        b = pd.read_csv(mp.DATA / f"{tag}_bcps.csv", usecols=["frame", "rho", "lap", "ellip", "inter_residue"])
        b = b[b.inter_residue.astype(str) == "True"].drop(columns="inter_residue")
        b["replica"] = rep
        ell.append(b)
    n0 = len(hb)
    hb = hb.merge(pd.concat(ell).drop_duplicates(["replica", "frame", "rho", "lap"]),
                  on=["replica", "frame", "rho", "lap"], how="left")
    assert len(hb) == n0 and hb.ellip.notna().all(), "ellipticity match failed"
    g = d["groups"]
    rid = str(d["partner"])
    q = g[((g.resid == "6") & g.group.isin(["linkage", "sugar", "base", "nucleotide"])) |
          ((g.resid == rid) & (g.group == "nucleotide"))].copy()
    q["col"] = np.where(q.resid == "6", "q6 " + q.group, "q partner")
    Q = q.pivot_table(index=["replica", "frame"], columns="col", values="q").reset_index()
    return hb, Q


def per_frame(ctx, data, mm):
    rows = []
    for tag, d in data.items():
        hb, Q = frames_qtaim(d)
        wc = hb[hb.key.isin(ql.WC[ctx])]
        R = wc.pivot_table(index=["replica", "frame"], columns="key", values="rho")
        R.columns = [f"rho {c}" for c in R.columns]
        R["rho sum"] = R.sum(axis=1, min_count=len(ql.WC[ctx]))
        if "ellip" in wc:
            E = wc.pivot_table(index=["replica", "frame"], columns="key", values="ellip")
            E.columns = [f"ellip {c}" for c in E.columns]
            R = R.join(E)
        f = Q.merge(R.reset_index(), on=["replica", "frame"], how="left")
        m = mm[mm.system == tag][["replica", "frame", "in_qtaim_set"] + EN]
        f = f.merge(m, on=["replica", "frame"], how="left", indicator=True)
        lost = (f._merge != "both").sum()
        notflag = (~f.in_qtaim_set.fillna(False).astype(bool)).sum()
        extra = mm[(mm.system == tag) & mm.in_qtaim_set].shape[0] - (f._merge == "both").sum()
        if lost or notflag or extra:
            print(f"[WARN] {ctx} {cs.SYSTEMS[tag][1]}: {lost} QTAIM frames without MM/GBSA, "
                  f"{notflag} not flagged in_qtaim_set, {extra} flagged frames not in the local QTAIM set")
        _, lab, chem, st, _ = cs.SYSTEMS[tag]
        f.insert(0, "system", lab)
        f.insert(1, "chem", chem)
        f.insert(0, "context", CTXMAP[ctx])
        rows.append(f.drop(columns="_merge"))
    return pd.concat(rows, ignore_index=True)


def pooled(pf, cols):
    """compare_systems statistics per system and quantity, difference vs the control of the context."""
    out = []
    for c in cols:
        ctrl = None
        for lab, s in pf.groupby("system", sort=False):
            v = s[["replica", c]].rename(columns={c: "v"}).dropna()
            if not len(v):
                continue
            p = cs.pooled_stats(v)
            if ctrl is None:
                ctrl = p
            ref = p is ctrl
            diff = p["mean"] - ctrl["mean"]
            se = np.hypot(p["sem"], ctrl["sem"])
            out.append(dict(context=s.context.iloc[0], quantity=c, system=lab, chem=s.chem.iloc[0], n=p["n"],
                            replicas=p["nrep"], sem_kind=p["sem_kind"], mean=p["mean"], sd=p["sd"], sem=p["sem"],
                            diff=np.nan if ref else diff, sem_diff=np.nan if ref else se,
                            z=np.nan if ref or not se > 0 else diff / se))
    return pd.DataFrame(out)


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return (np.corrcoef(x[ok], y[ok])[0, 1], int(ok.sum())) if ok.sum() > 2 else (np.nan, int(ok.sum()))


def main():
    mm = pd.read_csv(MMGBSA)
    allpf, allsum = [], []
    for ctx in ("U_v3", "A_v3", "GC_v3"):
        data = ql.load_context(ctx)
        pf = per_frame(ctx, data, mm)
        qcols = [c for c in pf.columns if c.startswith(("q6", "q partner", "rho", "ellip"))]
        s = pooled(pf, EN + qcols)
        s.to_csv(ql.TAB / f"binding_mmgbsa_{ctx}.csv", index=False, float_format="%.6g")
        allpf.append(pf)
        allsum.append(s)
    pf = pd.concat(allpf, ignore_index=True)
    s = pd.concat(allsum, ignore_index=True)
    pf.to_csv(ql.TAB / "binding_per_frame.csv", index=False, float_format="%.6g")

    # wide per-system table of differences
    w = s[s["diff"].notna()].pivot_table(index=["context", "system", "chem"], columns="quantity",
                                          values="diff").reset_index()
    z = s[s["diff"].notna()].pivot_table(index=["context", "system"], columns="quantity", values="z")
    w = w.merge(z[["DELTA TOTAL"]].rename(columns={"DELTA TOTAL": "z DELTA TOTAL"}).reset_index(),
                on=["context", "system"])
    w.to_csv(ql.TAB / "binding_vs_qtaim.csv", index=False, float_format="%.6g")

    # correlations
    rows = []
    obs = [c for c in w.columns if c.startswith(("rho", "ellip", "q6", "q partner"))]
    for e in ("DELTA TOTAL", "VDWAALS", "EEL", "EGB", "DELTA G gas"):
        for o in obs:
            for subset, sel in (("all", w.chem.notna()), ("anionic", ~w.chem.isin(NEUTRAL))):
                for c in ("U", "A", "GC", "all"):
                    t = w[sel & ((w.context == c) if c != "all" else True)]
                    if o not in t or t[o].notna().sum() < 4:
                        continue
                    r, n = corr(t[o].values, t[e].values)
                    rows.append(dict(level="between systems (diff vs control)", energy=e, observable=o,
                                     linkages=subset, context=c, r=r, n=n))
    # within systems: frame values minus the system mean
    for e in ("DELTA TOTAL", "VDWAALS", "EEL", "EGB"):
        for o in [c for c in pf.columns if c.startswith(("rho", "ellip"))]:
            for c in ("U", "A", "GC"):
                t = pf[pf.context == c]
                if o not in t or t[o].notna().sum() < 10:
                    continue
                dx = t[o] - t.groupby("system")[o].transform("mean")
                dy = t[e] - t.groupby("system")[e].transform("mean")
                r, n = corr(dx.values, dy.values)
                rows.append(dict(level="within systems (frames)", energy=e, observable=o, linkages="all",
                                 context=c, r=r, n=n))
    cr = pd.DataFrame(rows)
    cr.to_csv(ql.TAB / "binding_correlations.csv", index=False, float_format="%.4g")
    figure(pf, w)
    latex(s)
    show(s, cr)


def latex(s):
    """../article/tables/tabS_binding.tex (ESI table; caption in article/si.tex): MM/GBSA pairing energy, contexts side by
    side; control: mean +- SEM; other systems: difference vs control (|z| >= 3 in bold)."""
    t = s[s.quantity == "DELTA TOTAL"]
    from make_figures import ROW_OF
    lab = lambda x: ROW_OF.get(x, x)  # noqa: E731
    order = [x for x in ql.ORDER if x in set(map(lab, t.system))]
    out = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"System & U & A & G:C \\", r"\midrule"]
    for sysname in order:
        cells = []
        for c in ("U", "A", "GC"):
            r = t[(t.context == c) & (t.system.map(lab) == sysname)]
            if not len(r):
                cells.append("--")
                continue
            r = r.iloc[0]
            if np.isnan(r["diff"]):
                cells.append(f"\\({r['mean']:.2f}\\pm{r['sem']:.2f}\\)")
            else:
                v = f"\\({r['diff']:+.2f}\\)"
                cells.append((f"\\textbf{{{v}}}" if abs(r.z) >= 3 else v) + f" (\\({r.z:+.1f}\\))")
        name = sysname.replace("+", " + ").replace("Sp-PS + ", "PS + ").replace("2'", "2\\(^\\prime\\)")
        if sysname == "Sp-PS+2'-F":
            name += "\\(^\\ddagger\\)"
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (ql.ARTICLE / "tables" / "tabS_binding.tex").write_text("\n".join(out) + "\n")


def figure(pf, w):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), facecolor="white", gridspec_kw=dict(wspace=0.35))
    ax = axes[0]
    for c in CTX_ORDER:
        t = pf[pf.context == CTXMAP[c]]
        ax.scatter(t["rho sum"] * 1e3, t["DELTA TOTAL"], s=2, alpha=0.25, color=CTX_COL[c], edgecolor="none",
                   label=CTX_SHORT[c])
    ax.set_xlabel("Σρ(WC H-bond BCPs) / 10$^{-3}$ a.u.", fontsize=7.5)
    ax.set_ylabel("ΔE MM/GBSA / kcal mol$^{-1}$", fontsize=7.5)
    ax.set_title("per frame", fontsize=7.5, loc="left")
    ax = axes[1]
    for c in CTX_ORDER:
        t = w[w.context == CTXMAP[c]]
        an = ~t.chem.isin(NEUTRAL)
        ax.scatter(t["rho sum"][an] * 1e3, t["DELTA TOTAL"][an], s=14, color=CTX_COL[c], edgecolor="none")
        ax.scatter(t["rho sum"][~an] * 1e3, t["DELTA TOTAL"][~an], s=14, facecolor="white",
                   edgecolor=CTX_COL[c], lw=0.9)
    ax.axhline(0, color=mp.INK2, lw=0.5)
    ax.axvline(0, color=mp.INK2, lw=0.5)
    ax.set_xlabel("Δ Σρ vs control / 10$^{-3}$ a.u.", fontsize=7.5)
    ax.set_ylabel("Δ ΔE vs control / kcal mol$^{-1}$", fontsize=7.5)
    ax.set_title("system means (open: neutral linkages)", fontsize=7.5, loc="left")
    for ax in axes:
        mp.style(ax)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=6.5)
    h = [plt.Line2D([], [], marker="o", ls="", ms=4, color=CTX_COL[c]) for c in CTX_ORDER]
    fig.legend(h, [CTX_SHORT[c] for c in CTX_ORDER], loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(ql.FIG / f"fig_binding_vs_rho.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def show(s, cr):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    t = s[s.quantity == "DELTA TOTAL"][["context", "system", "n", "replicas", "mean", "sem", "diff", "sem_diff", "z"]]
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    k = cr[(cr.energy == "DELTA TOTAL") & (cr.observable.isin(["rho sum", "q partner", "q6 base", "q6 nucleotide"]))]
    print(k.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
