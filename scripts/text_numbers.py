#!/usr/bin/env python3
"""
Every number quoted in the article text (Results, Discussion, Conclusions, SI prose), computed from the
analysis outputs of one revision, so that the text can be updated (rev01) and the definitions checked
against the existing text (rev00).

  QTAIM_REV=rev00|rev01 ~/software/miniforge3/envs/qtaim/bin/python local_analysis/rev01/text_numbers.py
Output: <OUT>/text_numbers.md (OUT = local_analysis for rev00, local_analysis/rev01 for rev01).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LA))
import qtaim_local as ql  # noqa: E402

cs = ql.cs
TAB = ql.TAB
CTXN = {"U_v3": "U", "A_v3": "A", "GC_v3": "G:C"}
CTXS = ("U_v3", "A_v3", "GC_v3")
L = []


def out(*a):
    L.append(" ".join(str(x) for x in a))


def rng(v, n=3, scale=1.0):
    v = np.asarray([x for x in v if pd.notna(x)]) * scale
    if len(v) == 0:
        return "–"
    lo, hi = v.min(), v.max()
    return f"{lo:.{n}f}" if round(lo, n) == round(hi, n) else f"{lo:.{n}f}–{hi:.{n}f}"


S = pd.concat([pd.read_csv(TAB / f"summary_{c}.csv") for c in CTXS])
D = S[S["diff"].notna()].copy()


def dget(ctx, q, sysl):
    r = D[(D.context == ctx) & (D.quantity == q) & (D.system == sysl)]
    return (np.nan, np.nan) if r.empty else (r.iloc[0]["diff"], r.iloc[0].z)


def dtab(q, systems, n=3):
    out(f"\n**{q}** (diff, z)")
    out("| system | " + " | ".join(CTXN[c] for c in CTXS) + " |")
    out("|---|---|---|---|")
    for s in systems:
        cells = []
        for c in CTXS:
            d, z = dget(c, q, s)
            cells.append("–" if pd.isna(d) else f"{d:+.{n}f} ({z:+.1f})")
        out(f"| {s} | " + " | ".join(cells) + " |")


ANION = ["PS2", "Rp-PS", "Sp-PS", "Rp-MsPA", "Sp-MsPA"]
NEUT = ["Rp-MP", "Sp-MP", "Rp-guan", "Sp-guan"]
SUGAR = ["2'-F", "2'-OMe", "2'-MOE", "cEt"]
COMBO = ["Sp-PS+2'-F", "Rp-PS+2'-F", "Rp-PS+2'-OMe", "Rp-PS+2'-MOE", "Sp-MsPA+2'-MOE"]
COMBO_PARTS = {"Sp-PS+2'-F": ("Sp-PS", "2'-F"), "Rp-PS+2'-F": ("Rp-PS", "2'-F"),
               "Rp-PS+2'-OMe": ("Rp-PS", "2'-OMe"), "Rp-PS+2'-MOE": ("Rp-PS", "2'-MOE"),
               "Sp-MsPA+2'-MOE": ("Sp-MsPA", "2'-MOE")}
QS = ["q residue 6 linkage", "q residue 6 sugar", "q residue 6 base", "q residue 6 nucleotide", "q partner nucleotide"]

out(f"# Text numbers, {ql.REV}")

# ---- frames -----------------------------------------------------------------------------------------
fs = pd.read_csv(TAB / "frame_status.csv")
out("\n## Frames")
for c, g in fs.groupby("context"):
    out(f"- {c}: planned {g.frames_planned.sum()}, finished {g.frames_finished.sum()}, used (QC) {g.frames_used.sum()}, "
        f"QC-excluded {g.frames_excluded.sum()}, NNA frames {g.frames_with_NNA.sum()}, paired {g.frames_paired.sum()}, "
        f"open-excluded {g.frames_used.sum() - g.frames_paired.sum()}")
out(f"- total: planned {fs.frames_planned.sum()}, finished {fs.frames_finished.sum()}, used {fs.frames_used.sum()}, "
    f"QC-excluded {fs.frames_excluded.sum()} ({100 * fs.frames_excluded.sum() / fs.frames_finished.sum():.1f}% of finished), "
    f"paired (analysed) {fs.frames_paired.sum()}, open-excluded {fs.frames_used.sum() - fs.frames_paired.sum()} "
    f"({100 * (fs.frames_used.sum() - fs.frames_paired.sum()) / fs.frames_used.sum():.1f}% of used), "
    f"NNA frames {fs.frames_with_NNA.sum()}")
nna = fs[fs.frames_with_NNA > 0].groupby(["context", "system"]).agg(nna=("frames_with_NNA", "sum"), used=("frames_used", "sum"))
out("- NNA per system: " + "; ".join(f"{c} {s} {r.nna}/{r.used}" for (c, s), r in nna.iterrows()))
qc = fs[fs.frames_excluded > 0]
out("- QC-excluded per replica: " + "; ".join(f"{r.dir} {r.frames_excluded}" for _, r in qc.iterrows()))

# ---- controls ---------------------------------------------------------------------------------------
out("\n## Controls")
for c in CTXS:
    data = ql.load_context(c)
    ctrl = data[ql.CTRL[c]]
    hb, grp = ctrl["hb"], ctrl["groups"]
    nfr = grp[(grp.resid == "6") & (grp.group == "nucleotide")].shape[0]
    parts = []
    for k in ql.WC[c]:
        h = hb[hb.key == k]
        parts.append(f"{k}: CP in {len(h)}/{nfr}, mean {h.rho.mean():.4f}, sd {h.rho.std():.4f}, "
                     f"min–max {h.rho.min():.3f}–{h.rho.max():.3f}")
    out(f"- {CTXN[c]} " + "; ".join(parts))
    p = str(ctrl["partner"])
    def gs(rid, g):
        x = grp[(grp.resid == rid) & (grp.group == g)]
        rep = x.groupby("replica").q.mean()
        return x.q.mean(), x.q.std(), rep.max() - rep.min()
    vals = {n: gs(r, g) for n, (r, g) in {"nuc6": ("6", "nucleotide"), "partner": (p, "nucleotide"),
                                         "link": ("6", "linkage"), "sugar": ("6", "sugar"), "base": ("6", "base")}.items()}
    caps = grp[grp.group == "cap"].groupby(["replica", "frame"]).q.sum()   # frame numbers repeat across replicas
    cap5 = grp[(grp.group == "cap") & (grp.resid == "5")].q.mean()
    out(f"  - charges: " + ", ".join(f"{n} {m:+.3f} (sd {s:.3f}, replica-mean spread {sp:.4f})" for n, (m, s, sp) in vals.items())
        + f"; all caps + neighbour O3' {caps.mean():+.3f} (sd {caps.std():.3f}); 5'-side (O3'+cap of res 5) {cap5:+.3f}")

# ---- fits -------------------------------------------------------------------------------------------
f = pd.read_csv(TAB / "rho_distance_fits.csv")
out("\n## rho-distance fits")
for _, r in f.iterrows():
    out(f"- {r.hbond}: {r.a:.2f} exp(-{r.b:.2f} d), R2 {r.R2:.3f}, n {r.n}, d {r.d_min:.2f}–{r.d_max:.2f}")

# ---- group charge differences ------------------------------------------------------------------------
out("\n## Group-charge differences vs control")
for q in QS:
    dtab(q, ANION + NEUT + SUGAR + COMBO, 4)

# 5'-side: O3' + cap of residue 5
out("\n**5'-side (O3' + H cap of residue 5)** (diff, z)")
cap = []
for c in CTXS:
    data = ql.load_context(c)
    t = ql.compare(data, "q", [("5", "cap")])
    t["context"] = c
    cap.append(t)
cap = pd.concat(cap)
cap = cap[cap["diff"].notna()]
for grpname, systems in (("anionic", ANION), ("combined", list(COMBO)), ("neutral MP", ["Rp-MP", "Sp-MP"]),
                         ("neutral guan", ["Rp-guan", "Sp-guan"]), ("sugar", SUGAR)):
    x = cap[cap.system.isin(systems)]
    out(f"- {grpname}: diff {rng(x['diff'], 3)}, |z| min {x.z.abs().min():.1f}")
    if grpname == "anionic":
        for _, r in x.sort_values(["context", "system"]).iterrows():
            out(f"  - {CTXN[r.context]} {r.system}: {r['diff']:+.4f} ({r.z:+.1f})")

# PS2 ratio, Rp-Sp sugar
out("\n## Derived")
for c in CTXS:
    ps2 = dget(c, QS[0], "PS2")[0]
    single = np.mean([dget(c, QS[0], s)[0] for s in ("Rp-PS", "Sp-PS")])
    out(f"- {CTXN[c]} PS2 / mean(PS): {ps2 / single:.2f}; Rp − Sp sugar: PS {dget(c, QS[1], 'Rp-PS')[0] - dget(c, QS[1], 'Sp-PS')[0]:+.4f}, "
        f"MsPA {dget(c, QS[1], 'Rp-MsPA')[0] - dget(c, QS[1], 'Sp-MsPA')[0]:+.4f}, "
        f"MP {dget(c, QS[1], 'Rp-MP')[0] - dget(c, QS[1], 'Sp-MP')[0]:+.4f}, guan {dget(c, QS[1], 'Rp-guan')[0] - dget(c, QS[1], 'Sp-guan')[0]:+.4f}")
an = D[D.system.isin(ANION)]
for q in QS:
    x = an[an.quantity == q]
    out(f"- anionic {q}: diff {rng(x['diff'], 3)}, |z| {rng(x.z.abs(), 1)}, significant in "
        + ", ".join(f"{r.context}:{r.system}" for _, r in x[x.z.abs() >= 3].iterrows()))
ne = D[D.system.isin(NEUT)]
for q in QS:
    x = ne[ne.quantity == q]
    out(f"- neutral {q}: diff {rng(x['diff'], 3)}, significant in "
        + ", ".join(f"{r.context}:{r.system} {r['diff']:+.3f} ({r.z:+.1f})" for _, r in x[x.z.abs() >= 3].iterrows()))
pt = D[D.quantity == "q partner nucleotide"]
out("- partner: max |diff| " + f"{pt['diff'].abs().max():.4f}; significant: "
    + ", ".join(f"{r.context}:{r.system} {r['diff']:+.4f} ({r.z:+.1f})" for _, r in pt[pt.z.abs() >= 3].iterrows()))
gq = D[D.quantity.str.startswith("q") & ~D.system.isin(NEUT)]
out("- all significant group differences outside the neutral linkages, by system: ")
for s, x in gq[gq.z.abs() >= 3].groupby("system"):
    out(f"  - {s}: " + ", ".join(f"{r.context} {r.quantity.replace('q residue 6 ', '').replace('q ', '')} {r['diff']:+.4f} ({r.z:+.1f})" for _, r in x.iterrows()))

# additivity
out("\n## Additivity (combined − sum of parts)")
rows = []
for c in ("A_v3", "GC_v3"):
    for combo, (bb, su) in COMBO_PARTS.items():
        for q in QS:
            r = S[(S.context == c) & (S.quantity == q)].set_index("system")
            if combo not in r.index:
                continue
            dev = r.loc[combo, "diff"] - r.loc[bb, "diff"] - r.loc[su, "diff"]
            se = np.sqrt(r.loc[combo, "sem_diff"] ** 2 + r.loc[bb, "sem_diff"] ** 2 + r.loc[su, "sem_diff"] ** 2)
            rows.append(dict(ctx=c, combo=combo, q=q, combo_d=r.loc[combo, "diff"],
                             sum_d=r.loc[bb, "diff"] + r.loc[su, "diff"], dev=dev, ratio=dev / se))
ad = pd.DataFrame(rows)
out(f"- max |dev| {ad.dev.abs().max():.4f}; max |dev|/se {ad.ratio.abs().max():.2f}")
for _, r in ad[(ad.combo == "Sp-PS+2'-F") & (ad.ctx == "A_v3")].iterrows():
    out(f"  - A Sp-PS+2'-F {r.q}: combo {r.combo_d:+.4f}, sum {r.sum_d:+.4f}")

# ---- rho_b ----------------------------------------------------------------------------------------
out("\n## H-bond rho differences")
rh = D[D.quantity.str.startswith("rho")].copy()
i = rh['diff'].abs().idxmax() if False else rh['diff'].abs().values.argmax()
r = rh.iloc[i]
out(f"- n {len(rh)}; max |diff| {abs(r['diff']):.4f} ({r.context} {r.system} {r.quantity}, z {r.z:+.1f}); "
    f"significant {int((rh.z.abs() >= 3).sum())}, all negative: {bool((rh[rh.z.abs() >= 3]['diff'] < 0).all())}")
for _, r in rh[rh.z.abs() >= 3].iterrows():
    out(f"  - {r.context} {r.system} {r.quantity}: {r['diff']:+.4f} ({r.z:+.1f})")
ctrlm = S[S["diff"].isna() & S.quantity.str.startswith("rho")].set_index(["context", "quantity"])["mean"]
rh["rel"] = [abs(r['diff']) / ctrlm[(r.context, r.quantity)] for _, r in rh.iterrows()]
out(f"- max relative change {rh.rel.max() * 100:.0f}%")
for c in CTXS:
    x = rh[rh.context == c]
    out(f"- {CTXN[c]}: max |diff| {x['diff'].abs().max():.4f}, |z| max {x.z.abs().max():.1f}")
    for k in ql.WC[c]:
        y = x[x.quantity == f"rho {k}"]
        out(f"  - {k}: diff {rng(y['diff'], 4)}; all negative {bool((y['diff'] < 0).all())}")

dec = pd.read_csv(TAB / "rho_decomposition.csv")
out(f"- decomposition (1e-3 a.u.): r(drho, geometric) {np.corrcoef(dec.drho, dec.geometric)[0, 1]:.3f}, "
    f"max |residual| {dec.residual.abs().max():.2f}, n {len(dec)}")
sig = rh[rh.z.abs() >= 3]
for _, r in sig.iterrows():
    k = r.quantity.replace("rho ", "")
    y = dec[(dec.context == r.context) & (dec.system == r.system) & (dec.hbond == k)]
    if len(y):
        out(f"  - {r.context} {r.system} {k}: dd_HA {y.iloc[0].dd_HA:+.3f} Å, geometric {y.iloc[0].geometric:+.2f}, residual {y.iloc[0].residual:+.2f}")

# ---- NPA ----------------------------------------------------------------------------------------
npa = pd.read_csv(TAB / "npa_vs_qtaim.csv")
npa = npa[npa.diff_npa.notna()]
out("\n## NPA vs QTAIM")
for q, x in npa.groupby("quantity"):
    out(f"- {q}: r {np.corrcoef(x.diff_npa, x.diff_qtaim)[0, 1]:.3f}")
x = npa[(npa.quantity == "q residue 6 linkage") & npa.system.isin(ANION)]
out(f"- anionic linkage NPA/QTAIM ratio: median {np.median(x.diff_npa / x.diff_qtaim):.2f}, range {rng(x.diff_npa / x.diff_qtaim, 2)}")
for c in CTXS:
    y = x[x.context == c].set_index("system")
    out(f"  - {CTXN[c]} NPA linkage: " + ", ".join(f"{s} {y.loc[s, 'diff_npa']:+.3f}" for s in ANION if s in y.index))

# ---- MM/GBSA -----------------------------------------------------------------------------------
out("\n## MM/GBSA")
mm = pd.concat([pd.read_csv(TAB / f"binding_mmgbsa_{c}.csv") for c in CTXS])
t = mm[mm.quantity == "DELTA TOTAL"]
for c, x in t.groupby("context"):
    ct = x[x["diff"].isna()].iloc[0]
    sg = x[x.z.abs() >= 3]
    out(f"- {c}: control {ct['mean']:.2f} ± {ct['sem']:.2f}; significant: " + ", ".join(f"{r.system} {r['diff']:+.2f} ({r.z:+.1f})" for _, r in sg.iterrows()))
for qq in ("EEL", "EGB"):
    x = mm[(mm.quantity == qq) & mm.system.isin(NEUT)]
    out(f"- neutral {qq} diff {rng(x['diff'], 1)}")
bc = pd.read_csv(TAB / "binding_correlations.csv")
w = bc[(bc.level.str.startswith("within")) & (bc.observable == "rho sum")]
out("- within systems, rho sum vs: " + "; ".join(f"{r.energy} {r.context} r {r.r:.2f}" for _, r in w.iterrows()))
b = bc[(bc.level.str.startswith("between")) & (bc.observable == "rho sum") & (bc.energy == "DELTA TOTAL")]
out("- between systems, rho sum vs total: " + "; ".join(f"{r.linkages} {r.context} r {r.r:.2f}" for _, r in b.iterrows()))

# ---- atoms ---------------------------------------------------------------------------------------
out("\n## Atomic populations (dN = -dq, e)")
at = pd.concat([pd.read_csv(TAB / f"atom_dN_{c}.csv") for c in CTXS])
at = at[at.dN.notna()]


def arng(systems, slot_pred, label, n=2):
    x = at[at.system.isin(systems) & slot_pred(at)]
    out(f"- {label}: dN {rng(x.dN, n)}  (|z| min {x.z.abs().min():.1f})" if len(x) else f"- {label}: none")


sub = at.substituted == True  # noqa: E712
PS = ["Rp-PS", "Sp-PS"]
arng(PS, lambda a: a.substituted == True, "PS substituted atom (S for O)")  # noqa: E712
arng(["PS2"], lambda a: a.substituted == True, "PS2 substituted atoms")  # noqa: E712
arng(PS, lambda a: a.slot == "P", "PS P")
arng(["PS2"], lambda a: a.slot == "P", "PS2 P")
x = at[at.system.isin(PS + ["PS2"]) & (at.substituted != True) & (at.slot != "P")]  # noqa: E712
out(f"- PS/PS2 all other atoms: max |dN| {x.dN.abs().max():.3f}")
for sl in ("OP1", "OP2", "C5'", "C3'", "H5'", "H5''"):
    arng(PS + ["PS2"], lambda a, sl=sl: (a.slot == sl) & (a.substituted != True), f"PS/PS2 {sl}")  # noqa: E712
MS = ["Rp-MsPA", "Sp-MsPA"]
arng(MS, lambda a: a.substituted == True, "MsPA substituted atom(s)")  # noqa: E712
arng(MS, lambda a: a.slot == "P", "MsPA P")
for sl in ("OP1", "OP2"):
    arng(MS, lambda a, sl=sl: (a.slot == sl) & (a.substituted != True), f"MsPA {sl}")  # noqa: E712
arng(ANION, lambda a: a.slot.str.startswith("O3'(5)"), "anionic O3'(5)", 3)
arng(["Rp-MP", "Sp-MP"], lambda a: a.slot.isin(["OP1", "OP2"]) & (a.substituted != True), "MP remaining OP")  # noqa: E712
arng(["Rp-guan", "Sp-guan"], lambda a: a.slot.isin(["OP1", "OP2"]) & (a.substituted != True), "guan remaining OP")  # noqa: E712
arng(NEUT, lambda a: a.slot == "C5'", "neutral C5'")
arng(NEUT, lambda a: a.slot.isin(["H8", "H6"]) & (a.group == "base"), "neutral base H8/H6", 3)
arng(NEUT, lambda a: a.slot.str.startswith("O3'(5)"), "neutral O3'(5)", 3)
for c in CTXS:
    x = at[(at.context == c) & at.system.isin(NEUT) & (at.group == "base")]
    top = x.reindex(x.dN.abs().sort_values(ascending=False).index).head(4)
    out(f"  - {CTXN[c]} neutral, largest base atoms: " + ", ".join(f"{r.system} {r.slot} {r.dN:+.3f} ({r.z:+.1f})" for _, r in top.iterrows()))
for nm, systems in (("anionic", ANION), ("neutral", NEUT), ("sugar", SUGAR), ("all non-neutral", ANION + SUGAR + COMBO)):
    x = at[at.system.isin(systems) & (at.group == "base")]
    out(f"- {nm} base atoms: max |dN| {x.dN.abs().max():.3f}, fraction < 0.02: {(x.dN.abs() < 0.02).mean():.2f}, "
        f"n |z|>=3 {(x.z.abs() >= 3).sum()} of {len(x)}")
arng(["2'-OMe", "2'-MOE", "cEt"], lambda a: a.slot == "O2'", "2'-O of O-alkyl/cEt")
arng(["2'-F"], lambda a: a.substituted == True, "2'-F F vs O2'")  # noqa: E712
arng(["2'-F", "cEt"], lambda a: a.slot == "H2'", "H2' (2'-F, cEt)")
arng(["cEt"], lambda a: a.slot == "C4'", "cEt C4'")

dst = ql.OUT / "text_numbers.md"
dst.write_text("\n".join(L) + "\n")
print(dst)
