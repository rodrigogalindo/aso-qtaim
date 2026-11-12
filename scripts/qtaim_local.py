"""
Shared loader for the local analysis - V3 force field only (user decision 2026-10-05: V1/V2 builds are
not used anywhere). Reuses analysis/compare_systems.py (load, pooled_stats, SYSTEMS registry); the
registry is reduced at import to the V3 contexts, so no script can reach a V1/V2 system by accident.
Collected tables go to local_analysis/qtaim_data/ (the cluster's analysis/data is never touched).

Level of theory (2026-11-02): QTAIM_REV=rev00 (default; wB97X-D/def2-SVP gas, <replica>/qtaim_bp6, outputs in
local_analysis/{qtaim_data,figures,tables,structures}, article/rev00) or QTAIM_REV=rev01 (wB97X-D/def2-TZVP SMD
water, <replica>/qtaim_bp6_rev01, outputs in local_analysis/rev01/..., article/rev01). MD-only tables
(md_*, nastruct_*, ff_*) do not depend on the wavefunction and are always read from local_analysis/tables (MDTAB).
Test overrides: QTAIM_QMDIR, QTAIM_OUT, QTAIM_ARTICLE.

Statistics follow ONBOARDING_LOCAL_ANALYSIS.md section 4: QC-passing, Watson-Crick-paired frames only,
SEM across replica means when >= 2 replicas have >= 10 frames, otherwise block SEM; SEM of a
difference = sqrt(sem_mod^2 + sem_ctrl^2).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "analysis"))
import make_preliminary as mp  # noqa: E402
import compare_systems as cs  # noqa: E402
import plotfont  # noqa: E402

plotfont.apply()   # after make_preliminary, which sets its own font

REV = os.environ.get("QTAIM_REV", "rev00")
if REV not in ("rev00", "rev01"):
    sys.exit(f"QTAIM_REV must be rev00 or rev01, not {REV!r}")
QMDIR = os.environ.get("QTAIM_QMDIR", {"rev00": "qtaim_bp6", "rev01": "qtaim_bp6_rev01"}[REV])
OUT = Path(os.environ.get("QTAIM_OUT", HERE if REV == "rev00" else HERE / "rev01"))
ARTICLE = Path(os.environ.get("QTAIM_ARTICLE", ROOT / "article" / REV))
MDTAB = HERE / "tables"                      # MD-only tables, shared by rev00 and rev01
mp.FIG, mp.DATA = OUT / "qtaim_data" / "figures", OUT / "qtaim_data"


def _collect(rep):
    """mp.collect, but via collect_qtaim_safe.py (tolerates undecodable bytes in failed .sum files)."""
    import subprocess
    tag = rep.replace("/", "_")
    qm = ROOT / rep / QMDIR
    if not any(qm.glob("frame_*/frame_*.sum")):  # e.g. rev01 not computed yet for this replica: no data
        (mp.DATA / f"{tag}_groups.csv").write_text("qm_dir,frame,time_ns,resid,group,q\n")
        return tag
    out = subprocess.run([mp.PY, str(HERE / "collect_qtaim_safe.py"), str(qm),
                          "--out", str(mp.DATA / tag)], check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "QC-SKIP" in line or "WARN" in line:
            print(line)
    return tag


mp.collect = _collect
FIG, TAB, STRUCT = OUT / "figures", OUT / "tables", OUT / "structures"
for d in (mp.FIG, mp.DATA, FIG, TAB, STRUCT, ARTICLE / "tables"):
    d.mkdir(parents=True, exist_ok=True)

# ---- V3 contexts only ------------------------------------------------------------------------------
CONTEXTS = ("A_v3", "GC_v3", "U_v3")             # U_v3 (seq1) added 2026-10-09 (user)
CTRL = {"A_v3": "control02_v3", "GC_v3": "control03_v3", "U_v3": "control01_v3_rst"}   # U: restrained runs only (2026-10-09)
PARTNER = {"A_v3": 17, "GC_v3": 15, "U_v3": 17}
for _k in [k for k, v in cs.SYSTEMS.items() if v[4] not in CONTEXTS]:
    del cs.SYSTEMS[_k]

# The first V3 rebuilds live in *_std directories and are reached on the cluster through *_v3/<seq>
# symlinks, which do not survive every transfer. If a V3 entry's directory is missing locally, use the
# *_std sibling - but only if ff_consistency.py confirmed it is V3 (tables/ff_variant.csv).
_VAR = MDTAB / "ff_variant.csv"
_V3 = set(pd.read_csv(_VAR).query("ff_set == 'V3'").system) if _VAR.exists() else set()
for _k, _v in list(cs.SYSTEMS.items()):
    if not (ROOT / _v[0]).exists():
        _alt = _v[0].replace("_v3/", "_std/")
        if (ROOT / _alt).exists() and _alt in _V3:
            cs.SYSTEMS[_k] = (_alt,) + _v[1:]

ORDER = ["control (PO)", "PS2", "Rp-PS", "Sp-PS", "Rp-MsPA", "Sp-MsPA", "Rp-MP", "Sp-MP", "Rp-guan", "Sp-guan",
         "2'-F", "2'-OMe", "2'-MOE"]
ORDER += list(dict.fromkeys(v[1] for v in cs.SYSTEMS.values() if v[1] not in ORDER))  # set3 combinations
# Watson-Crick H-bonds per context (H...H contacts and bifurcations are left out)
WC = {"A_v3": ["17:N3–H···6:N1", "6:N6–H···17:O4", "6:C2–H···17:O2"],
      "GC_v3": ["6:N1–H···15:N3", "6:N2–H···15:O2", "15:N4–H···6:O6"],
      "U_v3": ["6:N3–H···17:N1", "17:N6–H···6:O4", "17:C2–H···6:O2"]}
GROUPS = [("6", "linkage"), ("6", "sugar"), ("6", "base"), ("6", "nucleotide"), ("partner", "nucleotide")]


def load_all(ctx):
    """{tag: data} for every V3 system of ctx with QTAIM data, whether or not the control has data
    (for quantities that need no control, e.g. the rho-distance relation)."""
    out = {}
    for tag, v in cs.SYSTEMS.items():
        if v[4] == ctx:
            d = cs.load(tag, v[0])
            if d is not None:
                out[tag] = d
    return out


def load_context(ctx):
    """{tag: data dict from compare_systems.load} for every system of ctx with data, control first
    ({} if the control has no QTAIM data yet)."""
    data = {}
    for tag, v in cs.SYSTEMS.items():
        if v[4] != ctx:
            continue
        d = cs.load(tag, v[0])
        if d is not None:
            data[tag] = d
    if CTRL[ctx] not in data:
        return {}
    tags = [CTRL[ctx]] + sorted((t for t in data if t != CTRL[ctx]), key=lambda t: ORDER.index(cs.SYSTEMS[t][1]))
    return {t: data[t] for t in tags}


_FFC = None


def ff_identical(tag):
    """True if ff_consistency.py found 0 differing bonded/LJ/improper terms outside residue 6 between the
    system and its V3 control (a safety check: every V3 build should pass)."""
    global _FFC
    if _FFC is None:
        f = MDTAB / "ff_consistency.csv"
        _FFC = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["system", "control", "total"])
    ctx = cs.SYSTEMS[tag][4]
    if tag == CTRL[ctx]:
        return True
    # restrained U runs (*_rst) use the same topologies as the seq1 builds (md5 checked 2026-10-13)
    r = _FFC[(_FFC.system == cs.SYSTEMS[tag][0].replace("_rst", "")) & (_FFC.control == CTRL[ctx].replace("_rst", ""))]
    return bool(len(r)) and int(r.iloc[0].total) == 0


def series(d, kind, key):
    """DataFrame(replica, v) for one quantity. kind 'q': key = (resid|'partner', group); kind 'rho': key = H-bond."""
    if kind == "q":
        rid, grp = key
        rid = str(d["partner"]) if rid == "partner" else rid
        g = d["groups"]
        return g[(g.resid == rid) & (g.group == grp)][["replica", "q"]].rename(columns={"q": "v"})
    hb = d["hb"]
    return hb[hb.key == key][["replica", "rho"]].rename(columns={"rho": "v"})


def compare(data, kind, keys):
    """Long table: one row per (quantity, system) with pooled stats and the difference from the V3
    control of the context (the control row has diff = NaN)."""
    rows = []
    for key in keys:
        ctrl = next(iter(data))
        c = cs.pooled_stats(series(data[ctrl], kind, key))
        for tag, d in data.items():
            s = series(d, kind, key)
            if len(s) == 0:
                continue
            p = cs.pooled_stats(s)
            _, lab, chem, st, ctx = cs.SYSTEMS[tag]
            ref = tag == ctrl
            diff = p["mean"] - c["mean"]
            se = np.hypot(p["sem"], c["sem"]) if not ref else np.nan
            rows.append(dict(context=ctx, quantity=qname(kind, key), system=lab, chem=chem, stereo=st,
                             n=p["n"], replicas=p["nrep"], sem_kind=p["sem_kind"], mean=p["mean"], sd=p["sd"],
                             sem=p["sem"], diff=diff if not ref else np.nan, sem_diff=se,
                             z=diff / se if np.isfinite(se) and se > 0 else np.nan,
                             ff_identical_to_control=ff_identical(tag)))
    return pd.DataFrame(rows)


def qname(kind, key):
    if kind == "q":
        rid, grp = key
        return f"q {'residue 6' if rid == '6' else 'partner'} {grp}"
    return f"rho {key}"


def frame_status(ctx):
    """Per replica: frame dirs, finished (.sum), QC-passing (used), excluded, marginal-L and NNA frames."""
    rows = []
    for tag, v in cs.SYSTEMS.items():
        if v[4] != ctx:
            continue
        for r in (1, 2, 3):
            qm = ROOT / v[0] / str(r) / QMDIR
            if not qm.is_dir():
                continue
            ndir = len(list(qm.glob("frame_*")))
            nsum = len(list(qm.glob("frame_*/frame_*.sum")))
            used = nna = marg = 0
            f = mp.DATA / f"{v[0].replace('/', '_')}_{r}_groups.csv"
            if nsum and f.exists():
                g = pd.read_csv(f, usecols=["frame", "n_nna", "max_absL"]).drop_duplicates("frame")
                used, nna = len(g), int((g.n_nna > 0).sum())
                marg = int(((g.max_absL > 1e-3) & (g.max_absL < 5e-3)).sum())
            rows.append(dict(context=ctx, system=v[1], replica=r, dir=f"{v[0]}/{r}", frames_planned=ndir,
                             frames_finished=nsum, frames_used=used, frames_excluded=nsum - used,
                             frames_marginalL=marg, frames_with_NNA=nna))
    return pd.DataFrame(rows)
