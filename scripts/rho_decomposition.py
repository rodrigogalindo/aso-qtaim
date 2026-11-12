#!/usr/bin/env python3
"""
Split the change of rho at a Watson-Crick H-bond critical point (modified - control) into a geometric part,
from the change of the H...acceptor distance distribution through the pooled fit rho = a exp(-b d)
(tables/rho_distance_fits.csv), and a residual (shift of rho at fixed distance). Same frames as the QTAIM
analysis. Run after make_si_figures.py:
  ~/software/miniforge3/envs/qtaim/bin/python local_analysis/rho_decomposition.py
Writes tables/rho_decomposition.csv (all modifications, all Watson-Crick H-bonds; rho in 1e-3 a.u., d in A).
"""
import numpy as np
import pandas as pd

import qtaim_local as ql

FITS = pd.read_csv(ql.TAB / "rho_distance_fits.csv").set_index("hbond")


def hb_type(key):
    don, acc = key.split("···")
    if "C2–H" in don:
        return "C–H···O"
    return "N–H···N" if acc.split(":")[1].startswith("N") else "N–H···O"


rows = []
for ctx in ql.CONTEXTS:
    data = ql.load_context(ctx)
    ctrl = data[ql.CTRL[ctx]]["hb"]
    for tag, d in data.items():
        if tag == ql.CTRL[ctx]:
            continue
        for key in ql.WC[ctx]:
            f = FITS.loc[hb_type(key)]
            fit = lambda x: f.a * np.exp(-f.b * x)  # noqa: E731
            c, m = ctrl[ctrl.key == key], d["hb"][d["hb"].key == key]
            if m.empty:
                continue
            geom = fit(m.d_HA).mean() - fit(c.d_HA).mean()
            rows.append(dict(context=ctx, system=ql.cs.SYSTEMS[tag][1], hbond=key,
                             drho=1e3 * (m.rho.mean() - c.rho.mean()), dd_HA=m.d_HA.mean() - c.d_HA.mean(),
                             geometric=1e3 * geom,
                             residual=1e3 * ((m.rho - fit(m.d_HA)).mean() - (c.rho - fit(c.d_HA)).mean())))
out = pd.DataFrame(rows)
out.to_csv(ql.TAB / "rho_decomposition.csv", index=False, float_format="%.5g")
print(out.describe().round(3).to_string())
print("corr(drho, geometric) =", round(np.corrcoef(out.drho, out.geometric)[0, 1], 3),
      " max |residual| =", round(out.residual.abs().max(), 3))
