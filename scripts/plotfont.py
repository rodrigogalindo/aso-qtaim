"""Helvetica for every figure (user request 2026-10-08).

Real Helvetica is used when installed; otherwise Nimbus Sans, the URW metric-compatible Helvetica
clone (the font Ghostscript and TeX substitute for Helvetica). Mathtext ($...$) uses the same font.
Nimbus Sans has no superscript minus (U+207B) or U+25B8, so labels use $10^{-3}$ and the U+25BA arrow.
"""
import matplotlib

HELVETICA = ["Helvetica", "Nimbus Sans", "TeX Gyre Heros", "Liberation Sans"]


def apply():
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": HELVETICA,
        "mathtext.fontset": "custom", "mathtext.rm": HELVETICA[1], "mathtext.it": f"{HELVETICA[1]}:italic",
        "mathtext.bf": f"{HELVETICA[1]}:bold", "mathtext.sf": HELVETICA[1], "mathtext.cal": HELVETICA[1],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
