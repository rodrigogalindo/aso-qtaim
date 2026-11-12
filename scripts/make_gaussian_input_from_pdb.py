#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import re

PDB_ATOM_RE = re.compile(r"^(ATOM  |HETATM)")

def guess_element(atom_name: str) -> str:
    s = atom_name.strip().replace("'", "")
    for ch in s:
        if ch.isalpha():
            first = ch.upper()
            break
    else:
        return "X"
    if s.upper().startswith(("CL", "BR", "NA", "MG", "ZN", "FE", "CA", "LI", "AL", "SI")):
        return s[:2].capitalize()
    return first

def read_pdb_atoms(pdb_path: Path):
    atoms = []
    with pdb_path.open() as fh:
        for line in fh:
            if not PDB_ATOM_RE.match(line):
                continue
            atom_name = line[12:16].strip()
            resname   = line[17:20].strip()
            chain     = line[21].strip() or "?"
            resid     = line[22:26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            elem = line[76:78].strip()
            if not elem:
                elem = guess_element(atom_name)

            atoms.append({
                "atom_name": atom_name,
                "resname": resname,
                "chain": chain,
                "resid": resid,
                "x": x, "y": y, "z": z,
                "elem": elem
            })
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {pdb_path}")
    return atoms

def write_atommap_csv(atoms, out_csv: Path):
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gauss_index", "element", "chain", "resid", "resname", "atom_name", "x", "y", "z"])
        for i, a in enumerate(atoms, start=1):
            w.writerow([i, a["elem"], a["chain"], a["resid"], a["resname"], a["atom_name"], a["x"], a["y"], a["z"]])

def gaussian_route(method: str, basis: str, solv: str, smd_solvent: str, wfx: bool):
    base = f"#p {method}/{basis} scf=tight nosymm pop=(nbo)"
    if solv.lower() == "smd":
        base += f" scrf=(smd,solvent={smd_solvent})"
    elif solv.lower() != "gas":
        raise ValueError("solv must be 'gas' or 'smd'")

    # AIMAll wants Extended AIM wavefunction (WFX).
    # Gaussian writes a .wfx when you include OUTPUT=WFX (or OUTPUT=WFNX) AND provide filename after geometry.
    if wfx:
        base += " output=wfx"

    return base

def write_gaussian_com(
    atoms,
    out_com: Path,
    jobname: str,
    method: str,
    basis: str,
    charge: int,
    mult: int,
    mem: str,
    nproc: int,
    solv: str,
    smd_solvent: str,
    wfx: bool,
    wfx_filename: str | None,
):
    route = gaussian_route(method, basis, solv, smd_solvent, wfx)

    if wfx:
        # Default to "<jobname>.wfx" unless overridden
        wf_name = wfx_filename.strip() if (wfx_filename and wfx_filename.strip()) else f"{jobname}.wfx"
        # If user passed something without .wfx, add it (nice UX)
        if "." not in Path(wf_name).name:
            wf_name += ".wfx"
    else:
        wf_name = None

    with out_com.open("w") as fh:
        fh.write(f"%Mem={mem}\n")
        fh.write(f"%NProcShared={nproc}\n")
        fh.write(f"{route}\n\n")
        fh.write(f"{jobname}\n\n")
        fh.write(f"{charge} {mult}\n")
        for a in atoms:
            fh.write(f"{a['elem']:2s}  {a['x']: .6f}  {a['y']: .6f}  {a['z']: .6f}\n")
        fh.write("\n")

        # If OUTPUT=WFX was requested, Gaussian expects a filename line here.
        if wf_name:
            fh.write(f"{wf_name}\n\n")

def main():
    ap = argparse.ArgumentParser(description="Generate Gaussian NBO input + atom mapping from a PDB.")
    ap.add_argument("--pdb", required=True, help="Input PDB (use the *edited.pdb from tleap/cpptraj).")
    ap.add_argument("--outdir", default=".", help="Output directory.")
    ap.add_argument("--jobname", default="QM_NBO", help="Gaussian title line / filename stem.")
    ap.add_argument("--method", default="wb97xd")
    ap.add_argument("--basis", default="def2SVP")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--mem", default="120GB")
    ap.add_argument("--nproc", type=int, default=48)
    ap.add_argument("--solv", choices=["gas", "smd"], default="gas")
    ap.add_argument("--smd-solvent", default="water")

    # Wavefunction export for AIMAll (WFX)
    ap.add_argument("--wfx", action="store_true",
                    help="Add output=wfx and write a WFX filename line (<jobname>.wfx by default).")
    ap.add_argument("--wfx-name", default="",
                    help="WFX filename (default: <jobname>.wfx).")

    # Backward-compatible aliases (your old flags)
    ap.add_argument("--wfnx", action="store_true",
                    help="ALIAS of --wfx (AIMAll needs WFX).")
    ap.add_argument("--wfnx-name", default="",
                    help="ALIAS of --wfx-name.")

    args = ap.parse_args()

    pdb_path = Path(args.pdb).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    atoms = read_pdb_atoms(pdb_path)

    stem = args.jobname
    out_com = outdir / f"{stem}.com"
    out_map = outdir / f"{stem}.atommap.csv"

    # Resolve wavefunction options
    wfx_enabled = bool(args.wfx or args.wfnx)
    wfx_name = args.wfx_name.strip() if args.wfx_name.strip() else args.wfnx_name.strip()

    write_atommap_csv(atoms, out_map)
    write_gaussian_com(
        atoms=atoms,
        out_com=out_com,
        jobname=args.jobname,
        method=args.method,
        basis=args.basis,
        charge=args.charge,
        mult=args.mult,
        mem=args.mem,
        nproc=args.nproc,
        solv=args.solv,
        smd_solvent=args.smd_solvent,
        wfx=wfx_enabled,
        wfx_filename=wfx_name,
    )

    print(f"[OK] Wrote: {out_com}")
    print(f"[OK] Wrote: {out_map}")
    if wfx_enabled:
        wf_name = wfx_name if wfx_name else f"{args.jobname}.wfx"
        if "." not in Path(wf_name).name:
            wf_name += ".wfx"
        print(f"[OK] Wavefunction export enabled: output=wfx -> {wf_name}")
    print("[INFO] Atom order matches PDB order; gauss_index in atommap.csv is the Gaussian input index.")

if __name__ == "__main__":
    main()
