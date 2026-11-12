#!/usr/bin/env python3
"""Run lib/central_bp/collect_qtaim.py unchanged, but decode .sum files with errors='replace'.

AIMAll can write raw bytes into the .sum of a grossly failed frame (seen: set1/sys07/seq2/1
frame_00851, sum q = -48). The original collector reads the file as strict UTF-8 before its QC
check and crashes; with replacement characters the frame parses, fails QC and is skipped.
Same arguments as collect_qtaim.py. Proposed upstream fix: patches/collect_qtaim_decode.patch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "central_bp"))
import collect_qtaim as cq  # noqa: E402



class _TolerantPath(type(Path())):
    """Path whose read_text() replaces undecodable bytes (used only inside collect_qtaim)."""
    def read_text(self, encoding=None, errors=None):
        return super().read_text(encoding=encoding, errors=errors or "replace")


cq.Path = _TolerantPath
cq.main()
