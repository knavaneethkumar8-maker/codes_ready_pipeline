
import re
from pathlib import Path
from typing import Dict, List, Tuple

Interval = Tuple[float, float, str]

def read_textgrid(path: str) -> str:
    raw = Path(path).read_bytes()
    # Most Praat TextGrids in your dataset are UTF-16 BE with BOM (FE FF)
    if raw[:2] == b"\xfe\xff":
        s = raw.decode("utf-16-be", errors="ignore")
    elif raw[:2] == b"\xff\xfe":
        s = raw.decode("utf-16-le", errors="ignore")
    else:
        # fallback
        s = raw.decode("utf-8", errors="ignore")
    return s.replace("\r\n", "\n")

def parse_textgrid(path: str) -> Dict[str, List[Interval]]:
    """Parse Praat long-text TextGrid (IntervalTier only) into tier->interval list."""
    s = read_textgrid(path)

    tiers: Dict[str, List[Interval]] = {}
    # Split by tier blocks
    parts = re.split(r"\n\s*item \[\d+\]:\s*\n", s)
    for part in parts[1:]:
        mname = re.search(r'\s*name\s*=\s*"([^"]*)"', part)
        if not mname:
            continue
        name = mname.group(1)

        # Extract intervals
        intervals: List[Interval] = []
        for m in re.finditer(r"\n\s*intervals \[\d+\]:\s*\n(.*?)(?=\n\s*intervals \[\d+\]:|\Z)", part, flags=re.S):
            block = m.group(1)
            xmin = re.search(r"\s*xmin\s*=\s*([0-9\.eE+-]+)", block)
            xmax = re.search(r"\s*xmax\s*=\s*([0-9\.eE+-]+)", block)
            text = re.search(r'\s*text\s*=\s*"([^"]*)"', block)
            if xmin and xmax and text:
                intervals.append((float(xmin.group(1)), float(xmax.group(1)), text.group(1)))
        tiers[name] = intervals

    return tiers
