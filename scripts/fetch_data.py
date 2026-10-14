"""Download the multi-view datasets used by EFIMVC.

Sources (all reachable from this machine; ``github.com`` HTML is blocked but
``raw.githubusercontent.com`` works, and the ``ghfast.top`` prefix is accepted
as a fallback):

* ProteinFold / WebKB / 100Leaves / MNIST-10k / Caltech101-7 ... complete
  ``.mat`` files from ``ChuanbinZhang/Multi-view-datasets``.
* CCV (complete) from the ``PGCL-DCL`` archive of ``ShizheHu``.
* ORL_mtv (already incomplete, nine ``Per`` files) from ``Tracesource/FIMVC-VIA``
  -- the same first author's earlier paper, generated with the same script.

The complete datasets are stored in ``data/complete/``; :mod:`efimvc.data.protocol`
turns each of them into nine missing-rate versions reproducing
``randomlyGeneratePartialData.m``.

Usage
-----
    python scripts/fetch_data.py                 # fetch whatever is missing
    python scripts/fetch_data.py --force         # re-download everything
    python scripts/fetch_data.py --only WebKB
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COMPLETE = DATA / "complete"
INCOMPLETE = DATA / "incomplete"

RAW = "https://raw.githubusercontent.com"

# (local name under data/complete/, url, optional member inside a zip)
FILES: list[tuple[str, str]] = [
    ("ProteinFold.mat", f"{RAW}/ChuanbinZhang/Multi-view-datasets/master/ProteinFold.mat"),
    ("WebKB.mat", f"{RAW}/ChuanbinZhang/Multi-view-datasets/master/WebKB.mat"),
    ("100Leaves.mat", f"{RAW}/ChuanbinZhang/Multi-view-datasets/master/100Leaves.mat"),
]

INCOMPLETE_SETS: list[tuple[str, str]] = [
    (f"ORL_mtv_Per{p}.mat", f"{RAW}/Tracesource/FIMVC-VIA/main/datasets/ORL_mtv_Per{p}.mat")
    for p in ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")
]

ZIPS: list[tuple[str, str, list[str]]] = [
    (
        "PGCL-DCL.zip",
        f"{RAW}/ShizheHu/TNNLS25_Code_PGCL-DCL/main/PGCL-DCL.zip",
        ["CCV"],
    ),
]

MIRRORS = ["", "https://ghfast.top/"]


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for prefix in MIRRORS:
        try:
            r = subprocess.run(
                ["curl", "-sSL", "--fail", "--connect-timeout", "15", "-o", str(tmp), prefix + url],
                capture_output=True,
                timeout=1800,
            )
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1024:
                tmp.replace(dest)
                return True
        except subprocess.TimeoutExpired:
            continue
        if tmp.exists():
            tmp.unlink()
    return False


def fetch_file(local: str, url: str, force: bool, target: Path) -> None:
    dest = target / local
    if dest.exists() and not force:
        print(f"  [skip] {local:28s} {dest.stat().st_size:>12,d} B")
        return
    ok = _download(url, dest)
    size = dest.stat().st_size if dest.exists() else 0
    print(f"  [{'ok' if ok else 'FAIL'}]   {local:28s} {size:>12,d} B")
    if not ok:
        print(f"         <- {url}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on the local file name")
    ap.add_argument("--skip-big", action="store_true", help="skip the zips / incomplete packs")
    args = ap.parse_args()

    def wanted(name: str) -> bool:
        return args.only is None or args.only.lower() in name.lower()

    print("complete multi-view datasets -> data/complete/")
    for local, url in FILES:
        if wanted(local):
            fetch_file(local, url, args.force, COMPLETE)

    if not args.skip_big:
        print("pre-generated incomplete sets (same generator, third-party) -> data/incomplete/")
        for local, url in INCOMPLETE_SETS:
            if wanted(local):
                fetch_file(local, url, args.force, INCOMPLETE)

        print("archives -> data/_downloads/")
        for local, url, members in ZIPS:
            if not wanted(local):
                continue
            dest = DATA / "_downloads" / local
            fetch_file(local, url, args.force, DATA / "_downloads")
            if dest.exists():
                outdir = DATA / "complete"
                outdir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(dest) as zf:
                    for m in members:
                        hits = [n for n in zf.namelist() if m.lower() in Path(n).name.lower()]
                        for h in hits:
                            zf.extract(h, outdir / "_tmp")
                            print(f"  [zip]  {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
