#!/usr/bin/env python3
"""Aggregate score_<metric>.json files under outputs/eval into one table.

    python3 scripts/eval/collect.py [--out_root outputs/eval]

Writes ``<out_root>/summary.csv`` and ``<out_root>/summary.md``: one row per
(tag, protocol). ``mean`` follows the training-time definition -- the unweighted
sum of the protocol's metrics:

    mix:  aesthetic + clipscore + hpsv2 + pickscore (drawbench) + ocr (ocr set)
    ocr:  aesthetic + clipscore + ocr (all on the ocr set)

The protocol of a tag is read from its suffix (``*_mix`` / ``*_ocr``); ``base``
gets one row per protocol. Extra metrics (imagereward / hpsv3 / geneval2) are
reported as separate columns and are not part of ``mean``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROTOCOLS = {
    "mix": {"aesthetic": "drawbench", "clipscore": "drawbench", "hpsv2": "drawbench",
            "pickscore": "drawbench", "ocr": "ocr"},
    "ocr": {"aesthetic": "ocr", "clipscore": "ocr", "ocr": "ocr"},
}
EXTRAS = {"imagereward": "drawbench", "hpsv3": "hpsv3", "geneval2": "geneval2"}
COLUMNS = ["aesthetic", "clipscore", "hpsv2", "pickscore", "ocr"]


def read_mean(tag_dir: Path, prompt_set: str, metric: str):
    f = tag_dir / prompt_set / f"score_{metric}.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text())["mean"]


def protocols_for(tag: str) -> list[str]:
    if tag.endswith("_mix"):
        return ["mix"]
    if tag.endswith("_ocr"):
        return ["ocr"]
    return ["mix", "ocr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="outputs/eval")
    args = ap.parse_args()
    root = Path(args.out_root)
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist -- run scripts/eval/eval_ckpt.sh first")

    rows = []
    for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tag = tag_dir.name
        extras = {m: read_mean(tag_dir, s, m) for m, s in EXTRAS.items()}
        for proto in protocols_for(tag):
            spec = PROTOCOLS[proto]
            vals = {m: read_mean(tag_dir, s, m) for m, s in spec.items()}
            if all(v is None for v in vals.values()):
                continue
            complete = all(v is not None for v in vals.values())
            row = {"tag": tag, "protocol": proto}
            row.update({c: vals.get(c) for c in COLUMNS})
            row["mean"] = sum(vals.values()) if complete else None
            row.update(extras)
            rows.append(row)

    rows.sort(key=lambda r: (r["protocol"], r["tag"] != "base", r["tag"]))
    header = ["tag", "protocol", *COLUMNS, "mean", *EXTRAS]

    def fmt(v):
        return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    with (root / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(r.get(h)) for h in header])

    md = []
    for proto in ("mix", "ocr"):
        sub = [r for r in rows if r["protocol"] == proto]
        if not sub:
            continue
        cols = ["tag", *[c for c in COLUMNS if c in PROTOCOLS[proto]], "mean",
                *[m for m in EXTRAS if any(r.get(m) is not None for r in sub)]]
        md.append(f"### {proto} protocol\n")
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "---|" * len(cols))
        for r in sub:
            md.append("| " + " | ".join(fmt(r.get(c)) or "--" for c in cols) + " |")
        md.append("")
    (root / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"wrote {root / 'summary.csv'} and {root / 'summary.md'}")


if __name__ == "__main__":
    main()
