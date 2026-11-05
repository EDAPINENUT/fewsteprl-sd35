#!/usr/bin/env python3
"""Score a directory of samples written by scripts/eval/sample.sh.

    python3 scripts/eval/score.py --sample_dir outputs/eval/meanflownft_mix/drawbench \
        --metrics aesthetic,clipscore,hpsv2,pickscore

    # hpsv3 needs its own environment (it pins transformers==4.45):
    python3 scripts/eval/score.py --sample_dir outputs/eval/meanflownft_mix/hpsv3 \
        --metrics hpsv3 --python /path/to/hpsv3_env/bin/python

Reads ``<sample_dir>/metadata.json`` (filename + prompt per image), scores every
image with one ``fewsteprl.rewards.MultiScorer`` per metric -- one at a time,
because holding every reward model resident at once does not fit on one GPU --
and writes, per metric:

    <sample_dir>/score_<metric>.json   {"metric", "n", "mean", "scores": [...]}

then rebuilds ``<sample_dir>/summary.json`` from every ``score_*.json`` present,
so metrics scored in different runs / interpreters merge into one summary.

Metrics: aesthetic clipscore hpsv2 pickscore ocr imagereward hpsv3 geneval2.

``ocr`` reads its target string from the double-quoted span of each prompt, so
it is only defined on the ``ocr`` prompt set; it is rejected up front on any
sample set whose prompts are not all quoted. ``geneval2`` looks up each
prompt's VQA list in ``dataset/geneval2/test.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUPPORTED = (
    "aesthetic", "clipscore", "hpsv2", "pickscore", "ocr",
    "imagereward", "hpsv3", "geneval2",
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--metrics", required=True,
                   help=f"comma-separated subset of {','.join(SUPPORTED)}")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true",
                   help="rescore metrics that already have a score_<metric>.json")
    p.add_argument("--python", default="",
                   help="re-run this script under another interpreter (e.g. the hpsv3 env)")
    return p.parse_args(argv)


def reexec_under(python: str) -> None:
    argv, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--python":
            skip = True
            continue
        if a.startswith("--python="):
            continue
        argv.append(a)
    os.execv(python, [python, str(Path(__file__).resolve()), *argv])


def setup_env() -> None:
    os.environ.setdefault("HF_HOME", str(REPO / "models" / "hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FEWSTEPRL_REWARD_CKPT_PATH", str(REPO / "models" / "reward_ckpts"))
    os.environ.setdefault("FEWSTEPRL_GENEVAL2_DATA_PATH", str(REPO / "dataset" / "geneval2" / "test.jsonl"))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def load_samples(sample_dir: Path) -> list[dict]:
    meta = sample_dir / "metadata.json"
    if not meta.is_file():
        raise FileNotFoundError(f"{meta} not found -- run scripts/eval/sample.sh first")
    rows = json.loads(meta.read_text())["images"]
    rows.sort(key=lambda r: r["index"])
    if not rows:
        raise ValueError(f"{meta} lists no images")
    return rows


def rebuild_summary(sample_dir: Path) -> dict:
    summary = {"sample_dir": str(sample_dir), "metrics": {}}
    for f in sorted(sample_dir.glob("score_*.json")):
        d = json.loads(f.read_text())
        summary["metrics"][d["metric"]] = {"mean": d["mean"], "n": d["n"]}
    (sample_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def score_metric(metric: str, rows: list[dict], sample_dir: Path, device: str,
                 batch_size: int) -> list[float]:
    import torch
    from PIL import Image
    from fewsteprl.rewards import MultiScorer

    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    scorer = MultiScorer(dev, {metric: 1.0}, allow_unavailable=False)
    values: list[float] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        images = [Image.open(sample_dir / r["filename"]).convert("RGB") for r in chunk]
        prompts = [r["prompt"] for r in chunk]
        details, _ = scorer(images, prompts)
        v = details[metric]
        values.extend(float(x) for x in (v.tolist() if hasattr(v, "tolist") else v))
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values


def main() -> None:
    args = parse_args()
    if args.python and os.path.realpath(args.python) != os.path.realpath(sys.executable):
        reexec_under(args.python)
    setup_env()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = sorted(set(metrics) - set(SUPPORTED))
    if unknown:
        sys.exit(f"ERROR: unsupported metric(s) {unknown}; supported: {list(SUPPORTED)}")

    sample_dir = Path(args.sample_dir).resolve()
    rows = load_samples(sample_dir)

    if "ocr" in metrics:
        unquoted = [r["prompt"] for r in rows if r["prompt"].count('"') < 2]
        if unquoted:
            sys.exit(f"ERROR: ocr needs a double-quoted target string in every prompt; "
                     f"{len(unquoted)} of {len(rows)} prompts in {sample_dir} have none "
                     f"(first: {unquoted[0][:80]!r}). Score ocr on the 'ocr' prompt set.")

    for metric in metrics:
        out = sample_dir / f"score_{metric}.json"
        if out.is_file() and not args.force:
            prev = json.loads(out.read_text())
            if prev.get("n") == len(rows):
                print(f"[score] skip {metric}: {out} exists (mean={prev['mean']:.4f})")
                continue
        values = score_metric(metric, rows, sample_dir, args.device, args.batch_size)
        if len(values) != len(rows):
            sys.exit(f"ERROR: {metric} returned {len(values)} scores for {len(rows)} images")
        mean = sum(values) / len(values)
        out.write_text(json.dumps({
            "metric": metric,
            "n": len(values),
            "mean": mean,
            "scores": [{"index": r["index"], "filename": r["filename"], "score": v}
                       for r, v in zip(rows, values)],
        }, indent=1))
        print(f"[score] {metric:12s} n={len(values):5d} mean={mean:.4f}  -> {out}")

    summary = rebuild_summary(sample_dir)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
