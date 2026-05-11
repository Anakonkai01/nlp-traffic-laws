"""
Group split QA dataset by (doc_id, article_number) → train/dev/test 80/10/10.

Acceptance: no (doc_id, article) appears in >1 split. Output split_meta.json
with seed, sha256 per split, group counts for reproducibility.

Run from nlp/:  python scripts/make_splits.py --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ART_RE = re.compile(r"Điều\s+(\d+)")


def group_key(row: dict) -> tuple[str, str]:
    art = row.get("article", "") or ""
    m = ART_RE.search(art)
    return (row.get("doc_id", ""), m.group(1) if m else "")


def assign_groups(groups: list[tuple], sizes: tuple[int, int, int], rng: random.Random):
    """Greedy bin packing: sort groups by size desc, place into bin with most remaining capacity."""
    n_train, n_dev, n_test = sizes
    sorted_g = sorted(groups, key=lambda x: -x[1])
    bins = {"train": ([], n_train), "dev": ([], n_dev), "test": ([], n_test)}

    def slack(name):
        rows, cap = bins[name]
        return cap - sum(c for _, c in rows)

    placed = {"train": [], "dev": [], "test": []}
    for gk, count in sorted_g:
        slacks = [(slack(n), n) for n in ("train", "dev", "test")]
        rng.shuffle(slacks)
        slacks.sort(key=lambda x: -x[0])
        target = slacks[0][1]
        bins[target][0].append((gk, count))
        placed[target].append(gk)
    return placed


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/qa_pairs_traffic.jsonl")
    ap.add_argument("--outdir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    qa_path = Path(args.qa)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in qa_path.read_text().splitlines() if l.strip()]
    n_total = len(rows)

    grouped: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[group_key(r)].append(i)

    group_counts = [(gk, len(idxs)) for gk, idxs in grouped.items()]
    rng.shuffle(group_counts)

    n_train = int(round(args.ratios[0] * n_total))
    n_dev = int(round(args.ratios[1] * n_total))
    n_test = n_total - n_train - n_dev

    placed = assign_groups(group_counts, (n_train, n_dev, n_test), rng)

    split_rows = {"train": [], "dev": [], "test": []}
    for split_name, gks in placed.items():
        for gk in gks:
            for idx in grouped[gk]:
                split_rows[split_name].append(rows[idx])

    for split_name in ("train", "dev", "test"):
        rng.shuffle(split_rows[split_name])

    paths = {}
    for split_name, items in split_rows.items():
        out = outdir / f"qa_{split_name}.jsonl"
        with open(out, "w") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[split_name] = out

    train_groups = set(placed["train"])
    dev_groups = set(placed["dev"])
    test_groups = set(placed["test"])
    leak_train_dev = train_groups & dev_groups
    leak_train_test = train_groups & test_groups
    leak_dev_test = dev_groups & test_groups
    assert not (leak_train_dev or leak_train_test or leak_dev_test), (
        f"Group leak: train∩dev={len(leak_train_dev)} train∩test={len(leak_train_test)} dev∩test={len(leak_dev_test)}"
    )

    meta = {
        "seed": args.seed,
        "ratios": args.ratios,
        "source": str(qa_path),
        "n_total_rows": n_total,
        "n_unique_groups": len(grouped),
        "group_key": "(doc_id, article_number)",
        "splits": {
            split_name: {
                "path": str(paths[split_name]),
                "n_rows": len(split_rows[split_name]),
                "n_groups": len(placed[split_name]),
                "sha256": sha256_file(paths[split_name]),
            }
            for split_name in ("train", "dev", "test")
        },
        "leak_check": {
            "train_dev_groups": 0,
            "train_test_groups": 0,
            "dev_test_groups": 0,
        },
    }
    meta_path = outdir / "split_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[ok] split written → {outdir}")
    for split_name in ("train", "dev", "test"):
        info = meta["splits"][split_name]
        print(f"  {split_name:5s}: {info['n_rows']:5d} rows · {info['n_groups']:4d} groups · sha256={info['sha256'][:12]}")
    print(f"[ok] meta → {meta_path}")


if __name__ == "__main__":
    main()
