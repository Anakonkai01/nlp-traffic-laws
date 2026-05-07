"""Filter QA rows whose question/gold-context embedding similarity is too low.

Run from nlp/ after scripts/diag_oracle_ceiling.py:
    python scripts/filter_qa_by_sim.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_sim_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summarize(name: str, sims: list[float]) -> dict:
    if not sims:
        return {"split": name, "n": 0, "mean": None, "p25": None, "kept_rate": 0.0}
    return {
        "split": name,
        "n": len(sims),
        "mean": round(sum(sims) / len(sims), 4),
        "p25": round(_percentile(sims, 0.25), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--sim", default="reports/traffic/oracle_sim.csv")
    parser.add_argument("--out", default="data/qa_pairs_traffic_filtered.jsonl")
    parser.add_argument("--report", default="reports/traffic/qa_sim_filter_report.json")
    parser.add_argument("--min-threshold", type=float, default=0.45)
    args = parser.parse_args()

    splits_dir = ROOT / args.splits
    sim_path = ROOT / args.sim
    out_path = ROOT / args.out
    report_path = ROOT / args.report

    split_rows = {
        "train": _load_jsonl(splits_dir / "qa_train.jsonl"),
        "dev": _load_jsonl(splits_dir / "qa_dev.jsonl"),
        "test": _load_jsonl(splits_dir / "qa_test.jsonl"),
    }
    sim_rows = _read_sim_rows(sim_path)
    n_split_rows = sum(len(rows) for rows in split_rows.values())
    if len(sim_rows) != n_split_rows:
        raise ValueError(f"oracle_sim rows ({len(sim_rows)}) != split rows ({n_split_rows}); regenerate splits/sim first")

    train_sims = [float(row["sim"]) for row in sim_rows if row["set"] == "train"]
    threshold = max(args.min_threshold, _percentile(train_sims, 0.25))

    kept_rows: list[dict] = []
    kept_sims_by_split: dict[str, list[float]] = {"train": [], "dev": [], "test": []}
    all_sims_by_split: dict[str, list[float]] = {"train": [], "dev": [], "test": []}
    kept_by_split = {"train": 0, "dev": 0, "test": 0}
    total_by_split = {"train": 0, "dev": 0, "test": 0}

    for sim_row in sim_rows:
        split = sim_row["set"]
        row_idx = int(sim_row["row_idx"])
        qa_row = split_rows[split][row_idx]
        sim = float(sim_row["sim"])
        total_by_split[split] += 1
        all_sims_by_split[split].append(sim)
        if sim >= threshold:
            kept_rows.append(qa_row)
            kept_by_split[split] += 1
            kept_sims_by_split[split].append(sim)

    _write_jsonl(out_path, kept_rows)

    report = {
        "source_splits": str(splits_dir),
        "oracle_sim": str(sim_path),
        "output": str(out_path),
        "threshold": round(threshold, 4),
        "n_before": n_split_rows,
        "n_after": len(kept_rows),
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        before = _summarize(split, all_sims_by_split[split])
        after = _summarize(split, kept_sims_by_split[split])
        report["splits"][split] = {
            "before": before,
            "after": after,
            "kept": kept_by_split[split],
            "total": total_by_split[split],
            "kept_rate": round(kept_by_split[split] / total_by_split[split], 4) if total_by_split[split] else 0.0,
            "mean_gain": round((after["mean"] or 0.0) - (before["mean"] or 0.0), 4),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] threshold={threshold:.4f}")
    print(f"[ok] kept {len(kept_rows)}/{n_split_rows} rows -> {out_path}")
    for split, info in report["splits"].items():
        print(
            f"  {split:5s}: kept={info['kept']:4d}/{info['total']:4d} "
            f"({info['kept_rate']:.3f}) mean_gain={info['mean_gain']:.4f}"
        )
    print(f"[ok] report -> {report_path}")


if __name__ == "__main__":
    main()
