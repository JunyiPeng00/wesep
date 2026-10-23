"""Summarize eval.py predictions: SI-SDRi, SDRi, FR, STOI and PESQ.

FR (failure rate) is the percentage of target cases with SI-SDRi < 0 dB.
Usage: python summarize.py exp/<name>/eval_test/predictions.jsonl
"""

import argparse
import json
import math


def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", help="predictions.jsonl written by eval.py")
    args = parser.parse_args()

    rows = []
    with open(args.predictions, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows in {args.predictions}")

    si_sdri = [float(row["si_sdri"]) for row in rows]
    summary = {
        "num_cases": len(rows),
        "SI-SDRi": mean(si_sdri),
        "SDRi": mean([row.get("sdri") for row in rows]),
        "FR_percent": 100.0 * sum(v < 0.0 for v in si_sdri) / len(si_sdri),
        "STOI": mean([row.get("stoi") for row in rows]),
        "PESQ": mean([row.get("pesq") for row in rows]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
