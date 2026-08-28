"""Parse lm-eval-harness JSON result files (MMLU / HellaSwag / ETHICS) produced while
sweeping activation-steering strength (alpha) per variation, into one results CSV per
benchmark.

Feeds the benchmark-accuracy-vs-steering-alpha figure in evaluate/ALL_FIGURES.ipynb.

Requires lm-eval-harness output JSON files named like
'{BENCHMARK}_{variation}_alpha{value}.pt_<timestamp>.json' under --results-dir; these are not
included in the repo.
"""

import argparse
import json
import os
import re

import pandas as pd

from src.config import PATH_ANALYSIS

BENCHMARKS = ["MMLU", "HELLASWAG", "ETHICS"]
ETHICS_TASKS = [
    "ethics_cm",
    "ethics_deontology",
    "ethics_justice",
    "ethics_utilitarianism",
    "ethics_virtue",
]


def extract_macro_avg(results_json: dict, tasks=ETHICS_TASKS) -> float:
    scores = [results_json["results"][t]["acc,none"] for t in tasks]
    return sum(scores) / len(scores)


def parse_filename(fname: str, benchmark: str):
    # Example: ETHICS_consequentialist_alpha2.0.pt_2026-...json
    match = re.search(rf"{benchmark}_(\w+)_alpha([-0-9.]+)", fname)
    if not match:
        return None, None
    variation = match.group(1)
    alpha = float(match.group(2)[:-1])
    return variation, alpha


def compute_benchmark_results(results_dir: str, benchmark: str) -> pd.DataFrame:
    rows = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue

        variation, alpha = parse_filename(fname, benchmark=benchmark)
        if variation is None:
            continue
        if not fname.startswith(benchmark):
            continue

        with open(os.path.join(results_dir, fname), "r") as f:
            data = json.load(f)

        if benchmark == "ETHICS":
            macro_avg = extract_macro_avg(data)
        elif benchmark == "HELLASWAG":
            macro_avg = data["results"]["hellaswag"]["acc,none"]
        elif benchmark == "MMLU":
            macro_avg = data["results"]["mmlu"]["acc,none"]

        rows.append(
            {
                "benchmark": benchmark,
                "variation": variation.capitalize(),
                "alpha": alpha,
                "macro_avg": macro_avg,
            }
        )

    return pd.DataFrame(rows).sort_values(["variation", "alpha"]).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse lm-eval-harness benchmark results")
    parser.add_argument(
        "--results-dir", default=str(PATH_ANALYSIS / "benchmarks/full_eval"), type=str
    )
    parser.add_argument("--output-dir", default=str(PATH_ANALYSIS / "benchmarks/results"), type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for benchmark in BENCHMARKS:
        df = compute_benchmark_results(args.results_dir, benchmark)
        output_path = f"{args.output_dir}/{benchmark}_results.csv"
        df.to_csv(output_path, index=False)
        print(f"{benchmark}: {len(df)} rows -> {output_path}")
