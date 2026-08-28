"""Refusal/invalid response-rate statistics per model.

Feeds the refusals/invalids bar chart in evaluate/ALL_FIGURES.ipynb.
"""

import argparse

import pandas as pd

from src.config import PATH_ANALYSIS
from src.metrics import get_answer_statistics
from evaluate.compute_marginal_action_likelihoods import iter_response_files


def compute_refusal_invalid_stats(responses_dir: str) -> pd.DataFrame:
    rows = []
    for model_id, path_file in iter_response_files(responses_dir):
        df_temp = pd.read_csv(path_file)
        stats = get_answer_statistics(df_temp)
        rows.append(
            {
                "model_id": model_id,
                **{f"{action}_count": v["count"] for action, v in stats.items()},
                **{f"{action}_proportion": v["proportion"] for action, v in stats.items()},
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute refusal/invalid statistics per model")
    parser.add_argument("--responses-dir", required=True, type=str)
    parser.add_argument(
        "--output", default=str(PATH_ANALYSIS / "refusal_invalid_stats.csv"), type=str
    )
    args = parser.parse_args()

    statistics_df = compute_refusal_invalid_stats(args.responses_dir)
    statistics_df.to_csv(args.output, index=False)
    print(statistics_df)
    print(f"Saved to {args.output}")
