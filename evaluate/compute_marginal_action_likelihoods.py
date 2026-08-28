"""Build the master marginal_action_likelihoods.csv: for every (model, scenario, variation),
the marginal action-2 likelihood and marginal action entropy in the Base scenario vs. the
variation scenario, and the resulting CPS (Contextual Preference Shift = p_variation - p_base).

This is the master aggregation step almost every other evaluate/ script and
evaluate/ALL_FIGURES.ipynb build on.

Input: one CSV per model, as written by `python -m src.collect` (optionally with `--refine`,
which resolves 'refusal'/'invalid' decisions in place -- there is only ever one `decision`
column per file).
"""

import argparse
import os

import pandas as pd

from src.config import PATH_ANALYSIS, PATH_RESULTS
from src.metrics import marginal_action_entropy, marginal_action_likelihood

VARIATIONS = ["Consequentialist", "Emotional", "Relational"]


def iter_response_files(responses_dir: str):
    """Yield (model_id, path) pairs for every per-model CSV in `responses_dir`.
    `model_id` is derived from the filename (the '{company}_{model}' convention
    src.collect saves under), NOT from the file's own `model_id` column, which
    only holds the bare model name.
    """
    for name in sorted(os.listdir(responses_dir)):
        if not name.endswith(".csv"):
            continue
        yield name[: -len(".csv")], os.path.join(responses_dir, name)


def compute_marginal_action_likelihoods(responses_dir: str) -> pd.DataFrame:
    rows = []
    for model_id, path_file in iter_response_files(responses_dir):
        model_df = pd.read_csv(path_file)

        for scenario_id in model_df["scenario_id"].unique():
            scenario_df = model_df[model_df["scenario_id"] == scenario_id]
            base_df = scenario_df[scenario_df["variation"] == "Base"]
            if base_df.empty:
                continue

            p_action2_base, num_valid_base = marginal_action_likelihood(
                base_df, action="action2", scenario_id=scenario_id, return_num_valid=True
            )
            mae_base = marginal_action_entropy(base_df, scenario_id=scenario_id)

            for variation in VARIATIONS:
                var_df = scenario_df[scenario_df["variation"] == variation]
                if var_df.empty:
                    continue

                p_action2_variation, num_valid_variation = marginal_action_likelihood(
                    var_df, action="action2", scenario_id=scenario_id, return_num_valid=True
                )
                mae_variation = marginal_action_entropy(var_df, scenario_id=scenario_id)

                rows.append(
                    {
                        "model_id": model_id,
                        "variation": variation,
                        "scenario_id": scenario_id,
                        "p_action2_base": p_action2_base,
                        "p_action2_variation": p_action2_variation,
                        "mae_base": mae_base,
                        "mae_variation": mae_variation,
                        "num_valid_base": num_valid_base,
                        "num_valid_variation": num_valid_variation,
                        "CPS": p_action2_variation - p_action2_base,
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute marginal action likelihoods and CPS")
    parser.add_argument(
        "--responses-dir",
        required=True,
        type=str,
        help=f"Directory of per-model CSVs written by src.collect, e.g. {PATH_RESULTS}/<experiment>/<dataset>",
    )
    parser.add_argument(
        "--output", default=str(PATH_ANALYSIS / "marginal_action_likelihoods.csv"), type=str
    )
    args = parser.parse_args()

    df = compute_marginal_action_likelihoods(args.responses_dir)
    df.to_csv(args.output, index=False)
    print(df)
    print(f"Saved to {args.output}")
