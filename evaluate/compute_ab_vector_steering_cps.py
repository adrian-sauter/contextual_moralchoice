"""CPS under activation-steering-vector interventions, swept over steering strength (alpha),
from both the variation-scenario perspective ('var ± vector') and the base-scenario
perspective ('base ± vector', i.e. can a steering vector simulate the contextual variation?).

Feeds the steering and steering-distribution figures in evaluate/ALL_FIGURES.ipynb.

Requires a response pickle from the activation-steering experiment (columns include scenario_id,
variation, var_vector, vector_type, alpha, steering_mode, question_type, decision); this raw
experiment data is not included in the repo.
"""

import argparse

import numpy as np
import pandas as pd

from src.config import PATH_ANALYSIS
from src.metrics import marginal_action_likelihood
from evaluate.bootstrap_utils import bootstrap_ci

VARIATIONS = ["Consequentialist", "Emotional", "Relational"]
VECTOR_TYPES = ["weighted", "unweighted"]
QUESTION_FORMATS = ["ab", "compare", "repeat"]


def _load_and_filter(path: str, steering_mode: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df.loc[df["steering_mode"] == steering_mode]
    df = df.loc[df["question_type"].isin(QUESTION_FORMATS)]
    return df


def compute_base_results(df: pd.DataFrame) -> pd.DataFrame:
    """Base scenario at alpha=0.0 (identical for every var_vector/vector_type, so one is enough)."""
    results = []
    subset = df[
        (df["variation"] == "Base")
        & (df["var_vector"] == "Consequentialist")
        & (df["vector_type"] == "weighted")
        & (df["alpha"] == 0.0)
    ]
    for scenario_id in subset["scenario_id"].unique():
        scenario_subset = subset[subset["scenario_id"] == scenario_id]
        results.append(
            {
                "scenario_id": scenario_id,
                "variation": "Base",
                "alpha": 0.0,
                "proportion_action2": marginal_action_likelihood(
                    scenario_subset, action="action2", scenario_id=scenario_id                ),
            }
        )
    return pd.DataFrame(results)


def compute_var_results(df: pd.DataFrame) -> pd.DataFrame:
    """Variation scenario ± steering vector, swept over alpha."""
    results = []
    for var in VARIATIONS:
        for vec_type in VECTOR_TYPES:
            subset = df[(df["variation"] == var) & (df["vector_type"] == vec_type)]
            for alpha in subset["alpha"].unique():
                alpha_subset = subset[subset["alpha"] == alpha]
                for scenario_id in subset["scenario_id"].unique():
                    scenario_subset = alpha_subset.loc[alpha_subset["scenario_id"] == scenario_id]
                    results.append(
                        {
                            "scenario_id": scenario_id,
                            "variation": var,
                            "vector_type": vec_type,
                            "alpha": alpha,
                            "proportion_action2": marginal_action_likelihood(
                                scenario_subset, action="action2", scenario_id=scenario_id                            ),
                        }
                    )
    return pd.DataFrame(results)


def compute_base_with_var_results(df: pd.DataFrame) -> pd.DataFrame:
    """Base scenario ± another variation's steering vector, swept over alpha
    (simulating the contextual variation on top of the base scenario)."""
    results = []
    for var_vector in VARIATIONS:
        for vec_type in VECTOR_TYPES:
            subset = df[
                (df["variation"] == "Base") & (df["var_vector"] == var_vector) & (df["vector_type"] == vec_type)
            ]
            for alpha in subset["alpha"].unique():
                alpha_subset = subset[subset["alpha"] == alpha]
                for scenario_id in subset["scenario_id"].unique():
                    scenario_subset = alpha_subset.loc[alpha_subset["scenario_id"] == scenario_id]
                    results.append(
                        {
                            "scenario_id": scenario_id,
                            "variation": "Base",
                            "var_vector": var_vector,
                            "vector_type": vec_type,
                            "alpha": alpha,
                            "proportion_action2": marginal_action_likelihood(
                                scenario_subset, action="action2", scenario_id=scenario_id                            ),
                        }
                    )
    return pd.DataFrame(results)


def compute_var_cps(var_df: pd.DataFrame, base_results_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for var in var_df["variation"].unique():
        for vec_type in var_df["vector_type"].unique():
            for alpha in var_df["alpha"].unique():
                subset = var_df.loc[
                    (var_df["vector_type"] == vec_type) & (var_df["variation"] == var) & (var_df["alpha"] == alpha)
                ]
                for scenario_id in subset["scenario_id"].unique():
                    p_action2_base = base_results_df.loc[
                        base_results_df["scenario_id"] == scenario_id, "proportion_action2"
                    ].values[0]
                    p_action2_var = subset.loc[subset["scenario_id"] == scenario_id, "proportion_action2"].values[0]
                    results.append(
                        {
                            "scenario_id": scenario_id,
                            "variation": var,
                            "vector_type": vec_type,
                            "alpha": alpha,
                            "proportion_action2_base": p_action2_base,
                            "proportion_action2_var": p_action2_var,
                            "CPS": p_action2_var - p_action2_base,
                        }
                    )
    return pd.DataFrame(results)


def compute_base_cps(base_with_var_df: pd.DataFrame, base_results_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for var in base_with_var_df["var_vector"].unique():
        for vec_type in base_with_var_df["vector_type"].unique():
            for alpha in base_with_var_df["alpha"].unique():
                subset = base_with_var_df.loc[
                    (base_with_var_df["vector_type"] == vec_type)
                    & (base_with_var_df["var_vector"] == var)
                    & (base_with_var_df["alpha"] == alpha)
                ]
                for scenario_id in subset["scenario_id"].unique():
                    p_action2_base = base_results_df.loc[
                        base_results_df["scenario_id"] == scenario_id, "proportion_action2"
                    ].values[0]
                    p_action2_var = subset.loc[subset["scenario_id"] == scenario_id, "proportion_action2"].values[0]
                    results.append(
                        {
                            "scenario_id": scenario_id,
                            "var_vector": var,
                            "vector_type": vec_type,
                            "alpha": alpha,
                            "proportion_action2_base": p_action2_base,
                            "proportion_action2_var": p_action2_var,
                            "CPS": p_action2_var - p_action2_base,
                        }
                    )
    return pd.DataFrame(results)


def bootstrap_var_cps(var_cps_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in var_cps_df["variation"].unique():
        for vec_type in var_cps_df["vector_type"].unique():
            for alpha in var_cps_df["alpha"].unique():
                sub_df = var_cps_df.loc[
                    (var_cps_df["vector_type"] == vec_type)
                    & (var_cps_df["variation"] == var)
                    & (var_cps_df["alpha"] == alpha)
                ]
                ci_lower, ci_upper, ci_mean, ci_std = bootstrap_ci(
                    sub_df["CPS"].to_numpy(), stat_fn=np.mean, B=10000, ci=(2.5, 97.5)
                )
                rows.append(
                    {
                        "variation": var,
                        "vector_type": vec_type,
                        "alpha": alpha,
                        "CPS_mean": ci_mean,
                        "CPS_lower": ci_lower,
                        "CPS_upper": ci_upper,
                        "CPS_std": ci_std,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_base_cps(base_cps_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in base_cps_df["var_vector"].unique():
        for vec_type in base_cps_df["vector_type"].unique():
            for alpha in base_cps_df["alpha"].unique():
                sub_df = base_cps_df.loc[
                    (base_cps_df["vector_type"] == vec_type)
                    & (base_cps_df["var_vector"] == var)
                    & (base_cps_df["alpha"] == alpha)
                ]
                ci_lower, ci_upper, ci_mean, ci_std = bootstrap_ci(
                    sub_df["CPS"].to_numpy(), stat_fn=np.mean, B=10000, ci=(2.5, 97.5)
                )
                rows.append(
                    {
                        "var_vector": var,
                        "vector_type": vec_type,
                        "alpha": alpha,
                        "CPS_mean": ci_mean,
                        "CPS_lower": ci_lower,
                        "CPS_upper": ci_upper,
                        "CPS_std": ci_std,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute activation-steering-vector CPS")
    parser.add_argument(
        "--responses",
        default=str(PATH_ANALYSIS / "robust_results/ab_vector_steering_modes/meta_llama-3.1-8b-instruct.pkl"),
        type=str,
    )
    parser.add_argument("--steering-mode", default="all_positions", type=str)
    parser.add_argument("--output-dir", default=str(PATH_ANALYSIS / "ab_vector_steering_dfs"), type=str)
    args = parser.parse_args()

    df = _load_and_filter(args.responses, args.steering_mode)

    base_results_df = compute_base_results(df)
    var_df = compute_var_results(df)
    base_with_var_df = compute_base_with_var_results(df)

    var_cps_df = compute_var_cps(var_df, base_results_df)
    var_cps_df.to_csv(f"{args.output_dir}/var.csv")

    base_cps_df = compute_base_cps(base_with_var_df, base_results_df)
    base_cps_df.to_csv(f"{args.output_dir}/base.csv")

    var_cps_bootstrap_df = bootstrap_var_cps(var_cps_df)
    var_cps_bootstrap_df.to_csv(f"{args.output_dir}/var_bootstrap.csv", index=False)

    base_cps_bootstrap_df = bootstrap_base_cps(base_cps_df)
    base_cps_bootstrap_df.to_csv(f"{args.output_dir}/base_bootstrap.csv", index=False)

    print(f"Saved var/base CPS and bootstrap CSVs to {args.output_dir}")
