"""CPS under system-prompt steering (Mute/Amplify) vs. no steering, bootstrapped.

Feeds the steering figure's system-prompt-baseline panel in evaluate/ALL_FIGURES.ipynb.

Requires a response CSV/pickle from the system-prompt-steering experiment (columns:
scenario_id, variation, var_type, steering_type, decision); this raw experiment data is not
included in the repo.
"""

import argparse

import numpy as np
import pandas as pd

from src.config import PATH_ANALYSIS
from src.metrics import marginal_action_likelihood
from evaluate.bootstrap_utils import bootstrap_ci

VAR_TYPES = ["Consequentialist", "Emotional", "Relational"]
STEERING_TYPES = ["Mute", "Amplify"]


def _load_and_prepare(path: str) -> pd.DataFrame:
    return pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)


def compute_base_perspective(df_base: pd.DataFrame, df_var: pd.DataFrame) -> pd.DataFrame:
    """Base scenario ± a variation's system prompt ('Base ± var')."""
    results = []
    for scenario_id in df_base["scenario_id"].unique():
        scenario_df = df_base[df_base["scenario_id"] == scenario_id]
        subset_base = scenario_df[scenario_df["var_type"].isna()].copy()
        p_action2_base = marginal_action_likelihood(
            subset_base, action="action2", scenario_id=scenario_id, decision_column="decision"
        )
        for var_type in VAR_TYPES:
            for steering_type in STEERING_TYPES:
                subset_var = scenario_df[
                    (scenario_df["var_type"] == var_type)
                    & (scenario_df["steering_type"] == steering_type)
                ].copy()
                p_action2_var = marginal_action_likelihood(
                    subset_var, action="action2", scenario_id=scenario_id, decision_column="decision"
                )
                results.append(
                    {
                        "scenario_id": scenario_id,
                        "variation": "Base",
                        "var_type": var_type,
                        "steering_type": steering_type,
                        "p_action2_base": p_action2_base,
                        "p_action2_variation": p_action2_var,
                        "CPS": p_action2_var - p_action2_base,
                    }
                )
            subset_var_no_steering = df_var[
                (df_var["scenario_id"] == scenario_id)
                & (df_var["variation"] == var_type)
                & (df_var["steering_type"].isna())
            ].copy()
            p_action2_var_no_steering = marginal_action_likelihood(
                subset_var_no_steering,
                action="action2",
                scenario_id=scenario_id,
                decision_column="decision",
            )
            results.append(
                {
                    "scenario_id": scenario_id,
                    "variation": "Base",
                    "var_type": var_type,
                    "steering_type": "No Steering",
                    "p_action2_base": p_action2_base,
                    "p_action2_variation": p_action2_var_no_steering,
                    "CPS": p_action2_var_no_steering - p_action2_base,
                }
            )
    return pd.DataFrame(results)


def compute_variation_perspective(df_base: pd.DataFrame, df_var: pd.DataFrame) -> pd.DataFrame:
    """Variation scenario ± its own system prompt ('Var ± Prompt')."""
    results = []
    for scenario_id in df_var["scenario_id"].unique():
        scenario_df = df_var[df_var["scenario_id"] == scenario_id]
        subset_base = df_base[
            (df_base["scenario_id"] == scenario_id) & (df_base["steering_type"].isna())
        ].copy()
        p_action2_base = marginal_action_likelihood(
            subset_base, action="action2", scenario_id=scenario_id, decision_column="decision"
        )
        for var_type in VAR_TYPES:
            for steering_type in STEERING_TYPES:
                subset_var = scenario_df[
                    (scenario_df["var_type"] == var_type)
                    & (scenario_df["steering_type"] == steering_type)
                ].copy()
                p_action2_var = marginal_action_likelihood(
                    subset_var, action="action2", scenario_id=scenario_id, decision_column="decision"
                )
                results.append(
                    {
                        "scenario_id": scenario_id,
                        "variation": var_type,
                        "var_type": var_type,
                        "steering_type": steering_type,
                        "p_action2_base": p_action2_base,
                        "p_action2_variation": p_action2_var,
                        "CPS": p_action2_var - p_action2_base,
                    }
                )
            subset_var_no_steering = df_var[
                (df_var["scenario_id"] == scenario_id)
                & (df_var["variation"] == var_type)
                & (df_var["steering_type"].isna())
            ].copy()
            p_action2_var_no_steering = marginal_action_likelihood(
                subset_var_no_steering,
                action="action2",
                scenario_id=scenario_id,
                decision_column="decision",
            )
            results.append(
                {
                    "scenario_id": scenario_id,
                    "variation": var_type,
                    "var_type": var_type,
                    "steering_type": "No Steering",
                    "p_action2_base": p_action2_base,
                    "p_action2_variation": p_action2_var_no_steering,
                    "CPS": p_action2_var_no_steering - p_action2_base,
                }
            )
    return pd.DataFrame(results)


def bootstrap_cps(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in results_df["var_type"].unique():
        for steering_type in results_df["steering_type"].unique():
            sub_df = results_df.loc[
                (results_df["steering_type"] == steering_type) & (results_df["var_type"] == var)
            ]
            ci_lower, ci_upper, ci_mean, ci_std = bootstrap_ci(
                sub_df["CPS"].to_numpy(), stat_fn=np.mean, B=10000, ci=(2.5, 97.5)
            )
            rows.append(
                {
                    "variation": var,
                    "steering_type": steering_type,
                    "CPS_mean": ci_mean,
                    "CPS_lower": ci_lower,
                    "CPS_upper": ci_upper,
                    "CPS_std": ci_std,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute system-prompt-steering CPS")
    parser.add_argument(
        "--responses",
        default=str(PATH_ANALYSIS / "system_prompt/meta_llama-3.1-8b-instruct.csv"),
        type=str,
    )
    parser.add_argument("--output-dir", default=str(PATH_ANALYSIS), type=str)
    args = parser.parse_args()

    df = _load_and_prepare(args.responses)
    df_base = df[df["variation"] == "Base"].copy()
    df_var = df[df["variation"] != "Base"].copy()

    base_perspective = compute_base_perspective(df_base, df_var)
    base_perspective.to_csv(f"{args.output_dir}/system_prompt_base_perspective.csv", index=False)

    var_perspective = compute_variation_perspective(df_base, df_var)
    var_perspective.to_csv(f"{args.output_dir}/system_prompt_variation_perspective.csv", index=False)

    var_cps_bootstrap_df = bootstrap_cps(var_perspective)
    var_cps_bootstrap_df.to_csv(f"{args.output_dir}/system_prompt_cps_bootstrap.csv", index=False)
    print(var_cps_bootstrap_df.round(3))
    print(f"Saved to {args.output_dir}/system_prompt_cps_bootstrap.csv")
