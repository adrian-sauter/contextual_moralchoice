"""Per (model, variation) aggregate: mean base/variation action-2 likelihood, mean CPS,
mean marginal action entropy, flip rate, and boundary mass.

Feeds the boundary-mass-vs-flip-rate and base-MAL-vs-variation-MAL figures in
evaluate/ALL_FIGURES.ipynb.
"""

import argparse

import pandas as pd

from src.config import PATH_ANALYSIS
from src.metrics import boundary_mass, flip_rate

VARIATIONS = ["Consequentialist", "Emotional", "Relational"]


def compute_flip_boundary_mass(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df[df["variation"].isin(VARIATIONS)]
        .groupby(["model_id", "variation"], as_index=False)
        .agg(
            p_action2_base=("p_action2_base", "mean"),
            cps=("CPS", "mean"),
            mae_base=("mae_base", "mean"),
            n_scenarios=("scenario_id", "nunique"),
        )
    )

    agg["flip_rate"] = 0.0
    agg["boundary_mass"] = 0.0
    for model in agg["model_id"].unique():
        model_sub = df[df["model_id"] == model]
        for var in VARIATIONS:
            var_sub = model_sub[model_sub["variation"] == var]
            mask = (agg["model_id"] == model) & (agg["variation"] == var)
            agg.loc[mask, "flip_rate"] = flip_rate(
                var_sub["p_action2_base"], var_sub["p_action2_variation"]
            )
            agg.loc[mask, "boundary_mass"] = boundary_mass(var_sub["p_action2_base"], delta=0.1)

    agg["p_action2_variation"] = agg["p_action2_base"] + agg["cps"]
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute flip rate / boundary mass per model/variation")
    parser.add_argument(
        "--marginal-action-likelihoods",
        default=str(PATH_ANALYSIS / "marginal_action_likelihoods.csv"),
        type=str,
    )
    parser.add_argument(
        "--output", default=str(PATH_ANALYSIS / "df_with_flips_and_boundary_mass.csv"), type=str
    )
    args = parser.parse_args()

    df = pd.read_csv(args.marginal_action_likelihoods)
    agg = compute_flip_boundary_mass(df)
    agg.to_csv(args.output, index=False)
    print(agg)
    print(f"Saved to {args.output}")
