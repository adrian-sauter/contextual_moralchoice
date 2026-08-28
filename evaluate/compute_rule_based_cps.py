"""Rule-based CPS scores: mean CPS per (rule, variation), with a one-sample
t-test against 0 and Benjamini-Hochberg FDR correction across all tested cells.

Requires marginal_action_likelihoods.csv (see evaluate/compute_marginal_action_likelihoods.py)
and the scenario definitions CSV.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.config import PATH_ANALYSIS

MIN_SCENARIOS = 5  # cells with fewer scenarios are reported but not tested
ALPHA = 0.05


def compute_rule_based_cps(df_cps: pd.DataFrame, df_scenarios: pd.DataFrame) -> pd.DataFrame:
    df_cps = df_cps.copy()
    df_cps["rule"] = df_cps["scenario_id"].map(
        df_scenarios.drop_duplicates("scenario_id").set_index("scenario_id")["rule"]
    )

    # average CPS over models -> one independent observation per scenario
    df_scen = df_cps.groupby(["rule", "variation", "scenario_id"], as_index=False)["CPS"].mean()

    # paired mean-difference test per rule x variation (CPS is already the
    # within-scenario difference p_variation - p_base, so this is a one-sample
    # t-test of H0: mean CPS = 0)
    rows = []
    for (rule, variation), g in df_scen.groupby(["rule", "variation"]):
        x = g["CPS"].to_numpy()
        n = len(x)
        rec = {"rule": rule, "variation": variation, "n_scenarios": n, "mean_CPS": x.mean()}
        if n >= MIN_SCENARIOS:
            t, p = stats.ttest_1samp(x, popmean=0.0)
            se = x.std(ddof=1) / np.sqrt(n)
            crit = stats.t.ppf(1 - ALPHA / 2, df=n - 1)
            rec.update(
                t=t,
                p_value=p,
                ci_low=x.mean() - crit * se,
                ci_high=x.mean() + crit * se,
                cohens_d=x.mean() / x.std(ddof=1),
            )
        else:
            rec.update(t=np.nan, p_value=np.nan, ci_low=np.nan, ci_high=np.nan, cohens_d=np.nan)
        rows.append(rec)

    res = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR across all tested cells
    tested = res["p_value"].notna()
    res.loc[tested, "q_value"] = multipletests(
        res.loc[tested, "p_value"], alpha=ALPHA, method="fdr_bh"
    )[1]
    res["significant"] = res["q_value"] < ALPHA

    return res.sort_values(["variation", "rule"]).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute rule-based CPS scores")
    parser.add_argument(
        "--marginal-action-likelihoods",
        default=str(PATH_ANALYSIS / "marginal_action_likelihoods.csv"),
        type=str,
    )
    parser.add_argument(
        "--scenarios",
        default=str(PATH_ANALYSIS / "variations_moralchoice_high_ambiguity.csv"),
        type=str,
    )
    parser.add_argument(
        "--output", default=str(PATH_ANALYSIS / "rule_based_cps_scores.csv"), type=str
    )
    args = parser.parse_args()

    df_cps = pd.read_csv(args.marginal_action_likelihoods)
    df_scenarios = pd.read_csv(args.scenarios)

    res = compute_rule_based_cps(df_cps, df_scenarios)
    res.to_csv(args.output, index=False)
    print(res.round(4))
    print(f"Saved to {args.output}")
