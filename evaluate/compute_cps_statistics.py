"""CPS (Contextual Preference Shift) statistics per (model, variation): mean/std,
one-sided one-sample t-test and Wilcoxon signed-rank test (H1: CPS > 0), bootstrap CI,
flip rate, and positive/negative/zero shift rates.

Also computes cps_statistics_avg.csv, the pooled-across-models CPS per variation used for the
"Average" row in the CPS-scores figure.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon

from src.config import PATH_ANALYSIS
from src.metrics import boundary_mass, flip_rate
from evaluate.bootstrap_utils import bootstrap_ci

MODEL_FAMILY = {
    "Llama": ["meta_llama-2-7b-chat", "meta_llama-3-8B-instruct", "meta_llama-3.1-8B-instruct", "meta_llama-3.1-70b-instruct"],
    "Mistral": ["mistral_mixtral-8x7b-instruct_8bit", "mistral_mistral-7b-instruct-v0.1", "huggingfaceh4_zephyr-7b-beta", "teknium_openhermes-2.5-mistral-7b"],
    "Qwen": ["qwen_qwen1.5-7b-chat", "qwen_qwen2-7b-instruct", "qwen_qwen3-4b-instruct", "qwen_qwen3-8b"],
    "DeepSeek": ["deepseek_deepseek-llm-7b-chat", "deepseek-ai_DeepSeek-V3", "deepseek-ai_DeepSeek-V3.1"],
    "GPT": ["openai_gpt-4o-mini", "openai_gpt-4.1", "openai_gpt-4.1-mini", "openai_gpt-5.1"],
    "Claude": ["claude_claude-3-haiku-20240307", "claude_claude-haiku-4-5-20251001", "claude_claude-sonnet-4-5-20250929"],
}


def get_family(model_id: str) -> str:
    for family, models in MODEL_FAMILY.items():
        if model_id in models:
            return family
    return "Other"


def _compute_group_stats(sub_df: pd.DataFrame) -> dict:
    var_scores = sub_df["p_action2_variation"].to_numpy()
    base_scores = sub_df["p_action2_base"].to_numpy()
    n = len(sub_df)

    # One-sample, one-sided t-test: H0: mean(CPS) = 0, H1: mean(CPS) > 0
    t_stat, p_value = stats.ttest_1samp(sub_df["CPS"], 0, alternative="greater")
    wil_stat, wil_p = wilcoxon(sub_df["CPS"], alternative="greater")

    ci_lower, ci_upper, ci_mean, ci_std = bootstrap_ci(
        sub_df["CPS"].to_numpy(), stat_fn=np.mean, B=10000, ci=(2.5, 97.5)
    )

    mean_cps = sub_df["CPS"].mean()
    std_cps = sub_df["CPS"].std(ddof=1)

    return {
        "mean_cps": mean_cps,
        "std_cps": std_cps,
        "n": n,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohen_d": mean_cps / std_cps,
        "flip_rate": flip_rate(base_scores, var_scores),
        "positive_shift_rate": len(sub_df[sub_df["CPS"] > 0]) / n,
        "negative_shift_rate": len(sub_df[sub_df["CPS"] < 0]) / n,
        "zero_shift_rate": len(sub_df[sub_df["CPS"] == 0]) / n,
        "average_positive_shift": sub_df[sub_df["CPS"] > 0]["CPS"].mean(),
        "average_negative_shift": sub_df[sub_df["CPS"] < 0]["CPS"].mean(),
        "wilcoxon_stat": wil_stat,
        "wilcoxon_p_value": wil_p,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_mean": ci_mean,
        "ci_std": ci_std,
        "boundary_mass": boundary_mass(base_scores),
    }


def compute_cps_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variation in ["Consequentialist", "Emotional", "Relational"]:
        for model in df["model_id"].unique():
            sub_df = df[(df["model_id"] == model) & (df["variation"] == variation)].copy()
            if sub_df.empty:
                continue
            rows.append({"model_id": model, "variation": variation, **_compute_group_stats(sub_df)})

    result_df = pd.DataFrame(rows)
    result_df["family"] = result_df["model_id"].apply(get_family)
    result_df["color"] = "gray"
    return result_df.reindex(index=result_df.index[::-1])


def compute_cps_statistics_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled-across-models bootstrapped CPS mean per variation, for the figure's 'Average' row.

    The bootstrap draws n_samples equal to a single model's scenario count (not the full pooled
    row count), so pooling more models doesn't artificially narrow the CI.
    """
    rows = []
    for variation in ["Consequentialist", "Emotional", "Relational"]:
        df_var = df[df["variation"] == variation]
        n_samples = len(df_var[df_var["model_id"] == df_var["model_id"].unique()[0]])
        ci_lower, ci_upper, ci_mean, ci_std = bootstrap_ci(
            df_var["CPS"].values, np.mean, n_samples=n_samples, B=10000, ci=(2.5, 97.5)
        )
        rows.append(
            {
                "variation": variation,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_mean": ci_mean,
                "ci_std": ci_std,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CPS statistics per model/variation")
    parser.add_argument(
        "--marginal-action-likelihoods",
        default=str(PATH_ANALYSIS / "marginal_action_likelihoods.csv"),
        type=str,
    )
    parser.add_argument("--output-dir", default=str(PATH_ANALYSIS), type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.marginal_action_likelihoods)

    result_df = compute_cps_statistics(df)
    result_df.to_csv(f"{args.output_dir}/cps_statistics.csv", index=False)

    result_df_avg = compute_cps_statistics_avg(df)
    result_df_avg.to_csv(f"{args.output_dir}/cps_statistics_avg.csv", index=False)

    print(result_df[["model_id", "variation", "mean_cps", "positive_shift_rate", "negative_shift_rate"]].sort_values("mean_cps", ascending=False))
    print(f"Saved to {args.output_dir}/cps_statistics.csv and cps_statistics_avg.csv")
