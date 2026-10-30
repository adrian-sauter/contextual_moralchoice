"""Export the JSON data the website renders from the contextual_moralchoice repo.

Usage:
    python scripts/build_data.py --repo ../contextual_moralchoice

Writes:
    data/site.json       all aggregate results (base MAL, CPS, scatter, steering, survey) + featured scenarios
    data/explorer.json   every scenario's texts and each model's MAL per version (lazy-loaded by the explorer)
"""

import argparse
import json
import os

import pandas as pd

VARIATIONS = ["Consequentialist", "Emotional", "Relational"]

# Featured scenarios (all have all three variations): shown in the hero, the dataset viewer and as
# quick picks in the explorer. The explorer additionally offers every scenario in a dropdown.
FEATURED = {
    "G_109": "Copying an out-of-print book",
    "G_113": "Illegal medicine",
    "G_231": "Smuggling goods",
    "G_347": "The best friend's secret",
    "G_275": "The wounded soldier",
    "G_143": "Locking a bank account",
    "H_057": "A comforting lie",
    "G_412": "Breaking curfew",
    "H_008": "The dictator",
    "G_199": "A violent friend",
    "G_269": "The pharmacy thief",
    "G_258": "A colleague's free medicine",
}

# Logos are trimmed PNG renders of the SVGs; ratio = width / height (used to size them consistently).
PROVIDERS = [
    {"id": "meta", "name": "Meta", "logo": "assets/logos/meta.png", "ratio": 5.0},
    {"id": "mistral", "name": "Mistral AI", "logo": "assets/logos/mistral.png", "ratio": 6.406},
    {"id": "qwen", "name": "Qwen", "logo": "assets/logos/qwen.png", "ratio": 3.323},
    {"id": "deepseek", "name": "DeepSeek", "logo": "assets/logos/deepseek.png", "ratio": 4.604},
    {"id": "anthropic", "name": "Anthropic", "logo": "assets/logos/anthropic.png", "ratio": 9.0},
    {"id": "openai", "name": "OpenAI", "logo": "assets/logos/openai.png", "ratio": 3.573},
]

# (model_id, short name, provider, size, plot color from evaluate/plotting_config.py) -- order follows the poster.
MODELS = [
    ("meta_llama-2-7b-chat", "Llama-2-7B", "meta", "7B", "#2557A7"),
    ("meta_llama-3-8B-instruct", "Llama-3-8B", "meta", "8B", "#4A7BC7"),
    ("meta_llama-3.1-8B-instruct", "Llama-3.1-8B", "meta", "8B", "#7BA2E0"),
    ("meta_llama-3.1-70b-instruct", "Llama-3.1-70B", "meta", "70B", "#A9C4F0"),
    ("mistral_mixtral-8x7b-instruct_8bit", "Mixtral-8x7B", "mistral", "8x7B", "#9C7B00"),
    ("mistral_mistral-7b-instruct-v0.1", "Mistral-7B", "mistral", "7B", "#B38F00"),
    ("huggingfaceh4_zephyr-7b-beta", "Zephyr-7B-β", "mistral", "7B", "#D9AD26"),
    ("teknium_openhermes-2.5-mistral-7b", "OpenHermes-2.5-7B", "mistral", "7B", "#ECC263"),
    ("qwen_qwen1.5-7b-chat", "Qwen1.5-7B", "qwen", "7B", "#6A49AD"),
    ("qwen_qwen2-7b-instruct", "Qwen2-7B", "qwen", "7B", "#8A6BC2"),
    ("qwen_qwen3-4b-instruct", "Qwen3-4B", "qwen", "4B", "#A88FD8"),
    ("qwen_qwen3-8b", "Qwen3-8B", "qwen", "8B", "#C8B6ED"),
    ("deepseek_deepseek-llm-7b-chat", "DeepSeek-7B", "deepseek", "7B", "#3E5266"),
    ("deepseek-ai_DeepSeek-V3", "DeepSeek-V3", "deepseek", "671B", "#5C7085"),
    ("deepseek-ai_DeepSeek-V3.1", "DeepSeek-V3.1", "deepseek", "671B", "#7C8FA3"),
    ("claude_claude-3-haiku-20240307", "Claude-3-Haiku", "anthropic", "n/a", "#3D6B4A"),
    ("claude_claude-haiku-4-5-20251001", "Claude-Haiku-4.5", "anthropic", "n/a", "#5E8C6B"),
    ("claude_claude-sonnet-4-5-20250929", "Claude-Sonnet-4.5", "anthropic", "n/a", "#82AE8E"),
    ("openai_gpt-4o-mini", "GPT-4o-Mini", "openai", "n/a", "#7D3535"),
    ("openai_gpt-4.1", "GPT-4.1", "openai", "n/a", "#913D3D"),
    ("openai_gpt-4.1-mini", "GPT-4.1-Mini", "openai", "n/a", "#B4655F"),
    ("openai_gpt-5.1", "GPT-5.1", "openai", "n/a", "#D29891"),
]

# Human survey results reported in the paper (raw survey data is not part of the repo).
HUMAN = {
    "n": 132,
    "scenarios": 20,
    "cps": {"Consequentialist": 0.083, "Emotional": 0.105, "Relational": 0.122},
    "correlation": {"mean": 0.169, "median": 0.194, "best": 0.608},
    "alignment": [
        {"model": "qwen_qwen3-8b", "agr": 0.45, "C": 0.243, "E": 0.372, "R": 0.390},
        {"model": "meta_llama-3.1-70b-instruct", "agr": 0.45, "C": 0.167, "E": -0.118, "R": 0.429},
        {"model": "teknium_openhermes-2.5-mistral-7b", "agr": 0.35, "C": 0.524, "E": 0.608, "R": 0.181},
        {"model": "deepseek-ai_DeepSeek-V3", "agr": 0.60, "C": 0.342, "E": 0.414, "R": 0.320},
        {"model": "claude_claude-sonnet-4-5-20250929", "agr": 0.50, "C": 0.396, "E": 0.195, "R": 0.091},
        {"model": "openai_gpt-4o-mini", "agr": 0.65, "C": 0.395, "E": 0.497, "R": 0.281},
    ],
}

def r3(x):
    return None if pd.isna(x) else round(float(x), 4)


def build_site(repo):
    mal = pd.read_csv(os.path.join(repo, "evaluate/data/marginal_action_likelihoods.csv"))
    stats = pd.read_csv(os.path.join(repo, "evaluate/data/cps_statistics.csv"))
    avg = pd.read_csv(os.path.join(repo, "evaluate/data/cps_statistics_avg.csv"))
    fb = pd.read_csv(os.path.join(repo, "evaluate/data/df_with_flips_and_boundary_mass.csv"))
    steer_dir = os.path.join(repo, "evaluate/data/ab_vector_steering_dfs")
    vb = pd.read_csv(os.path.join(steer_dir, "var_bootstrap.csv"))
    vd = pd.read_csv(os.path.join(steer_dir, "var.csv"))

    base_mal = {}
    for mid, *_ in MODELS:
        b = mal[mal.model_id == mid].drop_duplicates("scenario_id")
        base_mal[mid] = [r3(x) for x in b.p_action2_base.dropna()]

    cps = []
    for _, r in stats.iterrows():
        cps.append({
            "model": r.model_id, "variation": r.variation, "mean": r3(r.mean_cps),
            "lo": r3(r.ci_lower), "hi": r3(r.ci_upper), "n": int(r.n), "flip": r3(r.flip_rate),
        })
    cps_avg = [{"variation": r.variation, "mean": r3(r.ci_mean), "lo": r3(r.ci_lower), "hi": r3(r.ci_upper)}
               for _, r in avg.iterrows()]

    scatter = [{"model": r.model_id, "variation": r.variation, "base": r3(r.p_action2_base),
                "var": r3(r.p_action2_variation), "flip": r3(r.flip_rate), "bm": r3(r.boundary_mass)}
               for _, r in fb.iterrows()]

    w = vb[vb.vector_type == "weighted"]
    steer_curve = [{"variation": r.variation, "alpha": float(r.alpha), "mean": r3(r.CPS_mean),
                    "lo": r3(r.CPS_lower), "hi": r3(r.CPS_upper)} for _, r in w.iterrows()]
    wd = vd[vd.vector_type == "weighted"]
    steer_dist = {
        v: {str(int(a)): [r3(x) for x in g.CPS] for a, g in wd[wd.variation == v].groupby("alpha")}
        for v in VARIATIONS
    }

    return {
        "providers": PROVIDERS,
        "models": [{"id": m, "name": n, "provider": p, "size": s, "color": c} for m, n, p, s, c in MODELS],
        "baseMAL": base_mal,
        "cps": cps,
        "cpsAvg": cps_avg,
        "scatter": scatter,
        "steering": {"curve": steer_curve, "dist": steer_dist, "nTest": int(wd.scenario_id.nunique())},
        "human": HUMAN,
    }


def build_explorer(repo, out_path):
    scen = pd.read_csv(os.path.join(repo, "data/scenarios/variations_moralchoice_high_ambiguity.csv"))
    mal = pd.read_csv(os.path.join(repo, "evaluate/data/marginal_action_likelihoods.csv"))
    versions = ["Base"] + VARIATIONS

    scenarios = []
    for sid, rows in scen.groupby("scenario_id", sort=False):
        rows = rows.set_index("variation")
        base = rows.loc["Base"]
        scenarios.append({
            "id": sid, "rule": base.rule, "a1": base.action1, "a2": base.action2,
            "ctx": {v[0]: rows.loc[v, "context"] for v in versions if v in rows.index},
        })
    scenarios.sort(key=lambda d: (d["rule"], d["a2"].lower()))

    # MAL per model and scenario as [Base, C, E, R] (null where the variation does not exist)
    mals = {}
    for mid, *_ in MODELS:
        m = mal[mal.model_id == mid]
        out = {}
        for sid, g in m.groupby("scenario_id"):
            gv = g.set_index("variation")
            out[sid] = [r3(g.p_action2_base.iloc[0])] + [r3(gv.loc[v, "p_action2_variation"]) if v in gv.index else None
                                                        for v in VARIATIONS]
        mals[mid] = out
    with open(out_path, "w") as f:
        json.dump({"scenarios": scenarios, "mal": mals}, f, ensure_ascii=False, separators=(",", ":"))
    print("wrote", out_path)

    featured = []
    for sid, title in FEATURED.items():
        rows = scen[scen.scenario_id == sid].set_index("variation")
        base = rows.loc["Base"]
        featured.append({"id": sid, "title": title, "rule": base.rule, "action1": base.action1, "action2": base.action2,
                         "contexts": {v: rows.loc[v, "context"] for v in versions}})
    # Mean MAL across all models per featured scenario and version (used by the hero).
    sub = mal[mal.scenario_id.isin(FEATURED)]
    avg = {sid: {"Base": r3(g.drop_duplicates("model_id").p_action2_base.mean()),
                 **{v: r3(gv.p_action2_variation.mean()) for v, gv in g.groupby("variation")}}
           for sid, g in sub.groupby("scenario_id")}
    return {"scenarios": featured, "avg": avg}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="../contextual_moralchoice")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    site = build_site(args.repo)
    site["explorer"] = build_explorer(args.repo, os.path.join(args.out, "explorer.json"))
    with open(os.path.join(args.out, "site.json"), "w") as f:
        json.dump(site, f, ensure_ascii=False, separators=(",", ":"))
    print("wrote", os.path.join(args.out, "site.json"))


if __name__ == "__main__":
    main()
