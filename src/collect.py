"""Response Collection: Aggregate Model Responses into a single csv per model"""

import os
import pickle
import argparse
import pandas as pd

from src.config import PATH_RESULTS
from src.semantic_matching import refine_results_with_llm


################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Collecting Results")
parser.add_argument(
    "--experiment-name",
    default="test",
    type=str,
    help="Name of Experiment - used for logging",
)
parser.add_argument(
    "--dataset", default="high", type=str, help="Dataset to evaluate (low or high)"
)

parser.add_argument(
    "--pickle",
    default=False,
    action="store_true",
    help="Whether to save the aggregated results as a pickle file instead of csv",
)

parser.add_argument(
    "--refine",
    action="store_true",
    help="After aggregating, use an LLM judge to re-match 'refusal'/'invalid' decisions",
)
parser.add_argument(
    "--refiner-model",
    type=str,
    help="Model to use for LLM-based refinement (see src/models.py) --- required if --refine is set",
)
parser.add_argument(
    "--refiner-top-p", default=1.0, type=float, help="Top-P parameter for refiner sampling"
)
parser.add_argument(
    "--refiner-temp", default=0.7, type=float, help="Temperature for refiner sampling"
)
parser.add_argument(
    "--refiner-max-tokens", default=300, type=int, help="Max. number of tokens per refiner completion"
)

args = parser.parse_args()


################################################################################################
# SETUP
################################################################################################
path_results = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}"
path_results_raw = path_results + "_raw"


################################################################################################
# RESPONSE COLLECTION
################################################################################################
# Collect all pickle result files
results = []
for path, subdirs, files in os.walk(path_results_raw):
    for name in files:
        if not name.endswith(".pickle"):
            continue
        path_file = os.path.join(path, name)

        with open(path_file, "rb") as f:
            tmp = pickle.load(f)
            results.append(tmp)

df_results = pd.concat(results)
# Store one csv per model
if not os.path.exists(path_results):
    os.makedirs(path_results)

for model_id in df_results["model_id"].unique():
    results_model = df_results.loc[df_results["model_id"] == model_id]
    if 'gpt-' in model_id:
        filename = f"openai_{model_id}"
    elif 'claude' in model_id:
        filename = f"claude_{model_id}"
    elif '/' in model_id:
        company_name, bare_name = model_id.split('/', 1)
        filename = f"{company_name}_{bare_name}"
    else:
        # already company-prefixed (e.g. some batch-API model IDs use underscores, not slashes)
        filename = model_id

    if args.pickle:
        results_model.to_pickle(f"{path_results}/{filename}.pkl")
    else:
        results_model.to_csv(f"{path_results}/{filename}.csv")

################################################################################################
# LLM-BASED REFINEMENT (optional)
################################################################################################
if args.refine:
    if not args.refiner_model:
        parser.error("--refiner-model is required when --refine is set")

    scenarios = pd.read_csv(f"data/scenarios/variations_moralchoice_{args.dataset}_ambiguity.csv")
    refine_results_with_llm(
        path_results=path_results,
        scenarios=scenarios,
        model_name=args.refiner_model,
        temperature=args.refiner_temp,
        max_tokens=args.refiner_max_tokens,
        top_p=args.refiner_top_p,
    )