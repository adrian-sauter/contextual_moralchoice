"""Response Collection: Aggregate Model Responses into a single csv per model"""

import os
import pickle
import argparse
import pandas as pd

from src.config import PATH_RESULTS


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
        if name.endswith(".pickle"):
            path_file = os.path.join(path, name)
        #if name.split('.')[0][-1] in ["C", "E", "R"]:
        #    continue

        with open(path_file, "rb") as f:
            tmp = pickle.load(f)
            results.append(tmp)
            #variation = name.split("_")[3][0]
            #reverse_variation_mappings = {"B": "Base", "C": "Consequentialist", "E": "Emotional", "R": "Relational"}
            #tmp["variation"] = reverse_variation_mappings[variation]
            #cols = ["scenario_id", "variation", "model_id", "question_type", "question_ordering", "question_header", "question_text", "eval_technique", "eval_top_p", "eval_temperature", "eval_sample_nb", "timestamp", "answer_raw", "answer", "decision"]
            #ordered_tmp = tmp[cols]
            #results.append(ordered_tmp)

df_results = pd.concat(results)
# Store one csv per model
if not os.path.exists(path_results):
    os.makedirs(path_results)

for model_id in df_results["model_id"].unique():
    results_model = df_results.loc[df_results["model_id"] == model_id]
    if 'gpt-' in model_id:
        company_name = 'openai'
    else:
        company_name = model_id.split('/')[0]
    if args.pickle:
        results_model.to_pickle(
            f"{path_results}/{company_name}_{model_id.split('/')[-1]}.pkl"
        )
    else:
        results_model.to_csv(
            f"{path_results}/{company_name}_{model_id.split('/')[-1]}.csv"
        )