import os 
import pandas as pd
import argparse
from tqdm import tqdm

from src.semantic_matching import SemanticMatcherLLM, token_to_action_matching

from src.config import PATH_RESULTS

################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="LLM Evaluation on MoralChoice")
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
    "--model-name",
    #default="openai/text-babbage-001",
    type=str,
    help="Model to use to refine the responses --- see models.py for an overview of supported models",
)
parser.add_argument(
    "--batching",
    action="store_true",
    help="Enable batching for evaluation"
)
parser.add_argument(
    "--eval-top-p", default=1.0, type=float, help="Top-P parameter for top-p sampling"
)
parser.add_argument(
    "--eval-temp", default=1.0, type=float, help="Temperature for sampling"
)
parser.add_argument(
    "--eval-max-tokens",
    default=300,
    type=int,
    help="Max. number of tokens per completion",
)
parser.add_argument(
    "--batchsize",
    default=10,
    type=int,
    help="Batch size for refinement",
)

parser.add_argument(
    "--pickle",
    default=False,
    action="store_true",
    help="Whether to save the results as a pickle file (instead of csv)",
)

args = parser.parse_args()

################################################################################################
# REFINED SEMANTIC MATCHING SETUP
################################################################################################
print(f"START WITH REFINED SEMANTIC MATCHING FOR EXPERIMENT {args.experiment_name} AND MODEL {args.model_name}")
scenarios = pd.read_csv(f"data/scenarios/variations_moralchoice_{args.dataset}_ambiguity.csv")

path_results = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}"

semantic_matcher = SemanticMatcherLLM(
    model_name=args.model_name,
)

df_results = []
for path, subdirs, files in os.walk(path_results):
    for name in files:
        print(f"Found file: {name}")
        if name.endswith(".csv") or name.endswith(".pkl"):
            # if name.endswith("_REFINED.csv") or name.replace(".csv","_REFINED.csv") in files or name.endswith("_REFINED.pkl") or name.replace(".pkl","_REFINED.pkl") in files:
            #     print(f"Skipping already refined file: {name}")
            #     continue
            print(f"Processing file: {name}")
            path_file = os.path.join(path, name)
            if name.endswith(".csv"):
                df = pd.read_csv(path_file)
            else:
                df = pd.read_pickle(path_file)
            # df_update = df.copy()
            # df_update['refined_decision'] = None
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # common saved-index artefact
            df_update = df.copy().reset_index(drop=True)           # guarantees unique RangeIndex
            df_update["refined_decision"] = pd.NA
            #refusal_invalid_df = df_update[df_update['decision'].isin(['refusal','invalid'])]
            if args.batching:
                refusal_invalid_df = df_update[df_update['decision'].isin(['refusal','invalid'])]
                nb_batches = (len(refusal_invalid_df) + args.batchsize - 1) // args.batchsize
                for batch_idx in tqdm(range(nb_batches)):
                    batch_start = batch_idx * args.batchsize
                    batch_end = min((batch_idx + 1) * args.batchsize, len(refusal_invalid_df))
                    batch_df = refusal_invalid_df.iloc[batch_start:batch_end]
                    batch_answers = batch_df['answer'].to_list()
            else:
                # for idx, row in df_update.iterrows():
                #     if idx < 10:
                #         print(f'Processing row {idx}: {row["answer"]}')
                #     if row['decision'] not in ['refusal','invalid']:
                #         df_update.at[idx, 'refined_decision'] = row['decision']
                #         continue
                #     print(f'Processing row {idx}: {row["answer"]}')
                #     scenario = scenarios.loc[(scenarios['scenario_id'] == row['scenario_id']) & (scenarios['variation'] == row['variation'])].iloc[0]
                #     decision = semantic_matcher.semantic_matching_llm(row, scenario, batching=False)
                #     print(f"Model: {row['model_id']}, Scenario: {row['question_text'][-200:]}, Answer: {row['answer']}")
                #     print(f"Original Decision: {row['decision']} --> Refined Decision: {decision}\n")
                #     df_update.at[idx, 'refined_decision'] = decision
                refined_col = df_update.columns.get_loc("refined_decision")
                for i, row in enumerate(df_update.itertuples(index=False)):
                    if row.decision not in ("refusal", "invalid"):
                        df_update.iat[i, refined_col] = row.decision
                        continue

                    scenario = scenarios.loc[
                        (scenarios["scenario_id"] == row.scenario_id) &
                        (scenarios["variation"] == row.variation)
                    ].iloc[0]

                    # if semantic_matching_llm expects a Series/row, pass df_update.iloc[i]
                    decision = semantic_matcher.semantic_matching_llm(df_update.iloc[i], scenario, batching=False)
                    df_update.iat[i, refined_col] = decision

            if args.pickle:
                df_update.to_pickle(path_file.replace(".pkl", "_REFINED.pkl"))
            else:
                df_update.to_csv(path_file.replace(".csv", "_REFINED.csv"), index=False)
                    
#df_results = pd.concat(df_results)

#for model_id in df_results["model_id"].unique():
#    results_model = df_results.loc[df_results["model_id"] == model_id]
#    results_model.to_csv(
#        f"{path_results}/{model_id.split('/')[0]}_{model_id.split('/')[-1]}_REFINED.csv"
#    )