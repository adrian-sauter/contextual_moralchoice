import json
import pandas as pd
import numpy as np
import torch
import pickle
import os
import argparse
from tqdm import tqdm
from src.config import PATH_RESULTS, PATH_RESPONSE_TEMPLATES, VARIATION_MAPPINGS

from src.models import create_model

from src.question_form_generator import get_question_form
from src.semantic_matching import token_to_action_matching

################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Steer Model with Extracted Vectors")
parser.add_argument(
    "--model-name", 
    default="meta/llama-3.1-8b-instruct", 
    type=str
)

parser.add_argument(
    "--experiment-name", 
    type=str, 
    required=True
)

parser.add_argument(
    "--dataset", 
    default="high", 
    type=str
)

parser.add_argument(
    "--random-seed", 
    default=42, 
    type=int
)

parser.add_argument(
    "--test-id-path",
    type=str,
    help="Path to a JSON file containing test IDs to exclude from training"
)

parser.add_argument(
    "--eval-max-tokens",
    default=200,
    type=int,
    help="Max. number of tokens per completion",
)

parser.add_argument(
    "--eval-temp",
    default=1.0,
    type=float,
    help="Temperature for sampling during evaluation",
)

parser.add_argument(
    "--eval-nb-samples", default=1, type=int, help="Nb. of samples per question form"
)

parser.add_argument(
    "--question-types",
    default=["ab"],
    type=str,
    help="Question Templates to evaluate",
    nargs="+",
)

parser.add_argument(
    "--intervention-modes",
    default=["last_token", "generated_only", "all_positions"],
    type=str,
    help="Steering Intervention Mode(s) to evaluate",
    nargs="+",
)

parser.add_argument(
    "--get-probs",
    default=False,
    action="store_true",
    help="Whether to return probabilities of generated answer for evaluation (should only be done with A/B prompt format)"
)

parser.add_argument(
    "--vector-type",
    default=["unweighted"],
    type=str,
    nargs="+",
    help="Type of vector to use for steering (e.g., 'unweighted', 'weighted')"
)

args = parser.parse_args()

def get_ids(scenario_df, var):
    return scenario_df.loc[scenario_df['variation'] == var, 'scenario_id'].unique()

def filter_ids(orig_ids, ids_to_filter, exclude=False):
    if exclude:
        return [id_ for id_ in orig_ids if id_ not in ids_to_filter]
    else:
        return [id_ for id_ in orig_ids if id_ in ids_to_filter]
    
def sample_ids(orig_ids, sample_size):
    return np.random.choice(orig_ids, size=sample_size, replace=False)

def prep_var_ids(scenario_df, var, test_ids, sample_size=None):
    ids = get_ids(scenario_df, var)
    ids = filter_ids(ids, test_ids, exclude=True)
    if sample_size is not None:
        ids = sample_ids(ids, sample_size)
    return ids

# Set seed to ensure Train/Test split matches your Layer Selection script
np.random.seed(args.random_seed)

# 1. Load the Recommended Layers
FILE_BASE = args.model_name.split('/')[-1]
SUMMARY_PATH = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_activations/{FILE_BASE}_layer_selection_summary.json"

with open(SUMMARY_PATH, "r") as f:
    layer_summary = json.load(f)

VEC_DIR = f"{PATH_RESULTS}/{args.experiment_name}/vectors"
vectors = {
    'Consequentialist': {},
    'Emotional': {},
    'Relational': {}
}
for v_name in ['Consequentialist', 'Emotional', 'Relational']:
    layer = layer_summary['best_layers'][v_name]['recommended_layer']
    vectors[v_name]['layer'] = layer
    for weight_type in args.vector_type:
        vectors[v_name][weight_type] = torch.load(f"{VEC_DIR}/{v_name.lower()}_{weight_type}_L{layer}.pt")
        print(f"Loaded vector for {v_name} ({weight_type}) from {VEC_DIR}/{v_name.lower()}_{weight_type}_L{layer}.pt")
print(f"\nAll vectors successfully loaded from: {VEC_DIR}")

# 4. Filter for Training IDs (Excluding the Test IDs used in CV)
scenario_df = pd.read_csv(f"data/scenarios/variations_moralchoice_{args.dataset}_ambiguity.csv")
full_ids = [id for id in scenario_df['scenario_id'].unique() if scenario_df[scenario_df['scenario_id'] == id].shape[0] == 4]

# Re-create the same test split to avoid contamination
test_size = layer_summary['config']['test_size']
if args.test_id_path:
    with open(args.test_id_path, "r") as f:
        id_dict = json.load(f)
    test_ids = id_dict['test_ids']
    print(f'Loaded {len(test_ids)} test IDs from {args.test_id_path}.')
else:
    test_ids = np.random.choice(full_ids, size=int(test_size * len(full_ids)), replace=False)
    print(f'No test ID file provided. Randomly sampled {len(test_ids)} test IDs based on test size of {test_size}.')
print(f'Test IDs ({len(test_ids)}): {test_ids}')

# Load refusals and common answer patterns
with open(f"{PATH_RESPONSE_TEMPLATES}/refusals.txt", encoding="utf-8") as f:
    refusals = f.read().splitlines()

response_patterns = {}
for question_type in args.question_types:
    with open(f"{PATH_RESPONSE_TEMPLATES}/{question_type}.json", encoding="utf-8") as f:
        response_patterns[question_type] = json.load(f)

path_model = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_raw/{args.model_name.split('/')[-1]}"
for question_type in args.question_types:
    path_model_questiontype = path_model + f"/{question_type}"
    if not os.path.exists(path_model_questiontype):
        os.makedirs(path_model_questiontype)

model = create_model(args.model_name, load_with_transformerlens=True)


scenario_df = scenario_df[scenario_df['scenario_id'].isin(test_ids)]
    

for k, (identifier, scenario) in tqdm(
    enumerate(scenario_df.iterrows()),
    total=len(scenario_df),
    position=0,
    ncols=100,
    leave=True,
    desc=f"MoralChoice Eval: {model.get_model_id()}",
):
    for question_type in args.question_types:
        results = []
        for question_ordering in [0, 1]:
            var = scenario['variation']
            question_form, action_mapping = get_question_form(
                scenario=scenario, 
                question_type=question_type, 
                question_ordering=question_ordering, 
                system_instruction=True
            )
            result_base = {
                "scenario_id": scenario['scenario_id'],
                "variation": scenario['variation'],
                "model_id": model.get_model_id(),
                "question_type": question_type,
                "question_ordering": question_ordering,
                "question_header": question_form['question_header'],
                "question_text": question_form['question']
            }
            for steering_mode in args.intervention_modes:
                result_base["steering_mode"] = steering_mode
                for vec_type in args.vector_type:
                    result_base["vector_type"] = vec_type
                    for alpha in [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
                        result_base["alpha"] = alpha
                        if var == "Base":  # get all 3 variations' vectors for the Base scenario to test steering in both directions
                            for var_type in ["Consequentialist", "Emotional", "Relational"]:
                                for nb_query in range(args.eval_nb_samples):
                                    result_base['eval_sample_nb'] = nb_query
                                    response = model.get_steered_answer(
                                        prompt_base = question_form['question'],
                                        prompt_system = question_form['question_header'],
                                        steering_vector = vectors[var_type][vec_type],
                                        steering_mode = steering_mode,
                                        alpha = alpha,
                                        layer = vectors[var_type]['layer'],
                                        max_tokens = args.eval_max_tokens,
                                        temperature = args.eval_temp,
                                        action_mapping = action_mapping,
                                        return_probs = args.get_probs,
                                    )
                                    response["decision"] = token_to_action_matching(
                                        response["answer"],
                                        scenario,
                                        response_patterns,
                                        question_type,
                                        action_mapping,
                                        refusals,
                                    )
                                    result = {**result_base, **response}
                                    result["var_vector"] = var_type
                                    results.append(result)
                        else:
                            for nb_query in range(args.eval_nb_samples):
                                result_base['eval_sample_nb'] = nb_query
                                response = model.get_steered_answer(
                                    prompt_base = question_form['question'],
                                    prompt_system = question_form['question_header'],
                                    steering_vector = vectors[var][vec_type],
                                    steering_mode = steering_mode,
                                    alpha = alpha,
                                    layer = vectors[var]['layer'],
                                    max_tokens = args.eval_max_tokens,
                                    temperature = args.eval_temp,
                                    action_mapping = action_mapping,
                                    return_probs = args.get_probs
                                )
                                response["decision"] = token_to_action_matching(
                                    response["answer"],
                                    scenario,
                                    response_patterns,
                                    question_type,
                                    action_mapping,
                                    refusals,
                                )
                                result = {**result_base, **response}
                                result["var_vector"] = var
                                results.append(result)
        with open(f'{path_model}/{question_type}/scenario_{scenario["scenario_id"]}_{VARIATION_MAPPINGS[scenario["variation"]]}.pickle', 
                'wb',
        ) as f:
            pickle.dump(pd.DataFrame(results), f)