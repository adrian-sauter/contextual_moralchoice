import json
import pandas as pd
import numpy as np
import torch
import pickle
import os
import argparse
from src.config import PATH_RESULTS

################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Generate Steering Vectors")
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
    "--vector-type",
    default=["unweighted", "weighted"],
    type=str,
    nargs="+",
    help="Type of vector to generate (e.g., 'unweighted', 'weighted')"
)

parser.add_argument(
    "--question-type",
    type=str,
    help="Type of question to generate vectors for",
    nargs="+"
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

# 2. Load the Activations Dataframe
ACT_PATH = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_activations/{FILE_BASE}_activations.pkl"
df = pd.DataFrame(pd.read_pickle(ACT_PATH))
df = df[df['question_type'].isin(args.question_type)] if args.question_type else df
print(f"Loaded {len(df)} activations from {ACT_PATH}")

# 3. Setup output directory
VEC_DIR = f"{PATH_RESULTS}/{args.experiment_name}/vectors"
os.makedirs(VEC_DIR, exist_ok=True)

# 4. Filter for Training IDs (Excluding the Test IDs used in CV)
scenario_df = pd.read_csv(f"data/scenarios/variations_moralchoice_{args.dataset}_ambiguity.csv")
full_ids = [id for id in scenario_df['scenario_id'].unique() if scenario_df[scenario_df['scenario_id'] == id].shape[0] == 4]

# Re-create the same test split to avoid contamination
test_size = layer_summary['config']['test_size']
if args.test_id_path:
    with open(args.test_id_path, "r") as f:
        id_dict = json.load(f)
    test_ids = id_dict['test_ids']
    print(f'Loaded {len(test_ids)} test IDs from {args.test_id_path}: {test_ids}')
else:
    test_ids = np.random.choice(full_ids, size=int(test_size * len(full_ids)), replace=False)
print(f'Test IDs ({len(test_ids)}): {test_ids}')
train_pool = [id_ for id_ in full_ids if id_ not in test_ids]

WEIGHTED = args.vector_type == 'weighted'
print(f"Generating {'weighted' if WEIGHTED else 'unweighted'} vectors using {len(train_pool)} training scenarios (Test set excluded).")

# Balance training set across variations
min_size = min([len(get_ids(scenario_df, var)) for var in ['Consequentialist', 'Emotional', 'Relational']])
train_size = min_size - len(test_ids)
print(f'Train size per variation: {train_size}, Test size: {len(test_ids)}')

cons_ids = prep_var_ids(scenario_df, 'Consequentialist', test_ids, sample_size=train_size)
emo_ids = prep_var_ids(scenario_df, 'Emotional', test_ids, sample_size=train_size)
rel_ids = prep_var_ids(scenario_df, 'Relational', test_ids, sample_size=train_size)

id_groups = {
    'Consequentialist': cons_ids,
    'Emotional': emo_ids,
    'Relational': rel_ids
}

print(f"Generating vectors using {sum(len(ids) for ids in id_groups.values())} training scenarios (Test set excluded).")

# 5. Master Vector Generation Function
# def get_vector(v_name, target_layer, weighted=False):
#     base_sub = df[(df['variation'] == 'Base') & (df['scenario_id'].isin(id_groups[v_name]))]
#     var_sub = df[(df['variation'] == v_name) & (df['scenario_id'].isin(id_groups[v_name]))]
    
#     deltas = []
#     weights = []
    
#     for s_id in id_groups[v_name]:
#         b_scen = base_sub[base_sub['scenario_id'] == s_id]
#         v_scen = var_sub[var_sub['scenario_id'] == s_id]

#         if b_scen.empty or v_scen.empty:
#             print(f'Warning: Missing data for scenario_id {s_id} in variation {v_name}. Skipping this scenario.')
#             continue

#         if weighted:
#             scenario_weights = []

#             ab_base = base_sub[base_sub['question_type'] == 'ab']
#             ab_var = var_sub[var_sub['question_type'] == 'ab']

#             for q_order in ab_base['question_ordering'].unique():
#                 b_row = ab_base[ab_base['question_ordering'] == q_order]
#                 v_row = ab_var[ab_var['question_ordering'] == q_order]
#                 if not b_row.empty and not v_row.empty:
#                     #print(f'b_row.shape: {b_row.shape}, v_row.shape: {v_row.shape}')
#                     p_base = b_row.iloc[0]['scores']['prob_action2']
#                     p_var = v_row.iloc[0]['scores']['prob_action2']
#                     scenario_weights.append(max(0, p_var - p_base))
#             if not scenario_weights:
#                 continue

#             current_scenario_weight = np.mean(scenario_weights)
#             if current_scenario_weight <= 0:
#                 continue
#         else:
#             current_scenario_weight = 1.0
        
#         for q_type in b_scen['question_type'].unique():
#             for q_order in b_scen['question_ordering'].unique():
#                 b_match = b_scen[(b_scen['question_type'] == q_type) & (b_scen['question_ordering'] == q_order)]
#                 v_match = v_scen[(v_scen['question_type'] == q_type) & (v_scen['question_ordering'] == q_order)]
#                 #print(f'b_match.shape: {b_match.shape}, v_match.shape: {v_match.shape}')
#                 if b_match.empty or v_match.empty:
#                     print(f'Warning: Missing data for scenario_id {s_id}, question_type {q_type}, question_ordering {q_order}. Skipping this combination.')
#                     continue

#                 act_b = b_match.iloc[0]['activations'][f'layer_{target_layer}']
#                 act_v = v_match.iloc[0]['activations'][f'layer_{target_layer}']
#                 delta = act_v - act_b
#                 deltas.append(delta)
#                 weights.append(current_scenario_weight)
#                 #print(f'Scenario {s_id} | Question Type {q_type} | Question Ordering {q_order} | Weight: {current_scenario_weight:.4f}')

#     if not deltas:
#         return None
    
#     deltas_tensor = torch.stack(deltas).to(dtype=torch.float32)
#     weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

#     master_vector = (deltas_tensor * weights_tensor).sum(dim=0) / weights_tensor.sum()
#     return master_vector

# # 5. Master Vector Computation Logic
# # This is for cases where probabilitied were not derived (e.g., when using "compare" or "repeat"). We can still use the vectors from the ab question type, we just need to load it.

ab_weights_df = pd.DataFrame(pd.read_pickle(f"/home/asauter1/contextual_moralchoice/data/responses/llama_extraction/high_activations/llama-3.1-8b-instruct_activations.pkl"))
def get_vector(v_name, target_layer, weighted=False):
    base_sub = df[(df['variation'] == 'Base') & (df['scenario_id'].isin(id_groups[v_name]))]
    var_sub = df[(df['variation'] == v_name) & (df['scenario_id'].isin(id_groups[v_name]))]

    weights_base_sub = ab_weights_df[(ab_weights_df['variation'] == 'Base') & (ab_weights_df['scenario_id'].isin(id_groups[v_name]))]
    weights_var_sub = ab_weights_df[(ab_weights_df['variation'] == v_name) & (ab_weights_df['scenario_id'].isin(id_groups[v_name]))]

    
    deltas = []
    weights = []
    
    for s_id in id_groups[v_name]:
        b_scen = base_sub[base_sub['scenario_id'] == s_id]
        v_scen = var_sub[var_sub['scenario_id'] == s_id]

        weights_b_scen = weights_base_sub[weights_base_sub['scenario_id'] == s_id]
        weights_v_scen = weights_var_sub[weights_var_sub['scenario_id'] == s_id]

        if b_scen.empty or v_scen.empty:
            print(f'Warning: Missing data for scenario_id {s_id} in variation {v_name}. Skipping this scenario.')
            continue

        if weighted:
            scenario_weights = []

            ab_base = weights_b_scen[weights_b_scen['question_type'] == 'ab']
            ab_var = weights_v_scen[weights_v_scen['question_type'] == 'ab']

            for q_order in ab_base['question_ordering'].unique():
                b_row = ab_base[ab_base['question_ordering'] == q_order]
                v_row = ab_var[ab_var['question_ordering'] == q_order]
                if not b_row.empty and not v_row.empty:
                    p_base = b_row.iloc[0]['scores']['prob_action2']
                    p_var = v_row.iloc[0]['scores']['prob_action2']
                    scenario_weights.append(max(0, p_var - p_base))
            if not scenario_weights:
                continue

            current_scenario_weight = np.mean(scenario_weights)
            if current_scenario_weight <= 0:
                continue
        else:
            current_scenario_weight = 1.0
        
        for q_type in b_scen['question_type'].unique():
            for q_order in b_scen['question_ordering'].unique():
                b_match = b_scen[(b_scen['question_type'] == q_type) & (b_scen['question_ordering'] == q_order)]
                v_match = v_scen[(v_scen['question_type'] == q_type) & (v_scen['question_ordering'] == q_order)]
                #print(f'b_match.shape: {b_match.shape}, v_match.shape: {v_match.shape}')
                if b_match.empty or v_match.empty:
                    print(f'Warning: Missing data for scenario_id {s_id}, question_type {q_type}, question_ordering {q_order}. Skipping this combination.')
                    continue

                act_b = b_match.iloc[0]['activations'][f'layer_{target_layer}']
                act_v = v_match.iloc[0]['activations'][f'layer_{target_layer}']
                delta = act_v - act_b
                deltas.append(delta)
                weights.append(current_scenario_weight)
                #print(f'Scenario {s_id} | Question Type {q_type} | Question Ordering {q_order} | Weight: {current_scenario_weight:.4f}')

    if not deltas:
        return None
    
    deltas_tensor = torch.stack(deltas).to(dtype=torch.float32)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

    master_vector = (deltas_tensor * weights_tensor).sum(dim=0) / weights_tensor.sum()
    return master_vector

# 6. Run for all variations
for v_name in ['Consequentialist', 'Emotional', 'Relational']:
    target_L = layer_summary['best_layers'][v_name]['recommended_layer']
    print(f"\nProcessing {v_name} (Target Layer: {target_L})")

    for vector_type in args.vector_type:
    
        if vector_type == 'weighted':
            vec_weighted = get_vector(v_name, target_L, weighted=True)
            print(f" Weighted vector norm for {v_name}: {torch.norm(vec_weighted):.4f}")
            print(f'Weighted vector first 5 values for {v_name}: {vec_weighted[:5]}')
            torch.save(vec_weighted, f"{VEC_DIR}/{v_name.lower()}_weighted_L{target_L}.pt")
        else:
            vec_unweighted = get_vector(v_name, target_L, weighted=False)
            print(f" Unweighted vector norm for {v_name}: {torch.norm(vec_unweighted):.4f}")
            print(f'Unweighted vector first 5 values for {v_name}: {vec_unweighted[:5]}')
            torch.save(vec_unweighted, f"{VEC_DIR}/{v_name.lower()}_unweighted_L{target_L}.pt")
    
        print(f"  {vector_type.capitalize()} vector saved for {v_name}")

print(f"\nAll vectors successfully generated in: {VEC_DIR}")