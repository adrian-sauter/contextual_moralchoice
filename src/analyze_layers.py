import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import argparse

from src.config import PATH_RESULTS

################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Analyze Layer Activations for MoralChoice - Layer Selection")
parser.add_argument(
    "--model-name",
    default="meta/llama-3.1-8b-instruct",
    type=str,
    help="Path to the pickle file containing activations",
)

parser.add_argument(
    "--experiment-name",
    type=str,
    help="Name of Experiment - used for logging",
)

parser.add_argument(
    "--dataset", default="high", type=str, help="Dataset to evaluate (low or high)"
)

parser.add_argument(
    "--test-size",
    default=0.3,
    type=float,
    help="Proportion of scenarios to use as test set for each CV run",
)

parser.add_argument(
    "--cross-validation-runs",
    default=10,
    type=int,
    help="Number of cross validation runs to determine the best layer",
)

parser.add_argument(
    "--random-seed",
    default=42,
    type=int,
    help="Random seed for reproducibility",
)

parser.add_argument(
    "--n_layers",
    default=32,
    type=int,
    help="Number of layers in the model",
)

args = parser.parse_args()

# Set random seed
np.random.seed(args.random_seed)

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

FILE_NAME = args.model_name.split('/')[-1] + "_activations.pkl"
df_list = pd.read_pickle(f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_activations/{FILE_NAME}")
print(f"Loading activations from {PATH_RESULTS}/{args.experiment_name}/{args.dataset}_activations/{FILE_NAME}...")
df = pd.DataFrame(df_list)
print(f'Dataframe loaded with shape: {df.shape}')
scenario_df = pd.read_csv(f"data/scenarios/variations_moralchoice_{args.dataset}_ambiguity.csv")
os.makedirs(f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_activations", exist_ok=True)

# Get all valid scenario IDs (those with all 4 variations)
full_ids = []
for id in scenario_df['scenario_id'].unique():
    id_df = scenario_df[scenario_df['scenario_id'] == id]
    if id_df.shape[0] == 4:
        full_ids.append(id)

print(f'Total valid scenarios: {len(full_ids)}')
print(f'Running {args.cross_validation_runs} CV runs with {args.test_size*100}% test split each')

# Store results across all runs
cv_results = {
    'Consequentialist': [],
    'Emotional': [],
    'Relational': []
}

for run in range(args.cross_validation_runs):
    print(f'\n=== Cross Validation Run {run+1} / {args.cross_validation_runs} ===')
    
    # Create new train/test split for this run
    test_ids = np.random.choice(full_ids, size=int(args.test_size * len(full_ids)), replace=False)
    train_pool = [id_ for id_ in full_ids if id_ not in test_ids]
    
    # Balance training set across variations
    min_size = min([len(get_ids(scenario_df, var)) for var in ['Consequentialist', 'Emotional', 'Relational']])
    #print(f'Min group size: {min_size}')
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
    base_df = df.loc[(df['variation'] == 'Base')]

    for v_name, train_ids in id_groups.items():
        print(f"\n--- Evaluating Layers for {v_name} ---")
        layer_accs = []
        
        var_df = df[df['variation'] == v_name]
        
        def build_xy(scen_ids, layer_num):
            """Build X, y arrays for given scenario IDs and layer"""
            b_sub = base_df[base_df['scenario_id'].isin(scen_ids)]
            v_sub = var_df[(var_df['scenario_id'].isin(scen_ids) & (var_df['variation'] == v_name))]
            combined = pd.concat([b_sub, v_sub])
            
            X = np.stack([
                row['activations'][f'layer_{layer_num}'].numpy()
                for _, row in combined.iterrows()
            ])
            # Labels: 0 for Base, 1 for Variation
            y = (combined['variation'] == v_name).astype(int).values
            return X, y
        
        # Evaluate each layer
        for L in range(args.n_layers):
            X_train, y_train = build_xy(train_ids, L)
            X_test, y_test = build_xy(test_ids, L)

            print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
            print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

            # Train probe
            probe = LogisticRegression(max_iter=2000, solver='lbfgs')
            probe.fit(X_train, y_train)
            
            # Test accuracy
            acc = accuracy_score(y_test, probe.predict(X_test))
            layer_accs.append(acc)
            
            print(f"    Layer {L}: Accuracy = {acc:.4f}")

        # Store this run's results
        cv_results[v_name].append(layer_accs)
        
        # Report best layer for this run
        best_layer = np.argmax(layer_accs)
        print(f"\n  Best layer this run: Layer {best_layer} (Acc: {layer_accs[best_layer]:.4f})")

# Aggregate results across runs
print(f'\n{"="*70}')
print(f'=== FINAL RESULTS: Best Layers Across {args.cross_validation_runs} CV Runs ===')
print(f'{"="*70}\n')

final_results = {}
for var in cv_results:
    # Convert to numpy array: shape (n_runs, n_layers)
    accs_array = np.array(cv_results[var])
    
    # Mean and std across runs for each layer
    mean_accs = np.mean(accs_array, axis=0)
    std_accs = np.std(accs_array, axis=0)
    
    # Find best layer based on mean accuracy
    best_layer = np.argmax(mean_accs)
    
    # Count how often each layer was best across individual runs
    best_layers_per_run = [np.argmax(run_accs) for run_accs in accs_array]
    layer_counts = np.bincount(best_layers_per_run, minlength=args.n_layers)
    most_frequent_best = np.argmax(layer_counts)
    
    final_results[var] = {
        'mean_accs': mean_accs.tolist(),
        'std_accs': std_accs.tolist(),
        'best_layer_by_mean': int(best_layer),
        'best_layer_mean_acc': float(mean_accs[best_layer]),
        'best_layer_std_acc': float(std_accs[best_layer]),
        'most_frequent_best_layer': int(most_frequent_best),
        'frequency_of_most_common': int(layer_counts[most_frequent_best]),
        'all_runs_data': accs_array.tolist()
    }
    
    print(f'{var}:')
    print(f'  Best layer (by mean accuracy): Layer {best_layer}')
    print(f'  Mean accuracy: {mean_accs[best_layer]:.4f} ± {std_accs[best_layer]:.4f}')
    print(f'  Most frequently best layer: Layer {most_frequent_best} ({layer_counts[most_frequent_best]}/{args.cross_validation_runs} runs)')
    print()


# Save results
print(f'{"="*70}')
print(f'Saving results...\n')

output_path = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_activations/{FILE_NAME.replace('_activations.pkl', '')}_layer_selection_cv.pkl"
with open(output_path, "wb") as f:
    pickle.dump(final_results, f)
print(f'Full results saved at: {output_path}')

# Save readable summary
summary = {
    'config': {
        'cv_runs': args.cross_validation_runs,
        'test_size': args.test_size,
        'random_seed': args.random_seed
    },
    'best_layers': {}
}

for var in final_results:
    summary['best_layers'][var] = {
        'recommended_layer': int(final_results[var]['best_layer_by_mean']),
        'mean_accuracy': float(final_results[var]['best_layer_mean_acc']),
        'std_accuracy': float(final_results[var]['best_layer_std_acc']),
        'most_frequent_best': int(final_results[var]['most_frequent_best_layer']),
        'frequency': f"{final_results[var]['frequency_of_most_common']}/{args.cross_validation_runs}"
    }

summary_path = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_activations/{FILE_NAME.replace('_activations.pkl', '')}_layer_selection_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved at: {summary_path}')

print(f'\n{"="*70}')
print('Layer selection complete!')
print(f'{"="*70}')