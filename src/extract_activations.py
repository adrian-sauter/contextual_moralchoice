import torch
from functools import partial
from transformer_lens import HookedTransformer

import os
import pickle
import json
import argparse
import pandas as pd
from tqdm import tqdm

from src.models import (
    LlamaModel,
    create_model
)
from src.question_form_generator import get_question_form

from src.config import PATH_RESULTS


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    help="Model to evalute --- see models.py for an overview of supported models",
)

parser.add_argument(
    "--hookpoint",
    default="hook_resid_post",
    type=str,
    help="Hookpoint to extract activations from (only tested with 'hook_resid_post' so far)",
)

parser.add_argument(
    "--question-types",
    default=["ab"],
    type=str,
    help="Question Templates to evaluate",
    nargs="+",
)

parser.add_argument(
    "--return-probs",
    default=False,
    action="store_true",
    help="Whether to return probabilities of generated answer for evaluation (should only be done with A/B prompt format)"
)

args = parser.parse_args()

################################################################################################
# SETUP
################################################################################################

# Load scenarios
#scenarios = pd.read_csv(f"data/scenarios/dummy_set.csv")
scenarios = pd.read_csv(f"data/scenarios/variations_moralchoice_{args.dataset}_ambiguity.csv")

tl_model = create_model(args.model_name, load_with_transformerlens=True)

HOOKPOINT = args.hookpoint
# do handful of layers first
#LAYER_LIST = [0, tl_model.cfg.n_layers//4, tl_model.cfg.n_layers//2, (3*tl_model.cfg.n_layers)//4, tl_model.cfg.n_layers-1]
#LAYER_LIST = sorted(set([int(x) for x in LAYER_LIST]))
LAYER_LIST = list(range(tl_model.n_layers))  # do all layers

results = []
for k, (identifier, scenario) in tqdm(
    enumerate(scenarios.iterrows()),
    total=len(scenarios),
    position=0,
    ncols=100,
    leave=True,
    desc=f"MoralChoice Eval: {args.model_name}",
):
    for question_type in args.question_types:
        for question_ordering in [0, 1]:
            question_form, action_mapping = get_question_form(
                    scenario=scenario,
                    question_type=question_type,
                    question_ordering=question_ordering,
                    system_instruction=True,
                )
            result_base = {"scenario_id": scenario['scenario_id'],
                        "variation": scenario['variation'],
                        "model_id": tl_model.get_model_id(),
                        "question_type": question_type,
                        "question_ordering": question_ordering,
                        "question_header": question_form['question_header'],
                        "question_text": question_form['question']}
            
            activations = tl_model.get_activations_and_scores(
                prompt_base = question_form['question'],
                prompt_system = question_form['question_header'],
                layers = LAYER_LIST,
                action_mapping = action_mapping,
                loc = HOOKPOINT,
                return_probs = args.return_probs
            )
            result = {**result_base, **activations}
            results.append(result)

# Save results
path_output = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_activations"
if not os.path.exists(path_output):
    os.makedirs(path_output)

# Use pickle to save the list of dictionaries directly
with open(f"{path_output}/{args.model_name.split('/')[-1]}_activations.pkl", "wb") as f:
    pickle.dump(results, f)
        
        