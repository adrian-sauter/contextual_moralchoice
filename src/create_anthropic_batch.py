import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

import json, os, pandas as pd
from tqdm import tqdm
import argparse
from src.config import PATH_QUESTION_TEMPLATES
from src.question_form_generator import get_question_form


################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Batch creation of MoralChoice prompts for OpenAI API")
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
    "--question-types",
    default=["ab"],
    type=str,
    help="Question Templates to evaluate",
    nargs="+",
)

parser.add_argument(
    "--eval-technique",
    default="top_p_sampling",
    type=str,
    help="Evaluation Technique (top_p_sampling is only supported technique right now)",
)
parser.add_argument(
    "--eval-top-p", default=1.0, type=float, help="Top-P parameter for top-p sampling"
)
parser.add_argument(
    "--eval-temp", default=1.0, type=float, help="Temperature for sampling"
)
parser.add_argument(
    "--eval-max-tokens",
    default=200,
    type=int,
    help="Max. number of tokens per completion",
)
parser.add_argument(
    "--eval-nb-samples", default=1, type=int, help="Nb. of samples per question form"
)

args = parser.parse_args()

################################################################################################
# SETUP
################################################################################################

# Load scenarios
scenarios = pd.read_csv(f"data/scenarios/variations_moralchoice_{args.dataset}_ambiguity.csv")

with open('api_keys/anthropic_key.txt', encoding="utf-8") as f:
    key = f.read()
client = anthropic.Anthropic(api_key=key)

################################################################################################
# CREATE SMALL BATCH
################################################################################################
# requests = []

# for _, scenario in tqdm(scenarios.head(10).iterrows(), total=10):
#     question_form, _ = get_question_form(
#         scenario=scenario,
#         question_type="ab",
#         question_ordering=0,
#         system_instruction=True,
#     )

#     system_prompt = question_form["question_header"].strip()
#     user_prompt = question_form["question"].strip()

#     request = Request(
#         custom_id=f"{scenario['scenario_id']}_{scenario['variation']}_ab_0_0",
#         params=MessageCreateParamsNonStreaming(
#             model=args.model_name,
#             system=system_prompt,
#             messages=[{"role": "user", "content": user_prompt}],
#             #temperature=1.0,
#             top_p=1.0,
#             max_tokens=200,
#         ),
#     )
#     requests.append(request)

# print(f"🧩 Created {len(requests)} requests for test batch")

# ################################################################################################
# # SUBMIT TEST BATCH
# ################################################################################################
# batch = client.messages.batches.create(requests=requests)
# print(f"✅ Test batch created successfully!")
# print(f"Batch ID: {batch.id}")

requests = []

for k, (identifier, scenario) in tqdm(enumerate(scenarios.iterrows()), total=len(scenarios)):

    for question_type in args.question_types:
        for question_ordering in [0, 1]:
            question_form, action_mapping = get_question_form(
                scenario=scenario,
                question_type=question_type,
                question_ordering=question_ordering,
                system_instruction=True,
            )

            # same system/user structure as before
            system_prompt = question_form["question_header"].strip()
            user_prompt = question_form["question"].strip()

            # make one batch item per sample
            for nb_query in range(args.eval_nb_samples):
                request = Request(
                    custom_id = f"{scenario['scenario_id']}_{scenario['variation']}_{question_type}_{question_ordering}_{nb_query}",
                    params=MessageCreateParamsNonStreaming(
                        model=args.model_name,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt},],
                        #temperature=args.eval_temp,
                        top_p=args.eval_top_p,
                        max_tokens=args.eval_max_tokens,
                    ),
                )
                requests.append(request)

print(f"Total number of requests to batch: {len(requests)}")

# -------------------------------
# Submit batch to Anthropic
batch = client.messages.batches.create(requests=requests)
print(f"Batch created: {batch}")
# -------------------------------
