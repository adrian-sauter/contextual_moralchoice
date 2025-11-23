from together import Together
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

# path where JSONL will be written
#batch_file_path = f"data/scenarios/all_scenarios_batched.jsonl"
#os.makedirs("data/scenarios", exist_ok=True)

# collect all prompts
batch_lines = []

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
                body = {
                    "model": args.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": args.eval_temp,
                    "top_p": args.eval_top_p,
                    "max_tokens": args.eval_max_tokens,
                }

                line = {
                    "custom_id": f"{scenario['scenario_id']}_{scenario['variation']}_{question_type}_{question_ordering}_{nb_query}",
                    #"method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                batch_lines.append(line)

# -------------------------------
# Split batch into chunks of requests
# -------------------------------
BATCH_LIMIT = 50000
num_batches = (len(batch_lines) + BATCH_LIMIT - 1) // BATCH_LIMIT  # ceil division

base_path = f"data/scenarios/together_batches/"
os.makedirs(base_path, exist_ok=True)

for i in range(num_batches):
    start = i * BATCH_LIMIT
    end = min((i + 1) * BATCH_LIMIT, len(batch_lines))
    model_name = args.model_name.replace("/", "_")
    split_path = f"{base_path}{model_name}_part{i+1}.jsonl"

    with open(split_path, "w", encoding="utf-8") as f:
        for line in batch_lines[start:end]:
            f.write(json.dumps(line) + "\n")

    print(f"Wrote {end - start} batch entries to {split_path}")