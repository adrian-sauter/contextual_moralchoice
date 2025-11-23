from openai import OpenAI
import os, json

import argparse

################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="LLM Evaluation on MoralChoice")

parser.add_argument(
    "--model-name",
    #default="openai/text-babbage-001",
    type=str,
    help="Model to download --- see models.py for an overview of supported models",
)

parser.add_argument(
    "--batch-index",
    default=[1],
    type=int,
    help="Batch index to download (multiple allowed)",
    nargs="+"
)

args = parser.parse_args()

#BASE_PATH = "/home/asauter1/contextual_moralchoice/"

with open(f'api_keys/openai_key.txt', encoding="utf-8") as f:
    key = f.read()

client = OpenAI(api_key=key)

BATCH_INDICES = args.batch_index
MODEL_NAME = args.model_name

for BATCH_INDEX in BATCH_INDICES:
    batch_id = open(f"data/responses/openai_models/batch_id_{MODEL_NAME}_batch{BATCH_INDEX}.txt").read().strip()
    batch = client.batches.retrieve(batch_id)
    if batch.output_file_id is None:
        print(f"Batch {batch_id} is not yet complete. Skipping.")
        error = client.files.content(batch.error_file_id)
        print(f"Errors:\n{error.read().decode('utf-8')}")
        continue
    result = client.files.content(batch.output_file_id)
    output_path = f"data/scenarios/openai_batches/results_{MODEL_NAME}_batch{BATCH_INDEX}.jsonl"
    with open(output_path, "wb") as f:
        f.write(result.read())
    print(f"✅ Saved {output_path}")

# for i in [1, 2]:
#     batch_id = open(f"batch_id_part{i}.txt").read().strip()
#     batch = client.batches.retrieve(batch_id)
#     result_bytes = client.files.content(batch.output_file_id)
#     output_path = f"data/scenarios/all_scenarios_results_part{i}.jsonl"
#     with open(output_path, "wb") as f:
#         f.write(result_bytes)
#     print(f"✅ Saved {output_path}")