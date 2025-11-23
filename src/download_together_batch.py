from together import Together
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

with open(f'api_keys/together_key.txt', encoding="utf-8") as f:
    key = f.read()

client = Together(api_key=key)

BATCH_INDICES = args.batch_index
MODEL_NAME = args.model_name
model_name = args.model_name.replace("/", "_")

for BATCH_INDEX in BATCH_INDICES:
    batch_id = open(f"data/responses/together_models/batch_id_{model_name}_batch{BATCH_INDEX}.txt").read().strip()
    batch = client.batches.get_batch(batch_id)
    if batch.output_file_id is None:
        print(f"Batch {batch_id} is not yet complete. Skipping.")
        error = client.files.retrieve_content(batch.error_file_id)
        print(f"Errors:\n{error.content.decode('utf-8')}")
        continue
    output_path = f"data/scenarios/together_batches/results_{model_name}_batch{BATCH_INDEX}.jsonl"
    result = client.files.retrieve_content(batch.output_file_id, output=output_path)
    # with open(output_path, "wb") as f:
    #     f.write(result.content)
    print(f"✅ Saved {output_path}")
