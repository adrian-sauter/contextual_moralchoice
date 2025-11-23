from together import Together
import os

import argparse

################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="LLM Evaluation on MoralChoice")

parser.add_argument(
    "--model-name",
    #default="openai/text-babbage-001",
    type=str,
    help="Model to submit batch for --- see models.py for an overview of supported models",
)

parser.add_argument(
    "--batch-index",
    default=1,
    type=int,
    help="Batch index to submit"
)

args = parser.parse_args()

with open('api_keys/together_key.txt', encoding="utf-8") as f:
    key = f.read()

client = Together(api_key=key)

BATCH_INDEX = args.batch_index
MODEL_NAME = args.model_name
model_name = args.model_name.replace("/", "_")

batch_path = f"data/scenarios/together_batches/{model_name}_part{BATCH_INDEX}.jsonl"

batch_input_file = client.files.upload(file=batch_path, purpose="batch-api")

print(f'batch_input_file: {batch_input_file}')

batch_input_file_id = batch_input_file.id

print(f'batch_input_file_id: {batch_input_file_id}')
batch = client.batches.create_batch(
    batch_input_file_id,
    endpoint="/v1/chat/completions"
)
print(f"Batch: {batch}")

print(f"✅ Submitted batch: {batch_path}")
print(f"   → Batch ID: {batch.id}")

with open(f"data/responses/together_models/batch_id_{model_name}_batch{BATCH_INDEX}.txt", "w") as idfile:
    idfile.write(batch.id)
