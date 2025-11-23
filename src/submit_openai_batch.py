from openai import OpenAI
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

with open('api_keys/openai_key.txt', encoding="utf-8") as f:
    key = f.read()

client = OpenAI(api_key=key)

BATCH_INDEX = args.batch_index
MODEL_NAME = args.model_name
batch_path = f"data/scenarios/openai_batches/all_scenarios_batched_{MODEL_NAME}_part{BATCH_INDEX}.jsonl"

batch_input_file = client.files.create(
    file=open(batch_path, "rb"),
    purpose="batch"
)

print(f'batch_input_file: {batch_input_file}')

batch_input_file_id = batch_input_file.id

print(f'batch_input_file_id: {batch_input_file_id}')
batch = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": f"OpenAI API batch {BATCH_INDEX} for model {MODEL_NAME}",
    }
)
print(f"Batch: {batch}")

print(f"✅ Submitted batch: {batch_path}")
print(f"   → Batch ID: {batch.id}")

with open(f"data/responses/openai_models/batch_id_{MODEL_NAME}_batch{BATCH_INDEX}.txt", "w") as idfile:
    idfile.write(batch.id)


# model_names = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-5.1"]

#batch_paths = [f"data/scenarios/all_scenarios_batched_{model_name}_part{BATCH_INDEX}.jsonl" for model_name in model_names]

# for model_name, batch_path in zip(model_names * 4, batch_paths):
#     # Upload batch file to OpenAI
#     batch_input_file = client.files.create(
#         file=open(batch_path, "rb"),
#         purpose="batch"
#     )

#     print(f'batch_input_file: {batch_input_file}')

#     batch_input_file_id = batch_input_file.id

#     print(f'batch_input_file_id: {batch_input_file_id}')
#     batch = client.batches.create(
#         input_file_id=batch_input_file_id,
#         endpoint="/v1/chat/completions",
#         completion_window="24h",
#         metadata={
#             "description": f"OpenAI API batch test for model {model_name}",
#         }
#     )
#     print(f"Batch: {batch}")

#     print(f"✅ Submitted batch: {batch_path}")
#     print(f"   → Batch ID: {batch.id}")

#     with open(f"data/responses/openai_models/batch_id_{model_name}_batch{BATCH_INDEX}.txt", "w") as idfile:
#         idfile.write(batch.id)

# with open(batch_path, "rb") as f:
#         batch = client.batches.create(
#             input_file=f,
#             endpoint="/v1/responses",
#             completion_window="24h",
#         )
#         print(f"✅ Submitted batch: {batch_path}")
#         print(f"   → Batch ID: {batch.id}")
#         with open(f"data/responses/openai_models/batch_id.txt", "w") as idfile:
#             idfile.write(batch.id)


# batch_files = [
#     "data/scenarios/all_scenarios_batched_part1.jsonl",
#     "data/scenarios/all_scenarios_batched_part2.jsonl",
# ]

# for i, batch_path in enumerate(batch_files, start=1):
#     with open(batch_path, "rb") as f:
#         batch = client.batches.create(
#             input_file=f,
#             endpoint="/v1/responses",
#             completion_window="24h",
#         )
#         print(f"✅ Submitted batch {i}: {batch_path}")
#         print(f"   → Batch ID: {batch.id}")
#         with open(f"data/responses/openai_models/batch_id_part{i}.txt", "w") as idfile:
#             idfile.write(batch.id)