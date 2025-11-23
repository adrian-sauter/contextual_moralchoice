import anthropic
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
    type=str,
    help="Batch ID to download",
)

args = parser.parse_args()

#BASE_PATH = "/home/asauter1/contextual_moralchoice/"

with open('/home/asauter1/contextual_moralchoice/api_keys/anthropic_key.txt', encoding="utf-8") as f:
    key = f.read()

client = anthropic.Anthropic(api_key=key)

BATCH_ID = args.batch_index
MODEL_NAME = args.model_name

output_path = f"data/scenarios/claude_batches/results_{MODEL_NAME}.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f_out:
    for result in client.messages.batches.results(BATCH_ID):
        rtype = result.result.type
        if rtype == "succeeded":
            print(f"✅ Success: {result.custom_id}")
            # Write raw JSON line
            f_out.write(json.dumps(result.to_dict()) + "\n")
        elif rtype == "errored":
            if getattr(result.result.error, "type", "") == "invalid_request_error":
                print(f"⚠️ Validation error: {result.custom_id}")
            else:
                print(f"⚠️ Server error: {result.custom_id}")
        elif rtype == "expired":
            print(f"⌛ Expired: {result.custom_id}")

print(f"\n✅ Saved all results to {output_path}")

print(f"✅ Saved {output_path}")
