from openai import OpenAI
import datetime

import argparse 

################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Check status of OpenAI batches")
parser.add_argument(
    "--limit",
    type=int,
    help="Number of batches to list",
    default=4
)

args = parser.parse_args()

with open('/home/asauter1/contextual_moralchoice/api_keys/openai_key.txt', encoding="utf-8") as f:
    key = f.read()

client = OpenAI(api_key=key)

batches = client.batches.list(limit=args.limit)
for batch in batches.data:
    print(f"ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Model: {batch.model}")
    print(f"{batch.metadata['description']}")
    created_at = datetime.datetime.fromtimestamp(batch.created_at)
    print(f"Created At: {created_at}")
    if batch.error_file_id is not None:
        print(f"Error File ID: {batch.error_file_id}")
        print(f"Error File URL: {batch.error_file_url}")
    if batch.completed_at is None:
        print("Completed At: Still running")
    else:
        completed_at = datetime.datetime.fromtimestamp(batch.completed_at)
        print(f"Completed At: {completed_at}")
    print("##############")