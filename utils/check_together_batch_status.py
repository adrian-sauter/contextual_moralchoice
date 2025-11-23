import os
from together import Together
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

with open('/home/asauter1/contextual_moralchoice/api_keys/together_key.txt', encoding="utf-8") as f:
    key = f.read()

client = Together(api_key=key)

batch_ids = []
for file in os.listdir("/home/asauter1/contextual_moralchoice/data/responses/together_models/"):
    if file.startswith("batch_id_") and file.endswith(".txt"):
        with open(os.path.join("/home/asauter1/contextual_moralchoice/data/responses/together_models/", file), "r") as idfile:
            batch_id = idfile.read().strip()
            batch_ids.append(batch_id)
    
batches = [client.batches.get_batch(batch_id) for batch_id in batch_ids]
for batch in batches:
    print(f"ID: {batch.id}")
    print(f"Status: {batch.status}")
    #print(f"Model: {batch.model}")
    created_at = batch.created_at
    print(f"Created At: {created_at}")
    if batch.error_file_id is not None:
        print(f"Error File ID: {batch.error_file_id}")
        print(f"Error File URL: {batch.error_file_url}")
    if batch.completed_at is None:
        print("Completed At: Still running")
    else:
        completed_at = batch.completed_at
        print(f"Completed At: {completed_at}")
    print("##############")