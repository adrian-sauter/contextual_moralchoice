import anthropic

with open('/home/asauter1/contextual_moralchoice/api_keys/anthropic_key.txt', encoding="utf-8") as f:
    key = f.read()

client = anthropic.Anthropic(api_key=key)

# Automatically fetches more pages as needed.
for message_batch in client.messages.batches.list(
    limit=20
):
    print(message_batch)

for result in client.messages.batches.results(
    "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
):
    print(result)