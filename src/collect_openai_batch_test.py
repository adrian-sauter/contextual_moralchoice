import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import time

from src.semantic_matching import token_to_action_matching
from src.question_form_generator import get_question_form
from src.config import PATH_RESULTS, PATH_RESPONSE_TEMPLATES, VARIATION_MAPPINGS

################################################################################################
# CONFIGURATION
################################################################################################
DATASET = "high"
MODEL_ID = "gpt-4o-mini"
EXPERIMENT_NAME = "openai_models"
RESULT_DIR = f"{PATH_RESULTS}/{EXPERIMENT_NAME}/{DATASET}_raw/{MODEL_ID}"
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_FILES = [
    "/home/asauter1/contextual_moralchoice/data/scenarios/test_scenarios_results_gpt-4o-mini.jsonl"
]

# Load scenarios
scenarios = pd.read_csv(f"data/scenarios/variations_moralchoice_{DATASET}_ambiguity.csv")

# Load refusals and answer patterns
with open(f"{PATH_RESPONSE_TEMPLATES}/refusals.txt", encoding="utf-8") as f:
    refusals = f.read().splitlines()

response_patterns = {}
for fname in os.listdir(PATH_RESPONSE_TEMPLATES):
    if fname.endswith(".json"):
        qtype = fname.replace(".json", "")
        with open(os.path.join(PATH_RESPONSE_TEMPLATES, fname), encoding="utf-8") as f:
            response_patterns[qtype] = json.load(f)

################################################################################################
# HELPER
################################################################################################
def parse_custom_id(cid):
    """
    Expected format:
      {scenario_id}_{variation}_{question_type}_{question_ordering}_{eval_sample_nb}
    """
    "G_269_Relational_compare_1_6"
    parts = cid.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected custom_id format: {cid}")
    scenario_id = parts[0] + "_" + parts[1]
    variation = parts[2]
    question_type = parts[3]
    question_ordering = int(parts[4])
    eval_sample_nb = int(parts[5])
    return scenario_id, variation, question_type, question_ordering, eval_sample_nb


################################################################################################
# MAIN
################################################################################################
print("=== Collecting batch results ===")
all_results = []

for result_path in RESULT_FILES:
    with open(result_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            cid = data["custom_id"]

            # --- extract text safely from new Batch API structure ---
            output = ""
            try:
                output = (
                    data["response"]["body"]["choices"][0]["message"]["content"]
                ).strip()
            except KeyError:
                # fallback (older format)
                output = data.get("response", {}).get("output_text", "")
            except Exception as e:
                print(f"Could not parse response for {cid}: {e}")
                continue
            #if not output:
            #    continue

            # parse identifying info
            scenario_id, variation, qtype, ordering, nb_query = parse_custom_id(cid)

            # match to correct scenario row (ID + variation)
            scenario = scenarios[
                (scenarios["scenario_id"] == scenario_id)
                & (scenarios["variation"] == variation)
            ].iloc[0]


            # regenerate question form & mapping for this configuration
            question_form, action_mapping = get_question_form(
                scenario=scenario,
                question_type=qtype,
                question_ordering=ordering,
                system_instruction=True,
            )

            result_base = {
                "scenario_id": scenario_id,
                "variation": variation,
                "model_id": f"openai/{MODEL_ID}",
                "question_type": qtype,
                "question_ordering": ordering,
                "question_header": question_form["question_header"],
                "question_text": question_form["question"],
                "eval_technique": "top_p_sampling",
                "eval_top_p": 1.0,
                "eval_temperature": 1.0,
                "eval_sample_nb": nb_query,
                "timestamp": None,
                "answer_raw": output,
                "answer": output,
            }

            result_base["decision"] = token_to_action_matching(
                output,
                scenario,
                response_patterns,
                qtype,
                action_mapping,
                refusals,
            )

            all_results.append(result_base)

################################################################################################
# EXPORT
################################################################################################
df = pd.DataFrame(all_results)
print(f"Collected {len(df)} results")

# Save separate pickles per question_type (to match your old format)
for qtype, group_df in df.groupby("question_type"):
    outdir = os.path.join(RESULT_DIR, qtype)
    os.makedirs(outdir, exist_ok=True)
    for (sid, variation), sdf in group_df.groupby(["scenario_id", "variation"]):
        path_out = f"{outdir}/scenario_{sid}_{VARIATION_MAPPINGS[variation]}.pickle"
        with open(path_out, "wb") as f:
            pickle.dump(sdf, f, protocol=0)
print("✅ Results stored in the same per-scenario format.")