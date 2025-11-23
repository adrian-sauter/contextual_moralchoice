"""PATH CONFIG"""

from pathlib import Path
import os

# Locate root path
ROOT = Path(__file__).parent.parent

# SET PATHS
PATH_API_KEYS = ROOT / "api_keys"
PATH_AZURE_ENDPOINT = ROOT / "azure_endpoints"
PATH_HF_CACHE = f"/scratch-shared/{os.environ['USER']}/huggingface/hub" # ROOT / "cache"
PATH_OFFLOAD = f"/scratch-shared/{os.environ['USER']}/huggingface/hub" # ROOT / "offload"

PATH_RESULTS = ROOT / "data/responses"
PATH_QUESTION_TEMPLATES = ROOT / "data/question_templates"
PATH_RESPONSE_TEMPLATES = ROOT / "data/response_templates"


# set variation mappings
VARIATION_MAPPINGS = {
    "Base": "B",
    "Consequentialist": "C",
    "Emotional": "E",
    "Relational": "R"
}
