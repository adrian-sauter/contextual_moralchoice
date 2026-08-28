"""PATH CONFIG

Values a new user needs to set before running anything:
- Place API key files under `api_keys/{company}_key.txt` (e.g. `api_keys/openai_key.txt`),
  matching the company identifiers used in `src/models.py`.
- Place Azure endpoint files under `azure_endpoints/{company}_azure_endpoint.txt` if using
  an Azure-hosted model.
- Optionally set the `MORALCHOICE_HF_CACHE` / `MORALCHOICE_OFFLOAD` environment variables to
  point Hugging Face model downloads/offloading at a scratch/shared filesystem (e.g. on a
  cluster). Defaults to local `cache/` / `offload/` directories under the repo root.
"""

from pathlib import Path
import os

# Locate root path
ROOT = Path(__file__).parent.parent

# SET PATHS
PATH_API_KEYS = ROOT / "api_keys"
PATH_AZURE_ENDPOINT = ROOT / "azure_endpoints"
PATH_HF_CACHE = os.environ.get("MORALCHOICE_HF_CACHE", str(ROOT / "cache"))
PATH_OFFLOAD = os.environ.get("MORALCHOICE_OFFLOAD", str(ROOT / "offload"))

PATH_RESULTS = ROOT / "data/responses"
PATH_QUESTION_TEMPLATES = ROOT / "data/question_templates"
PATH_RESPONSE_TEMPLATES = ROOT / "data/response_templates"

# Intermediate/aggregated analysis data consumed by scripts in evaluate/ and
# evaluate/ALL_FIGURES.ipynb
PATH_ANALYSIS = ROOT / "evaluate/data"


# set variation mappings
VARIATION_MAPPINGS = {
    "Base": "B",
    "Consequentialist": "C",
    "Emotional": "E",
    "Relational": "R"
}
