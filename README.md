# On the Context Sensitivity of LLM Moral Judgment


**Authors:** [Adrian Sauter](https://adrian-sauter.github.io/) and [Mona Schirmer](https://monasch.github.io/)  

**Paper:** [On the Context Sensitivity of LLM Moral Judgment](https://arxiv.org/abs/2603.23114) (EMNLP 2026).  

```bibtex
@article{sauter2026contextual,
  title={On the Context Sensitivity of LLM Moral Judgment},
  author={Sauter, Adrian and Schirmer, Mona},
  journal={arXiv preprint arXiv:2603.23114},
  year={2026}
}
```

This repository builds on the [MoralChoice](https://github.com/ninodimontalcino/moralchoice) benchmark
([Scherrer et al., NeurIPS 2023](https://arxiv.org/abs/2307.14324)) to study whether LLMs' moral
"choices" hold up under contextual reframing. Each MoralChoice scenario states a rule (e.g. "do not
break the law") and two candidate actions. We rewrite the base scenario into three contextual
variations — **Consequentialist**, **Emotional**, **Relational** — that change only the surrounding
context while keeping the rule and the two candidate actions fixed, and measure how much a model's
preference shifts (the **Contextual Preference Shift**, CPS). We evaluate 22 models and additionally
use contrastive activation steering to reproduce or reverse the observed shifts.

## Pipeline overview

1. **Collect model responses** — `src/evaluate.py` queries a model over every scenario/variation/
   question format and writes raw per-scenario pickles.
2. **Aggregate** — `src/collect.py` merges those pickles into one CSV per model, optionally refining
   `refusal`/`invalid` decisions with an LLM judge (`--refine`).
3. **Compute metrics** — the scripts in `evaluate/` turn the aggregated response CSVs into the
   intermediate CSVs (CPS statistics, flip rate/boundary mass, refusal rates, steering CPS, ...)
   under `evaluate/data/` (see `PATH_ANALYSIS` in `src/config.py`).
4. **Plot** — `evaluate/ALL_FIGURES.ipynb` loads those intermediate CSVs and produces every figure
   in the paper.

`reproduce_figures.ipynb` (repo root) walks through steps 3-4 on the response data already in this
repo and plots the result. `evaluate/ALL_FIGURES.ipynb` has the full set of figures, including a
few that need raw experiment data not included here (noted inline).

## Installation

```bash
python -m pip install -r requirements.txt
```

## Configuration

In order to use the code, please set the paths in `src/config.py` or in the following directories:

- **API keys**: add a `<company>_key.txt` file per provider under `api_keys/` (e.g.
  `api_keys/openai_key.txt`, `api_keys/anthropic_key.txt`), matching the `company` field used in
  `src/models.py`'s `MODELS` registry.
- **Hugging Face cache/offload** (only relevant for locally-run models): by default, downloaded
  weights and offloaded layers go to `cache/` and `offload/` under the repo root. Override with the
  `MORALCHOICE_HF_CACHE` / `MORALCHOICE_OFFLOAD` environment variables (e.g. to point at a cluster's
  shared scratch filesystem).

## Running the pipeline

```bash
# 1. Query a model over the scenario set
python -m src.evaluate \
    --experiment-name "my_experiment" \
    --dataset "high" \
    --model-name "openai/gpt-4.1" \
    --question-types "ab" "compare" "repeat" \
    --eval-nb-samples 10

# 2. Aggregate into one CSV per model, optionally refining refusal/invalid decisions with an LLM
#    judge (this resolves 'refusal'/'invalid' rows in place -- there's only ever one `decision`
#    column per file, never a separate refined copy)
python -m src.collect \
    --experiment-name "my_experiment" \
    --dataset "high" \
    --refine --refiner-model "openai/gpt-4o-mini"

# 3. Compute the master marginal-action-likelihoods / CPS table from the aggregated CSVs
python -m evaluate.compute_marginal_action_likelihoods \
    --responses-dir "data/responses/my_experiment/high"

# 4. Compute downstream statistics (see `evaluate/` for the full set of scripts)
python -m evaluate.compute_cps_statistics
python -m evaluate.compute_flip_boundary_mass
python -m evaluate.compute_refusal_invalid_stats --responses-dir "data/responses/my_experiment/high"
```

Each `evaluate/*.py` script's docstring names the figure it feeds; run any of them with `--help`
for its exact arguments and defaults. Or open `reproduce_figures.ipynb` at the repo root, which
runs steps 3-4 end to end and plots the result.

## Data layout

- `data/scenarios/` — MoralChoice scenario/variation definitions.
- `data/question_templates/`, `data/response_templates/` — question phrasings and the common
  answer patterns used for heuristic decision-matching.
- `data/responses/` — one subdirectory per experiment; `data/responses/paper/` holds the original
  2023 MoralChoice paper's 28-model reproduction data, kept for reference.
- `evaluate/data/` — the intermediate CSVs `evaluate/*.py` scripts write to
  (`marginal_action_likelihoods.csv`, `cps_statistics.csv`, ...) and the artifacts that can't be
  regenerated from data in this repo (`ab_vector_steering_dfs/`, `layer_accuracies_by_variation.pdf`),
  all consumed by `evaluate/ALL_FIGURES.ipynb`.

## Evaluation methodology

Not every model in the paper was queried the same way. Some were evaluated through a provider's
async Batch API; open-weight models were run directly (optionally with `src.evaluate --batching` for
throughput) on a compute cluster:

| Method | Models | Data location |
|---|---|---|
| Provider batch API (OpenAI) | gpt-4.1, gpt-4.1-mini, gpt-4o-mini, gpt-5.1 | `data/responses/openai_models/` |
| Provider batch API (Anthropic) | claude-3-haiku, claude-haiku-4-5, claude-sonnet-4-5 | `data/responses/anthropic_models/` |
| Provider batch API (Together-hosted) | DeepSeek-V3, DeepSeek-V3.1 | `data/responses/together_models/` |
| Direct run on cluster (`src.evaluate`) | Llama, Mistral, Qwen, DeepSeek (local) models | `data/responses/{llama,mistral,qwen,deepseek}_models/` |
| Original MoralChoice (2023) reproduction | 28 original models (AI21, Cohere, GPT-3/3.5/4, Claude v1/v2, PaLM 2, OPT-IML, Bloomz, ...) | `data/responses/paper/{orig_high,orig_low}/` |

## Models supported

Model identifiers are registered in the `MODELS` dict in `src/models.py`. Current roster:

- **OpenAI** (API): `openai/gpt-4`, `openai/gpt-4.1`, `openai/gpt-4.1-mini`, `openai/gpt-4o-mini`,
  `openai/gpt-5.1`, `openai/gpt-3.5-turbo`, plus the original GPT-3 family
  (`text-ada/babbage/curie/davinci-001`, `text-davinci-002/003`). (The current Claude 3/4.5 models
  and Together-hosted DeepSeek-V3 used in the paper were queried directly through their provider's
  batch API rather than through this registry — see
  [Evaluation methodology](#evaluation-methodology).)
- **Anthropic** (API): `anthropic/claude-v1.0/1.2/1.3`, `anthropic/claude-v2.0`,
  `anthropic/claude-instant-v1.0/1.1`.
- **Meta / Llama** (HF Hub): `meta/llama-2-7b-chat`, `meta/llama-3-8b-instruct`,
  `meta/llama-3.1-8b-instruct`, `meta/llama-3.1-70b-instruct`; plus the original
  `meta/opt-iml-{regular,max}-{small,large}`.
- **Mistral** (HF Hub): `mistral/mistral-7b-instruct-v0.1`, `mistral/mixtral-8x7b-instruct_8bit`,
  `huggingfaceh4/zephyr-7b-beta`, `teknium/openhermes-2.5-mistral-7b`.
- **Qwen** (HF Hub): `qwen/qwen1.5-7b-chat`, `qwen/qwen2-7b-instruct`, `qwen/qwen3-4b-instruct`,
  `qwen/qwen3-8b`.
- **DeepSeek**: `deepseek/deepseek-llm-7b-chat` (HF Hub), `deepseek/deepseek-V3.1` (API).
- **AI21** (API): `ai21/j2-grande-instruct`, `ai21/j2-jumbo-instruct`.
- **Cohere** (API): `cohere/command-xlarge`, `cohere/command-medium`.
- **Google** (API/HF Hub): `google/text-bison-001` (PaLM 2), `google/t5-{small,base,large,xl,xxl}`,
  `google/flan-t5-{small,base,large,xl,xxl}` (optionally `-8bit`).
- **BigScience** (HF Hub): `bigscience/bloomz-{560m,1b1,1b7,3b,7b1,7b1-mt,7b1-p3}`.

### Adding new models
- For most API-based models from a supported provider, add an entry to the `MODELS` dict in
  `src/models.py`.
- Otherwise, add a new model handler class to `src/models.py` and register it in `MODELS`.

## Adding more question templates

- Add `<question_name>.json` to `data/question_templates/` with the question template.
- Add `<question_name>.json` to `data/response_templates/` with common answer patterns (generate a
  set of answers first and cluster common prefixes to find these).
- The current evaluation workflow only supports binary decision settings, though the pipeline could
  be extended to more options if needed.

## Semantic matching

Mapping raw model text to `action1`/`action2`/`refusal`/`invalid` happens in two stages
(`src/semantic_matching.py`):

1. **Heuristic matching** (`token_to_action_matching`) — exact/stemmed string matching against known
   answer patterns and refusal phrases, run automatically inside `src.evaluate`. This resolves the
   large majority of responses.
2. **Optional LLM-based refinement** (`refine_results_with_llm`, exposed via `src.collect --refine`)
   — an LLM judge re-classifies whatever the heuristic left as `refusal`/`invalid`, to recover
   responses with an unambiguous preference that didn't match a known pattern. This updates the
   single `decision` column in place — there is never a separate `refined_decision` column or a
   sibling `*_REFINED.csv` file.

## Activation steering

`src/extract_activations.py`, `src/generate_vector.py`, and `src/steer_model.py` implement a separate
experiment: extracting residual-stream activations, building a contrastive steering vector between a
base scenario and its contextual variation, and applying it (at a chosen layer and strength) to
reproduce or reverse a model's contextual preference shift. `evaluate/analyze_layers.py` selects the
best layer per variation via cross-validated logistic-regression probes over those activations.

## License

MIT — see `LICENSE`.
