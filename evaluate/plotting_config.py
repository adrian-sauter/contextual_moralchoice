"""Display metadata (colors, markers, names) used by evaluate/ALL_FIGURES.ipynb."""

VARIATION_MAPPINGS = {
    "Consequentialist": {
        'display_color': "#00AEB3"
    },
    "Emotional": {
        'display_color': "#F58137"
    },
    "Relational": {
        'display_color': "#ED017D"
    }
}

MODEL_MAPPINGS = {
    ### LLAMA MODELS (Muted Blues) ###
    'meta_llama-2-7b-chat': {
        'display_name': 'Llama-2-7B-Chat',
        'company': 'Meta',
        'base_company': 'Meta',
        'size': '7B',
        'display_color': "#2557A7",
        'marker': "v",
    },
    'meta_llama-3-8B-instruct': {
        'display_name': 'Llama-3-8B-Instruct',
        'company': 'Meta',
        'base_company': 'Meta',
        'size': '8B',
        'display_color': "#4A7BC7",
        'marker': '<',
    },
    'meta_llama-3.1-8B-instruct': {
        'display_name': 'Llama-3.1-8B-Instruct',
        'company': 'Meta',
        'base_company': 'Meta',
        'size': '8B',
        'display_color': "#7BA2E0",
        'marker': '^',
    },
    'meta_llama-3.1-70b-instruct': {
        'display_name': 'Llama-3.1-70B-Instruct',
        'company': 'Meta',
        'base_company': 'Meta',
        'size': '70B',
        'display_color': "#A9C4F0",
        'marker': '>',
    },

    ### MISTRAL MODELS (Muted Teal-Green) ###
    'mistral_mixtral-8x7b-instruct_8bit': {
        'display_name': 'Mixtral-8x7B-Instruct-v0.1',
        'company': 'Mistral',
        'base_company': 'Mistral',
        'size': '8x7B',
        'display_color': "#9C7B00",
        'marker': 'v',
    },
    'mistral_mistral-7b-instruct-v0.1': {
        'display_name': 'Mistral-7B-Instruct-v0.1',
        'company': 'Mistral',
        'base_company': 'Mistral',
        'size': '7B',
        'display_color': "#B38F00",
        'marker': '<',
    },
    'huggingfaceh4_zephyr-7b-beta': {
        'display_name': 'Zephyr-7B-Beta',
        'company': 'HuggingFace H4',
        'base_company': 'Mistral',
        'size': '7B',
        'display_color': "#D9AD26",
        'marker': '^',
    },
    'teknium_openhermes-2.5-mistral-7b': {
        'display_name': 'OpenHermes-2.5-Mistral-7B',
        'company': 'Teknium',
        'base_company': 'Mistral',
        'size': '7B',
        'display_color': "#ECC263",
        'marker': '>',
    },

    ### QWEN MODELS (Muted Purple) ###
    'qwen_qwen1.5-7b-chat': {
        'display_name': 'Qwen1.5-7B-Chat',
        'company': 'Qwen',
        'base_company': 'Qwen',
        'size': '7B',
        'display_color': "#6A49AD",
        'marker': 'v',
    },
    'qwen_qwen2-7b-instruct': {
        'display_name': 'Qwen2-7B-Instruct',
        'company': 'Qwen',
        'base_company': 'Qwen',
        'size': '7B',
        'display_color': "#8A6BC2",
        'marker': '<',
    },
    'qwen_qwen3-4b-instruct': {
        'display_name': 'Qwen3-4B-Instruct',
        'company': 'Qwen',
        'base_company': 'Qwen',
        'size': '4B',
        'display_color': "#A88FD8",
        'marker': '^',
    },
    'qwen_qwen3-8b': {
        'display_name': 'Qwen3-8B',
        'company': 'Qwen',
        'base_company': 'Qwen',
        'size': '8B',
        'display_color': "#C8B6ED",
        'marker': '>',
    },

    ### DEEPSEEK MODELS (Muted Red-Orange) ###
    'deepseek_deepseek-llm-7b-chat': {
        'display_name': 'DeepSeek-LLM-7B-Chat',
        'company': 'DeepSeek',
        'base_company': 'DeepSeek',
        'size': '7B',
        'display_color': "#3E5266",
        'marker': 'v',
    },
    'deepseek-ai_DeepSeek-V3': {
        'display_name': 'DeepSeek-V3',
        'company': 'DeepSeek',
        'base_company': 'DeepSeek',
        'size': '671B',
        'display_color': "#5C7085",
        'marker': '<',
    },
    'deepseek-ai_DeepSeek-V3.1': {
        'display_name': 'DeepSeek-V3.1',
        'company': 'DeepSeek',
        'base_company': 'DeepSeek',
        'size': '671B',
        'display_color': "#7C8FA3",
        'marker': '^',
    },

    ### ANTHROPIC MODELS (Muted Grape-Red) ###
    'claude_claude-3-haiku-20240307': {
        'display_name': 'Claude-3-Haiku',
        'company': 'Anthropic',
        'base_company': 'Anthropic',
        'size': 'N/A',
        'display_color': "#3D6B4A",
        'marker': 'o',
    },
    'claude_claude-haiku-4-5-20251001': {
        'display_name': 'Claude-Haiku-4.5',
        'company': 'Anthropic',
        'base_company': 'Anthropic',
        'size': 'N/A',
        'display_color': "#5E8C6B",
        'marker': 'p',
    },
    'claude_claude-sonnet-4-5-20250929': {
        'display_name': 'Claude-Sonnet-4.5',
        'company': 'Anthropic',
        'base_company': 'Anthropic',
        'size': 'N/A',
        'display_color': "#82AE8E",
        'marker': 's',
    },

    ### OPENAI MODELS (Muted Cyan-Blue) ###
    'openai_gpt-4o-mini': {
        'display_name': 'GPT-4o-Mini',
        'company': 'OpenAI',
        'base_company': 'OpenAI',
        'size': 'N/A',
        'display_color': "#7D3535",
        'marker': 'o',
    },
    'openai_gpt-4.1': {
        'display_name': 'GPT-4.1',
        'company': 'OpenAI',
        'base_company': 'OpenAI',
        'size': 'N/A',
        'display_color': "#913D3D",
        'marker': 'p',
    },
    'openai_gpt-4.1-mini': {
        'display_name': 'GPT-4.1-Mini',
        'company': 'OpenAI',
        'base_company': 'OpenAI',
        'size': 'N/A',
        'display_color': "#B4655F",
        'marker': 's',
    },
    'openai_gpt-5.1': {
        'display_name': 'GPT-5.1',
        'company': 'OpenAI',
        'base_company': 'OpenAI',
        'size': 'N/A',
        'display_color': "#D29891",
        'marker': 'h',
    }
}
