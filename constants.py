"""
Application-wide constants and configuration defaults.
"""

# Flask Configuration Defaults
SECRET_KEY_DEFAULT = 'dev-secret-key-change-in-production'
MAX_CONTENT_LENGTH = 10 * 1024 * 1024 * 1024  # 10GB max file size for large datasets
SESSION_TYPE = 'filesystem'
ALLOWED_EXTENSIONS = {'json', 'jsonl', 'csv', 'parquet'}

# CORS Configuration
CORS_RESOURCES = {r"/api/*": {"origins": "*"}}

# SocketIO Configuration
SOCKETIO_CORS_ORIGINS = "*"
SOCKETIO_ASYNC_MODE = 'threading'

# Directory Paths (relative to project root)
UPLOAD_FOLDER = 'uploads'
CONFIGS_FOLDER = 'configs'
LOGS_FOLDER = 'logs'
OUTPUTS_FOLDER = 'outputs'
EXPORTS_FOLDER = 'exports'
CACHE_FOLDER = 'cache'
TEMPLATES_FOLDER = 'templates'
STATIC_FOLDER = 'static'
PRESETS_FOLDER = 'presets'

REQUIRED_DIRECTORIES = [
    CONFIGS_FOLDER,
    LOGS_FOLDER,
    OUTPUTS_FOLDER,
    EXPORTS_FOLDER,
    CACHE_FOLDER,
    TEMPLATES_FOLDER,
    STATIC_FOLDER,
]

# Popular Datasets Catalog
# This catalog matches the frontend dataset selection interface
POPULAR_DATASETS = {
    'tatsu-lab/alpaca': {
        'name': 'Alpaca',
        'size': '52K samples',
        'estimated_mb': 45,
        'sample_count': 52000,
        'category': 'general'
    },
    'openai/gsm8k': {
        'name': 'GSM8K',
        'size': '8.5K problems',
        'estimated_mb': 12,
        'sample_count': 8500,
        'category': 'math',
        'field_mapping': {
            'instruction': 'question',
            'response': 'answer'
        }
    },
    'nvidia/OpenMathReasoning': {
        'name': 'OpenMath Reasoning',
        'size': '100K problems',
        'estimated_mb': 200,
        'sample_count': 100000,
        'category': 'math',
        'default_split': 'cot',
        'field_mapping': {
            'instruction': 'problem',
            'response': 'generated_solution'
        }
    },
    'sahil2801/CodeAlpaca-20k': {
        'name': 'Code Alpaca',
        'size': '20K examples',
        'estimated_mb': 35,
        'sample_count': 20000,
        'category': 'coding',
        'field_mapping': {
            'instruction': 'prompt',
            'response': 'completion'
        }
    },
    'databricks/databricks-dolly-15k': {
        'name': 'Dolly 15k',
        'size': '15K samples',
        'estimated_mb': 30,
        'sample_count': 15000,
        'category': 'general',
        'default_split': 'train',
        'field_mapping': {
            'instruction': 'instruction',
            'response': 'response'
        }
    },
    'microsoft/orca-math-word-problems-200k': {
        'name': 'Orca Math',
        'size': '200K problems',
        'estimated_mb': 350,
        'sample_count': 200000,
        'category': 'math',
        'field_mapping': {
            'instruction': 'question',
            'response': 'answer'
        }
    },
    'squad': {
        'name': 'SQuAD v2',
        'size': '130K questions',
        'estimated_mb': 120,
        'sample_count': 130000,
        'category': 'qa',
        'field_mapping': {
            'instruction': 'question',
            'response': 'answers'
        }
    }
}

# Training Configuration Defaults
DEFAULT_TRAINING_CONFIG = {
    'model_name': 'unsloth/Qwen3-0.6B',
    'lora_rank': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.0,
    'lora_bias': 'none',
    'num_epochs': 3,
    'batch_size': 1,
    'gradient_accumulation_steps': 1,
    'learning_rate': 2e-4,
    'warmup_steps': 10,
    'weight_decay': 0.001,
    'max_grad_norm': 0.3,
    'lr_scheduler_type': 'constant',
    'optim': 'paged_adamw_32bit',
    'logging_steps': 1,
    'save_steps': 100,
    'eval_steps': 100,
    'seed': 42,
    'max_sequence_length': 2048,
    'max_new_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.95,
    'top_k': 50,
    'repetition_penalty': 1.0,
    'loss_type': 'grpo',
    'importance_sampling_level': 'token',
    'kl_penalty': 0.05,
    'clip_range': 0.2,
    'value_coefficient': 1.0,
    'epsilon': 3e-4,
    'epsilon_high': 4e-4,
    'use_4bit': False,
    'use_8bit': False,
    'use_flash_attention': False,
    'gradient_checkpointing': False,
    'fp16': False,
    'bf16': False,
}

# Pre-training Configuration Defaults
DEFAULT_PRE_TRAINING_CONFIG = {
    'enabled': True,
    'epochs': 2,
    'max_samples': None,
    'filter_by_length': False,
    'max_length_ratio': 0.5,
    'learning_rate': 5e-5,
    'adaptive': True,
    'min_success_rate': 0.8,
    'max_additional_epochs': 3,
    'validation_samples': 5,
}

# Dataset Configuration Defaults
DEFAULT_DATASET_CONFIG = {
    'source_type': 'huggingface',
    'split': 'train',
    'instruction_field': 'instruction',
    'response_field': 'output',
    'max_samples': None,
}

# Template Configuration Defaults
DEFAULT_TEMPLATE_CONFIG = {
    'instruction_template': '{instruction}',
    'response_template': '{response}',
    'reasoning_start': '<start_working_out>',
    'reasoning_end': '<end_working_out>',
    'solution_start': '<SOLUTION>',
    'solution_end': '</SOLUTION>',
    'system_prompt': (
        'You are given a problem.\n'
        'Think about the problem and provide your working out.\n'
        'Place it between <start_working_out> and <end_working_out>.\n'
        'Then, provide your solution between <SOLUTION></SOLUTION>'
    ),
    'chat_template_type': 'grpo',
    'prepend_reasoning_start': True,
}

# Reward Configuration Defaults
DEFAULT_REWARD_CONFIG = {
    'type': 'preset',
    'preset_name': 'math',
}

# Model Configuration
# Comprehensive model definitions with metadata
MODEL_DEFINITIONS = {
    'qwen': [
        {
            'id': 'unsloth/Qwen3-0.6B',
            'name': 'Qwen3 0.6B',
            'size': '600M',
            'vram': '~1.2GB',
            'category': 'qwen'
        },
        {
            'id': 'unsloth/Qwen3-1.7B',
            'name': 'Qwen3 1.7B',
            'size': '1.7B',
            'vram': '~3.4GB',
            'category': 'qwen'
        },
        {
            'id': 'unsloth/Qwen3-4B',
            'name': 'Qwen3 4B',
            'size': '4B',
            'vram': '~8GB',
            'category': 'qwen'
        },
        {
            'id': 'unsloth/Qwen3-8B',
            'name': 'Qwen3 8B',
            'size': '8B',
            'vram': '~16GB',
            'category': 'qwen'
        }
    ],
    'llama': [
        {
            'id': 'unsloth/Llama-3.2-1B-Instruct',
            'name': 'LLaMA 3.2 1B',
            'size': '1B',
            'vram': '~2GB',
            'category': 'llama'
        },
        {
            'id': 'unsloth/Llama-3.2-3B-Instruct',
            'name': 'LLaMA 3.2 3B',
            'size': '3B',
            'vram': '~6GB',
            'category': 'llama'
        }
    ],
    'phi': [
        {
            'id': 'unsloth/phi-4-reasoning',
            'name': 'Phi-4 Reasoning',
            'size': '15B',
            'vram': '~30GB',
            'category': 'phi'
        }
    ]
}

# Model size estimates (in MB) - used for memory planning and optimization
MODEL_SIZES = {
    "unsloth/phi-4-reasoning": 30720,
    "unsloth/Qwen3-0.6B": 1229,
    "unsloth/Qwen3-1.7B": 3482,
    "unsloth/Qwen3-4B": 8192,
    "unsloth/Qwen3-8B": 16384,
    "unsloth/Llama-3.2-1B-Instruct": 2048,
    "unsloth/Llama-3.2-3B-Instruct": 6144,
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit": 4096,
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit": 1024,
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit": 4096
}
