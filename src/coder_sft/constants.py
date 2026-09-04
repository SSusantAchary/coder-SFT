"""Shared constants for training and normalized data."""

MODEL_NAME = "unsloth/Qwen3.5-2B"
SYSTEM_PROMPT = (
    "You are a precise software engineering assistant. Prefer minimal, "
    "correct, verifiable changes and preserve existing interfaces."
)

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PROJECTION_FAMILIES = {
    "full_attention": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "linear_attention": {
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    },
    "mlp": {"gate_proj", "up_proj", "down_proj"},
}

SUPPORTED_LANGUAGES = (
    "python",
    "javascript",
    "typescript",
    "java",
    "cpp",
    "go",
    "rust",
    "shell",
    "sql",
)

LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "bash": "shell",
    "sh": "shell",
}

EXTENSIONS = {
    "python": {".py"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx"},
    "java": {".java"},
    "cpp": {".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"},
    "go": {".go"},
    "rust": {".rs"},
    "shell": {".sh", ".bash"},
    "sql": {".sql"},
}

