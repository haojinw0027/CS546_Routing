"""
Utility functions and configurations for baseline.py
"""

# Valid model name mappings - maps full model name to short name for filenames
VALID_MODELS = {
    "meta-llama/Llama-3.2-3B": "llama3.2_3B",
    "Qwen/Qwen3-8B": "qwen3_8B",
    "Qwen/Qwen3-8B-Base": "qwen3_8B_base",
}


def get_model_short_name(model_name: str) -> str:
    """
    Get short model name for filename usage

    Args:
        model_name: Full model name (e.g., "meta-llama/Llama-3.2-3B")

    Returns:
        Short model name (e.g., "llama3.2_3B") or sanitized original name
    """
    if model_name in VALID_MODELS:
        return VALID_MODELS[model_name]

    # Fallback: sanitize the model name for filename usage
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()


def is_valid_model(model_name: str) -> bool:
    """
    Check if model name is in the valid models list

    Args:
        model_name: Model name to check

    Returns:
        True if model is in VALID_MODELS, False otherwise
    """
    return model_name in VALID_MODELS


def get_available_models() -> list:
    """
    Get list of available model names

    Returns:
        List of valid model names
    """
    return list(VALID_MODELS.keys())