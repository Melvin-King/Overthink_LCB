#added by melvin

from transformers import AutoTokenizer
from ..lm_styles import LanguageModelStore, LMStyle

# cache tokenizer to improve efficiency
_tokenizer_cache = {}

def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    aquire tokenizer based on model names and cache them
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    
    try:
        # select tokenizer based on model name
        if model_name.startswith("qwen") or model_name.startswith("Qwen"):
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        elif model_name.startswith("meta-llama") or model_name.startswith("Llama"):
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", trust_remote_code=True)
        elif model_name.startswith("deepseek") or model_name.startswith("DeepSeek"):
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-33b-instruct", trust_remote_code=True)
        elif model_name.startswith("claude") or model_name.startswith("anthropic"):
            # (to be edited?)Claude usually doesn't support Hugging Face，using other tokenizer to simulate
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")  # use common tokenizer as replacement
        elif model_name.startswith("gemini") or model_name.startswith("google"):
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")  # use Gemma as approximation
        else:
            # select a default tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        _tokenizer_cache[model_name] = tokenizer
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        # go back to default tokenizer
        return AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)

def count_tokens(generation: str, model_name: str) -> int:
    """
    count number of tokens
    """
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(generation, add_special_tokens=False))