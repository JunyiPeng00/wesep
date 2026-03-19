"""Convert Hugging Face SSL models to WeSpeaker format."""

import os
import warnings
from typing import Literal, Tuple, Dict, Any, Union, Optional

import torch
from transformers import (
    WavLMModel, 
    HubertModel, 
    Wav2Vec2Model,
    AutoModel,
    AutoConfig
)

from wesep.modules.wavlm_frontend.model import wav2vec2_model, wavlm_model
from wesep.modules.wavlm_frontend.utils.import_huggingface_wavlm import import_huggingface_model

# Check if we're in a distributed environment
def is_rank_zero():
    """Check if current process is rank 0 in distributed training."""
    # First check environment variable (for manual testing)
    rank_env = os.environ.get('RANK')
    if rank_env is not None:
        return int(rank_env) == 0
    
    # Then check distributed training
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True  # If not distributed, always log
    return torch.distributed.get_rank() == 0


# Extended model mapping for various SSL models
MODEL_ID_MAP: Dict[str, str] = {
    # WavLM models
    "wavlm_base": "microsoft/wavlm-base",
    "wavlm_base_plus": "microsoft/wavlm-base-plus", 
    "wavlm_large": "microsoft/wavlm-large",
    
    # HuBERT models
    "hubert_base": "facebook/hubert-base-ls960",
    "hubert_large": "facebook/hubert-large-ls960",
    "hubert_xlarge": "facebook/hubert-xlarge-ls960",
    
    # Wav2Vec2 models
    "wav2vec2_base": "facebook/wav2vec2-base-960h",
    "wav2vec2_large": "facebook/wav2vec2-large-960h",
    "wav2vec2_large_lv60k": "facebook/wav2vec2-large-lv60",
    "wav2vec2_xlsr_53": "facebook/wav2vec2-xlsr-53",
    "wav2vec2_xlsr_128": "facebook/wav2vec2-xlsr-128",
    
}


def convert_hf_ssl_model(
    model_name: str,
    exp_dir: str,
    hf_cache_dir: str,
    local_files_only: bool = False,
    enable_pruning: bool = False,
    pruning_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Download and convert a Hugging Face SSL model to WeSpeaker format.

    Args:
        model_name: Model name from MODEL_ID_MAP (e.g., 'hubert_base', 'wav2vec2_large').
        exp_dir: Output directory where the converted checkpoint will be saved.
        hf_cache_dir: Local Hugging Face cache directory (will be created if missing).
        local_files_only: If True, load only from local cache without downloading.
        enable_pruning: Whether to enable pruning support in the converted model.
        pruning_config: Configuration for pruning (if enabled).

    Returns:
        A tuple of:
            - Path to the saved WeSpeaker checkpoint (.pth)
            - The model configuration dictionary
    """
    if model_name not in MODEL_ID_MAP:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_ID_MAP.keys())}")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(hf_cache_dir, exist_ok=True)

    model_id = MODEL_ID_MAP[model_name]
    out_path = os.path.join(exp_dir, f"{model_name}.hf.pth")
    
    # 1) Load (or download) from Hugging Face
    if is_rank_zero():
        print(f"Loading model: {model_id}")
    
    # Suppress warnings on non-rank-0 processes
    if not is_rank_zero():
        warnings.filterwarnings("ignore")
    
    # original = AutoModel.from_pretrained(
    #     model_id,
    #     cache_dir=hf_cache_dir,
    #     local_files_only=local_files_only,
    # )
    if "wavlm" in model_id.lower():
        original = WavLMModel.from_pretrained(
            model_id,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
    else: # "hubert" in model_id.lower():
        original = AutoModel.from_pretrained(
             model_id,
             cache_dir=hf_cache_dir,
             local_files_only=local_files_only,
         )
    
    # Restore warnings after model loading
    if not is_rank_zero():
        warnings.resetwarnings()

    # 2) Convert to WeSpeaker structure
    imported, config = import_huggingface_model(original)
    imported.eval()


    # 4) Save the converted checkpoint
    # Check if file already exists and is valid
    if os.path.exists(out_path):
        try:
            test_ckpt = torch.load(out_path, map_location="cpu")
            if "state_dict" in test_ckpt and "config" in test_ckpt:
                if is_rank_zero():
                    print(f"Model {model_name} already converted and valid, skipping conversion.")
                return out_path, config
        except Exception:
            # File exists but is corrupted, need to recreate
            if is_rank_zero():
                print(f"Model {model_name} file exists but corrupted, recreating...")
    
    # Save the model
    if is_rank_zero():
        print(f"Converting model {model_name}...")
    
    torch.save({"state_dict": imported.state_dict(), "config": config}, out_path)

    # 5) Verify by loading into WeSpeaker's wav2vec2 model
    ckpt = torch.load(out_path, map_location="cpu")
    
    # Ensure aux_num_out is present
    model_config = ckpt["config"].copy()
    if 'aux_num_out' not in model_config:
        model_config['aux_num_out'] = None
    
    # Determine which model class to use based on the original model type
    if "wavlm" in model_name.lower():
        model = wavlm_model(**model_config)
    else:
        model = wav2vec2_model(**model_config)
    
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if is_rank_zero():
        print(f"[{model_name}] Missing keys: {len(missing)}")
        print(f"[{model_name}] Unexpected keys: {len(unexpected)}")

    return out_path, config


def get_model_output_size(model_name: str) -> int:
    """Get the output size for a given model name."""
    if "large" in model_name or "xlsr" in model_name or "xlarge" in model_name:
        return 1024
    elif "base" in model_name:
        return 768
    elif model_name in ("xls_r_300m", "xls_r_1b"):
        return 1024
    elif model_name == "xls_r_2b":
        return 1920
    else:
        # Default fallback - try to determine from model config
        try:
            model_id = MODEL_ID_MAP[model_name]
            config = AutoConfig.from_pretrained(model_id)
            return config.hidden_size
        except:
            return 768  # Default fallback


if __name__ == "__main__":
    HF_CACHE = "./hf_models/"
    EXP_DIR = "./hf_models_convert/"

    # Test conversion for different model types
    test_models = ["hubert_base", "wav2vec2_base", "wavlm_base"]
    
    for model_name in test_models:
        try:
            print(f"\n=== Converting {model_name} ===")
            convert_hf_ssl_model(
                model_name=model_name,
                exp_dir=EXP_DIR,
                hf_cache_dir=HF_CACHE,
                local_files_only=False,
            )
            print(f"Successfully converted {model_name}")
        except Exception as e:
            print(f"Failed to convert {model_name}: {e}")
