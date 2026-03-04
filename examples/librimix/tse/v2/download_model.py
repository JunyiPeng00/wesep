#!/usr/bin/env python3
"""Download and convert models from Hugging Face without GPU."""

import os
import sys
import torch
import warnings

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from wespeaker.frontend.wav2vec2.convert_hf_ssl_models import convert_hf_ssl_model

def download_models():
    """Download and convert models from Hugging Face."""
    
    print("=" * 60)
    print("MODEL DOWNLOAD AND CONVERSION (CPU ONLY)")
    print("=" * 60)
    
    # Set device to CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Models to download
    models_to_download = [
        # 'wavlm_base',
        # 'hubert_base',
        'wavlm_base_plus',
        'wavlm_large',
    ]
    
    # Create convert directory
    convert_dir = "convert"
    os.makedirs(convert_dir, exist_ok=True)
    print(f"Convert directory: {convert_dir}")
    
    success_count = 0
    
    for model_name in models_to_download:
        print(f"\n{'='*40}")
        print(f"Downloading and converting: {model_name}")
        print(f"{'='*40}")
        
        try:
            # Convert model
            converted_path, config = convert_hf_ssl_model(
                model_name=model_name,
                exp_dir=convert_dir,
                hf_cache_dir=convert_dir,
                local_files_only=False,  # Allow download from Hugging Face
            )
            
            print(f"✓ Successfully converted {model_name}")
            print(f"  Output path: {converted_path}")
            print(f"  Config keys: {list(config.keys()) if config else 'None'}")
            
            # Verify the converted model
            try:
                ckpt = torch.load(converted_path, map_location='cpu')
                print(f"  ✓ Model verification successful")
                print(f"  ✓ State dict keys: {len(ckpt.get('state_dict', {}))}")
                print(f"  ✓ Config keys: {len(ckpt.get('config', {}))}")
                success_count += 1
            except Exception as e:
                print(f"  ✗ Model verification failed: {e}")
                
        except Exception as e:
            print(f"✗ Failed to convert {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully converted: {success_count}/{len(models_to_download)} models")
    
    if success_count == len(models_to_download):
        print("✓ All models downloaded and converted successfully!")
        print("\nYou can now run GPU training without EOFError issues.")
    else:
        print("⚠ Some models failed to convert. Check the errors above.")
    
    return success_count == len(models_to_download)

def verify_models():
    """Verify that all required models are available and valid."""
    
    print("\n" + "="*60)
    print("MODEL VERIFICATION")
    print("="*60)
    
    convert_dir = "convert"
    required_models = [
        "wavlm_base_plus.hf.pth",
        'wavlm_large.hf.pth',
        # "hubert_base.hf.pth"
    ]
    
    all_valid = True
    
    for model_file in required_models:
        model_path = os.path.join(convert_dir, model_file)
        
        if not os.path.exists(model_path):
            print(f"✗ Missing: {model_file}")
            all_valid = False
            continue
            
        try:
            ckpt = torch.load(model_path, map_location='cpu')
            if 'state_dict' in ckpt and 'config' in ckpt:
                print(f"✓ Valid: {model_file}")
            else:
                print(f"✗ Invalid structure: {model_file}")
                all_valid = False
        except Exception as e:
            print(f"✗ Corrupted: {model_file} - {e}")
            all_valid = False
    
    if all_valid:
        print("\n✓ All models are valid and ready for training!")
    else:
        print("\n⚠ Some models are missing or corrupted. Run download_models() to fix.")
    
    return all_valid

def main():
    """Main function."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_models()
    else:
        # Download and convert models
        success = download_models()
        
        if success:
            # Verify after download
            verify_models()
        
        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
