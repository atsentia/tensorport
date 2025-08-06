#!/usr/bin/env python3
"""
Simple script to load and test the GPT-OSS-20B model.
Uses the production-ready implementation from evennewerfile.py
"""

import sys
import os
from pathlib import Path

# Check if we need to install dependencies
try:
    import safetensors
except ImportError:
    print("Installing safetensors...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "-q"])

# Import the complete implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evennewerfile import load_model, GPTOSSConfig

def main():
    model_path = "gpt-oss-20b"
    
    print(f"📂 Loading GPT-OSS-20B model from: {model_path}")
    print("=" * 60)
    
    try:
        # Load the model
        model, config = load_model(model_path)
        
        print("✅ Model loaded successfully!")
        print(f"\n📊 Model Configuration:")
        print(f"  • Hidden size: {config.hidden_size}")
        print(f"  • Layers: {config.num_hidden_layers}")
        print(f"  • Attention heads: {config.num_attention_heads}")
        print(f"  • KV heads: {config.num_key_value_heads}")
        print(f"  • Vocab size: {config.vocab_size:,}")
        print(f"  • Max positions: {config.max_position_embeddings:,}")
        print(f"  • Experts: {config.num_local_experts}")
        print(f"  • Active experts: {config.num_experts_per_tok}")
        print(f"  • Quantization: {config.quantization_method}")
        
        # Quick test with dummy input
        print(f"\n🧪 Running quick inference test...")
        import numpy as np
        test_input = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        
        try:
            output = model(test_input)
            print(f"✅ Inference successful!")
            print(f"  • Input shape: {test_input.shape}")
            print(f"  • Output shape: {output.shape}")
            print(f"  • Output dtype: {output.dtype}")
            
            # Check output validity
            if np.isfinite(output).all():
                print(f"  • Output values are finite ✓")
            else:
                print(f"  • Warning: Output contains NaN or Inf values")
                
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            
        print("\n" + "=" * 60)
        print("🎉 Model is ready for use!")
        print("\nNext steps:")
        print("  1. Load proper tokenizer for text processing")
        print("  2. Implement generation/sampling strategies")
        print("  3. Run comprehensive validation tests")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()