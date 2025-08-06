#!/usr/bin/env python3
"""
GPT-OSS Model Runner
Main entry point for loading, validating, and running inference with GPT-OSS models.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json

from gptoss_loader import load_model
from gptoss_inference import InferencePipeline, create_generation_config, benchmark_inference
from gptoss_validator import validate_model


def main():
    parser = argparse.ArgumentParser(
        description="GPT-OSS Model Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and validate model
  python run_gptoss.py --model-path gpt-oss-20b --validate
  
  # Generate text
  python run_gptoss.py --model-path gpt-oss-20b --generate "Hello, my name is"
  
  # Run benchmark
  python run_gptoss.py --model-path gpt-oss-20b --benchmark
  
  # Skip validation for faster loading
  python run_gptoss.py --model-path gpt-oss-20b --skip-validation --generate "Test"
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="gpt-oss-20b",
        help="Path to model directory or file"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to config.json"
    )
    
    # Action arguments
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation tests on the model"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip parameter validation during loading"
    )
    parser.add_argument(
        "--generate",
        type=str,
        default=None,
        help="Generate text from a prompt (uses dummy tokenization)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 = greedy)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream generated tokens"
    )
    
    args = parser.parse_args()
    
    # Check model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        sys.exit(1)
    
    print("="*60)
    print("GPT-OSS MODEL RUNNER")
    print("="*60)
    
    try:
        # Load model
        print(f"\n📂 Loading model from: {model_path}")
        model, config = load_model(
            model_path,
            config_path=args.config_path,
            validate=not args.skip_validation
        )
        
        print(f"\n✅ Model loaded successfully!")
        print(f"📊 Configuration:")
        print(f"  • Hidden size: {config.hidden_size}")
        print(f"  • Layers: {config.num_hidden_layers}")
        print(f"  • Attention heads: {config.num_attention_heads}")
        print(f"  • Vocab size: {config.vocab_size:,}")
        print(f"  • Quantization: {config.quantization_method}")
        
        # Run validation if requested
        if args.validate:
            print("\n" + "="*60)
            print("VALIDATION")
            print("="*60)
            results = validate_model(model, config, save_results=True)
            
            if not results["all_passed"]:
                print("\n⚠️  Validation failed - model may not work correctly")
                if not args.generate and not args.benchmark:
                    sys.exit(1)
        
        # Create inference pipeline
        pipeline = InferencePipeline(model, config)
        
        # Generate text if requested
        if args.generate:
            print("\n" + "="*60)
            print("TEXT GENERATION")
            print("="*60)
            print(f"📝 Input: '{args.generate}'")
            
            # Simple tokenization (in production, use proper tokenizer)
            # For demo, we'll use dummy token IDs
            input_ids = list(range(1, min(len(args.generate.split()) + 1, 100)))
            
            print(f"🤖 Generating up to {args.max_tokens} tokens...")
            
            if args.stream:
                print("Generated tokens: ", end="", flush=True)
                for token_id in pipeline.generate(
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    stream=True
                ):
                    print(f"{token_id} ", end="", flush=True)
                print()
            else:
                generated = pipeline.generate(
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                
                print(f"✅ Generated sequence shape: {generated.shape}")
                print(f"📊 Generated {generated.shape[1] - len(input_ids)} new tokens")
                
                # Show first few generated token IDs
                generated_tokens = generated[0, len(input_ids):len(input_ids)+10].tolist()
                print(f"First 10 generated token IDs: {generated_tokens}")
                
                print("\nNote: To see actual text, integrate with a proper tokenizer")
        
        # Run benchmark if requested
        if args.benchmark:
            print("\n" + "="*60)
            print("PERFORMANCE BENCHMARK")
            print("="*60)
            
            # Create test sequences
            test_sequences = [
                list(range(1, 11)),  # 10 tokens
                list(range(1, 51)),  # 50 tokens
                list(range(1, 101)), # 100 tokens
            ]
            
            metrics = benchmark_inference(
                pipeline,
                test_sequences,
                max_new_tokens=50
            )
            
            # Save benchmark results
            benchmark_file = Path("benchmark_results.json")
            with open(benchmark_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n📝 Benchmark results saved to {benchmark_file}")
        
        # Quick test if no specific action requested
        if not args.validate and not args.generate and not args.benchmark:
            print("\n🧪 Running quick test...")
            test_input = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
            
            try:
                output, _ = model(test_input, use_cache=False)
                print(f"✅ Quick test successful!")
                print(f"  • Input shape: {test_input.shape}")
                print(f"  • Output shape: {output.shape}")
                
                if np.isfinite(output).all():
                    print(f"  • Output values are finite ✓")
                else:
                    print(f"  • Warning: Output contains NaN or Inf values")
                    
            except Exception as e:
                print(f"❌ Quick test failed: {e}")
                sys.exit(1)
        
        print("\n" + "="*60)
        print("✨ COMPLETE")
        print("="*60)
        print("Model is loaded and ready for use!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()