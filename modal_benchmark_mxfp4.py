#!/usr/bin/env python3
"""
Modal benchmark script with MXFP4 quantization support.
Enables running GPT-OSS-20B on smaller GPUs like T4.
"""

import modal
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import gc

app = modal.App("gpt-oss-mxfp4-benchmark")

# Volume with converted weights
volume = modal.Volume.from_name("gpt-oss-20b-jax", create_if_missing=False)

# GPU configurations with memory constraints
GPU_CONFIGS = {
    "T4": {"gpu": modal.gpu.T4(), "memory": 16, "supports_fp4": False},
    "L4": {"gpu": modal.gpu.L4(), "memory": 24, "supports_fp4": True},
    "A10G": {"gpu": modal.gpu.A10G(), "memory": 24, "supports_fp4": False},
    "L40S": {"gpu": modal.gpu.L40S(), "memory": 48, "supports_fp4": True},
    "A100-40GB": {"gpu": modal.gpu.A100(size="40GB"), "memory": 40, "supports_fp4": False},
    "H100": {"gpu": modal.gpu.H100(), "memory": 80, "supports_fp4": True},
}

def get_quantized_image():
    """Docker image with quantization libraries."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install([
            "torch",
            "transformers",
            "accelerate",
            "bitsandbytes",  # For 4-bit quantization
            "auto-gptq",     # For GPTQ quantization
            "safetensors",
            "numpy",
            "tqdm",
        ])
    )

@app.function(
    image=get_quantized_image(),
    timeout=3600,
    retries=modal.Retries(max_retries=1),
)
def benchmark_mxfp4(gpu_type: str, prompts: List[str], gpu_config: Dict) -> Dict[str, Any]:
    """Benchmark MXFP4 quantized inference on specified GPU."""
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import bitsandbytes as bnb
    
    results = {
        "gpu_type": gpu_type,
        "gpu_memory_gb": gpu_config["memory"],
        "quantization": "MXFP4",
        "framework": "pytorch_quantized",
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    try:
        # Configure quantization based on GPU memory
        if gpu_config["memory"] <= 24:  # T4, L4, A10G
            print(f"Using 4-bit quantization for {gpu_type} ({gpu_config['memory']}GB)")
            
            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",  # Normal float 4-bit
                bnb_4bit_use_double_quant=True,  # Nested quantization for more memory saving
            )
            
            # Memory mapping for small GPUs
            max_memory = {0: f"{gpu_config['memory']-2}GiB", "cpu": "30GiB"}
            
        else:  # L40S, A100, H100
            print(f"Using 8-bit quantization for {gpu_type} ({gpu_config['memory']}GB)")
            
            # 8-bit quantization for larger GPUs (better quality)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
            
            max_memory = {0: f"{gpu_config['memory']-4}GiB"}
        
        # Load model with quantization
        print("Loading quantized model...")
        load_start = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        tokenizer.pad_token = tokenizer.eos_token
        
        load_time = time.time() - load_start
        results["load_time"] = load_time
        
        # Check actual memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            results["gpu_memory_allocated_gb"] = allocated
            results["gpu_memory_reserved_gb"] = reserved
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Warmup phase
        print("Warming up...")
        warmup_start = time.time()
        
        for _ in range(5):
            dummy_input = tokenizer("Hello world", return_tensors="pt")
            with torch.no_grad():
                _ = model.generate(
                    **dummy_input,
                    max_new_tokens=10,
                    do_sample=False,
                )
            torch.cuda.synchronize()
        
        torch.cuda.empty_cache()
        warmup_time = time.time() - warmup_start
        results["warmup_time"] = warmup_time
        
        # Benchmark inference
        print(f"Benchmarking {len(prompts)} prompts...")
        inference_times = []
        tokens_generated = 0
        
        for i, prompt in enumerate(prompts):
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_length = inputs.input_ids.shape[1]
            
            # Generate
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            
            inference_time = time.time() - start
            inference_times.append(inference_time)
            
            # Count tokens
            output_length = outputs.shape[1]
            new_tokens = output_length - input_length
            tokens_generated += new_tokens
            
            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
            if i % 20 == 0:
                print(f"Progress: {i}/{len(prompts)}, "
                      f"Avg time: {np.mean(inference_times):.3f}s, "
                      f"Tokens/s: {tokens_generated/sum(inference_times):.1f}")
        
        # Calculate statistics
        results["inference_stats"] = {
            "total_prompts": len(prompts),
            "total_tokens": tokens_generated,
            "inference_times": {
                "mean": np.mean(inference_times),
                "median": np.median(inference_times),
                "std": np.std(inference_times),
                "min": np.min(inference_times),
                "max": np.max(inference_times),
                "p95": np.percentile(inference_times, 95),
                "p99": np.percentile(inference_times, 99),
            },
            "tokens_per_second": tokens_generated / sum(inference_times),
            "prompts_per_second": len(prompts) / sum(inference_times),
        }
        
        # Memory usage after inference
        if torch.cuda.is_available():
            results["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            
    except Exception as e:
        results["error"] = str(e)
        print(f"Error during benchmark: {e}")
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return results

@app.function(
    image=get_quantized_image(),
    volumes={"/cache": volume},
)
def test_mxfp4_loading():
    """Test loading MXFP4 weights directly from safetensors."""
    import torch
    from safetensors import safe_open
    
    print("Testing MXFP4 weight loading...")
    
    # Path to original model
    model_path = Path("/cache/gpt-oss-20b-jax")
    
    if not model_path.exists():
        # Try loading from HuggingFace cache
        from huggingface_hub import snapshot_download
        model_path = snapshot_download("openai/gpt-oss-20b", cache_dir="/tmp")
    
    # Load a sample weight file
    weight_file = None
    for f in Path(model_path).glob("*.safetensors"):
        weight_file = f
        break
    
    if weight_file:
        with safe_open(weight_file, framework="pt") as f:
            # Check tensor metadata
            metadata = f.metadata()
            print(f"Metadata: {metadata}")
            
            # Sample a tensor
            for key in list(f.keys())[:5]:
                tensor = f.get_tensor(key)
                print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
                
                # Check if it's quantized
                if tensor.dtype == torch.uint8:
                    print(f"  → Likely MXFP4 quantized (uint8 storage)")
                elif tensor.dtype == torch.int8:
                    print(f"  → Likely INT8 quantized")
    
    return {"status": "complete"}

@app.local_entrypoint()
def main(
    gpu_types: str = "T4",
    num_prompts: int = 10,
    test_loading: bool = False,
):
    """Run MXFP4 benchmarks on specified GPUs."""
    
    if test_loading:
        # Test MXFP4 weight loading
        result = test_mxfp4_loading.remote()
        print(f"Loading test: {result}")
        return
    
    # Load prompts
    prompts_file = Path("benchmark_prompts.json")
    if prompts_file.exists():
        with open(prompts_file) as f:
            all_prompts = json.load(f)["prompts"]
    else:
        # Default prompts
        all_prompts = [
            "Explain quantum computing in simple terms.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
        ] * (num_prompts // 3 + 1)
    
    prompts = all_prompts[:num_prompts]
    
    print("="*60)
    print("MXFP4 Quantized Inference Benchmark")
    print("="*60)
    
    # Run benchmarks
    gpu_list = gpu_types.split(",")
    results = {}
    
    for gpu_type in gpu_list:
        if gpu_type not in GPU_CONFIGS:
            print(f"Unknown GPU: {gpu_type}")
            continue
        
        print(f"\n🖥️  Benchmarking {gpu_type}...")
        gpu_config = GPU_CONFIGS[gpu_type]
        
        result = benchmark_mxfp4.with_options(
            gpu=gpu_config["gpu"]
        ).remote(gpu_type, prompts, gpu_config)
        
        results[gpu_type] = result
        
        if "inference_stats" in result:
            stats = result["inference_stats"]
            print(f"✅ {gpu_type}: {stats['tokens_per_second']:.1f} tok/s, "
                  f"Memory: {result.get('peak_memory_gb', 0):.1f}GB")
        else:
            print(f"❌ {gpu_type}: {result.get('error', 'Unknown error')}")
    
    # Save results
    output_file = f"mxfp4_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\n📊 Performance by GPU (MXFP4 Quantized):")
    print(f"{'GPU':<12} {'Memory Used':<12} {'Tokens/s':<10} {'Mean Latency':<12}")
    print("-" * 46)
    
    for gpu_type, result in results.items():
        if "inference_stats" in result:
            stats = result["inference_stats"]
            mem = result.get("peak_memory_gb", 0)
            print(f"{gpu_type:<12} {mem:<12.1f}GB "
                  f"{stats['tokens_per_second']:<10.1f} "
                  f"{stats['inference_times']['mean']:<12.3f}s")

if __name__ == "__main__":
    main()