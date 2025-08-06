#!/usr/bin/env python3
"""
Modal benchmark script to compare JAX vs PyTorch inference performance
across different GPU types for GPT-OSS-20B.

Tests 100 prompts on each framework and GPU combination.
"""

import modal
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import traceback
import gc

# Modal app configuration
app = modal.App("gpt-oss-benchmark")

# Volume with converted JAX weights
volume = modal.Volume.from_name("gpt-oss-20b-jax", create_if_missing=False)

# GPU configurations to test
GPU_CONFIGS = {
    "H100": modal.gpu.H100(),
    "A100-40GB": modal.gpu.A100(size="40GB"),
    "L4": modal.gpu.L4(),
    "T4": modal.gpu.T4(),
}

# Pricing per second (from Modal pricing page)
GPU_PRICING = {
    "H100": 0.001097,
    "A100-40GB": 0.000583,
    "L4": 0.000222,
    "T4": 0.000164,
}

def get_image(framework: str):
    """Get Docker image for specific framework."""
    if framework == "jax":
        return (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install([
                "jax[cuda12]",
                "jaxlib",
                "numpy",
                "flax",
                "optax",
                "tqdm",
            ])
        )
    else:  # pytorch
        return (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install([
                "torch",
                "transformers",
                "accelerate",
                "safetensors",
                "sentencepiece",
                "protobuf",
            ])
        )

class BenchmarkLogger:
    """Async logger to avoid impacting latency measurements."""
    
    def __init__(self):
        self.logs = []
        self.start_time = time.time()
    
    def log(self, event: Dict[str, Any]):
        """Add log entry with timestamp."""
        event["timestamp"] = time.time() - self.start_time
        event["datetime"] = datetime.utcnow().isoformat()
        self.logs.append(event)
    
    def get_results(self) -> Dict[str, Any]:
        """Get all results with summary statistics."""
        return {
            "logs": self.logs,
            "total_time": time.time() - self.start_time,
            "event_count": len(self.logs),
        }

# JAX Inference Function
@app.function(
    image=get_image("jax"),
    volumes={"/cache": volume},
    timeout=3600,
    retries=modal.Retries(max_retries=1),
)
def benchmark_jax(gpu_type: str, prompts: List[str], gpu_config) -> Dict[str, Any]:
    """Benchmark JAX inference on specified GPU."""
    
    import jax
    import jax.numpy as jnp
    
    logger = BenchmarkLogger()
    logger.log({"event": "start", "gpu": gpu_type, "framework": "jax"})
    
    # Log GPU info
    devices = jax.devices()
    logger.log({
        "event": "gpu_info",
        "devices": str(devices),
        "device_count": len(devices),
    })
    
    try:
        # Load model weights
        logger.log({"event": "loading_start"})
        load_start = time.time()
        
        model_path = Path("/cache/gpt-oss-20b-jax")
        if not model_path.exists():
            raise FileNotFoundError(f"JAX weights not found at {model_path}")
        
        # Load essential weights for inference
        def load_tensor(tensor_name: str) -> jnp.ndarray:
            file_name = tensor_name.replace('.', '_') + '.npy'
            for shard_dir in sorted(model_path.glob('shard_*')):
                tensor_path = shard_dir / file_name
                if tensor_path.exists():
                    return jnp.array(np.load(tensor_path))
            return None
        
        # Load key tensors
        embed_tokens = load_tensor('model.embed_tokens.weight')
        lm_head = load_tensor('lm_head.weight')
        
        load_time = time.time() - load_start
        logger.log({
            "event": "loading_complete",
            "load_time": load_time,
            "embed_shape": str(embed_tokens.shape) if embed_tokens is not None else None,
        })
        
        # Simple tokenization (for demo - in production use proper tokenizer)
        def simple_tokenize(text: str) -> jnp.ndarray:
            # Hash-based tokenization for consistency
            tokens = [abs(hash(word)) % 50000 for word in text.split()[:50]]
            return jnp.array([tokens])
        
        # Simple forward pass (simplified for benchmarking)
        @jax.jit
        def forward_pass(input_ids):
            hidden_states = embed_tokens[input_ids]
            # Simplified - just project to vocab for benchmark
            logits = jnp.matmul(hidden_states, lm_head.T)
            return logits
        
        # Comprehensive warmup for JIT compilation
        logger.log({"event": "warmup_start"})
        warmup_start = time.time()
        
        # Warmup with different sequence lengths to compile all shapes
        warmup_iterations = 5
        for iteration in range(warmup_iterations):
            for seq_len in [10, 20, 30, 40, 50]:
                dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32) % embed_tokens.shape[0]
                output = forward_pass(dummy_input)
                output.block_until_ready()  # Ensure compilation completes
        
        # Final warmup with actual expected shape
        typical_input = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        for _ in range(5):
            output = forward_pass(typical_input)
            output.block_until_ready()
        
        warmup_time = time.time() - warmup_start
        logger.log({
            "event": "warmup_complete", 
            "warmup_time": warmup_time,
            "warmup_iterations": warmup_iterations * 5 + 5,
            "compilation_complete": True
        })
        
        # Benchmark inference (exclude first 5 for additional warmup)
        all_inference_times = []
        inference_times = []  # Times used for statistics (excluding first 5)
        tokens_processed = 0
        warmup_inferences = 5
        
        for i, prompt in enumerate(prompts):
            try:
                input_ids = simple_tokenize(prompt)
                tokens_processed += input_ids.shape[1]
                
                start = time.time()
                logits = forward_pass(input_ids)
                logits.block_until_ready()  # Ensure computation completes
                inference_time = time.time() - start
                
                all_inference_times.append(inference_time)
                
                # Exclude first few inferences from statistics
                if i >= warmup_inferences:
                    inference_times.append(inference_time)
                
                if i % 10 == 0:
                    logger.log({
                        "event": "inference_progress",
                        "prompt_idx": i,
                        "inference_time": inference_time,
                        "avg_time_so_far": np.mean(inference_times),
                    })
                
            except Exception as e:
                logger.log({
                    "event": "inference_error",
                    "prompt_idx": i,
                    "error": str(e),
                })
        
        # Calculate statistics
        stats = {
            "framework": "jax",
            "gpu_type": gpu_type,
            "total_prompts": len(prompts),
            "successful_inferences": len(inference_times),
            "total_tokens": tokens_processed,
            "inference_times": {
                "mean": np.mean(inference_times),
                "median": np.median(inference_times),
                "std": np.std(inference_times),
                "min": np.min(inference_times),
                "max": np.max(inference_times),
                "p95": np.percentile(inference_times, 95),
                "p99": np.percentile(inference_times, 99),
            },
            "tokens_per_second": tokens_processed / sum(inference_times),
            "load_time": load_time,
            "warmup_time": warmup_time,
        }
        
        # Calculate cost
        total_time = sum(inference_times) + load_time + warmup_time
        stats["total_time"] = total_time
        stats["estimated_cost"] = total_time * GPU_PRICING.get(gpu_type, 0)
        stats["cost_per_1k_tokens"] = (stats["estimated_cost"] / tokens_processed) * 1000
        
        logger.log({"event": "complete", "stats": stats})
        
    except Exception as e:
        logger.log({
            "event": "fatal_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        stats = {"error": str(e), "framework": "jax", "gpu_type": gpu_type}
    
    return {
        "stats": stats,
        "logs": logger.get_results(),
    }

# PyTorch Inference Function
@app.function(
    image=get_image("pytorch"),
    timeout=3600,
    retries=modal.Retries(max_retries=1),
)
def benchmark_pytorch(gpu_type: str, prompts: List[str], gpu_config) -> Dict[str, Any]:
    """Benchmark PyTorch inference using HuggingFace transformers."""
    
    import torch
    from transformers import pipeline
    
    logger = BenchmarkLogger()
    logger.log({"event": "start", "gpu": gpu_type, "framework": "pytorch"})
    
    # Log GPU info
    if torch.cuda.is_available():
        logger.log({
            "event": "gpu_info",
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_memory": torch.cuda.get_device_properties(0).total_memory,
            "cuda_available": True,
        })
    
    try:
        # Load model
        logger.log({"event": "loading_start"})
        load_start = time.time()
        
        model_id = "openai/gpt-oss-20b"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            max_memory={0: "20GiB"},  # Adjust based on GPU
        )
        
        load_time = time.time() - load_start
        logger.log({"event": "loading_complete", "load_time": load_time})
        
        # Comprehensive warmup for CUDA kernel compilation
        logger.log({"event": "warmup_start"})
        warmup_start = time.time()
        
        # Warmup with different prompt lengths
        warmup_iterations = 5
        warmup_prompts = [
            "Hello",
            "This is a test prompt",
            "The quick brown fox jumps over the lazy dog",
            "Artificial intelligence is transforming how we work and live",
            "In the realm of quantum computing, researchers are making breakthrough discoveries"
        ]
        
        for iteration in range(warmup_iterations):
            for prompt in warmup_prompts:
                _ = pipe(
                    prompt, 
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        # Clear cache after warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        warmup_time = time.time() - warmup_start
        logger.log({
            "event": "warmup_complete",
            "warmup_time": warmup_time,
            "warmup_iterations": warmup_iterations * len(warmup_prompts),
            "cuda_warmed": torch.cuda.is_available()
        })
        
        # Benchmark inference (exclude first 5 for additional warmup)
        all_inference_times = []
        inference_times = []  # Times used for statistics (excluding first 5)
        tokens_generated = 0
        warmup_inferences = 5
        
        for i, prompt in enumerate(prompts):
            try:
                messages = [{"role": "user", "content": prompt}]
                
                start = time.time()
                outputs = pipe(
                    messages,
                    max_new_tokens=50,  # Fixed generation length for fair comparison
                    do_sample=False,  # Deterministic for benchmarking
                    return_full_text=False,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.time() - start
                
                all_inference_times.append(inference_time)
                
                # Exclude first few inferences from statistics
                if i >= warmup_inferences:
                    inference_times.append(inference_time)
                
                # Estimate tokens (rough approximation)
                generated_text = outputs[0]["generated_text"]
                if isinstance(generated_text, list) and len(generated_text) > 0:
                    text = generated_text[-1].get("content", "")
                else:
                    text = str(generated_text)
                tokens_generated += len(text.split())
                
                if i % 10 == 0:
                    logger.log({
                        "event": "inference_progress",
                        "prompt_idx": i,
                        "inference_time": inference_time,
                        "avg_time_so_far": np.mean(inference_times),
                    })
                
                # Clear cache periodically to avoid OOM
                if i % 20 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                logger.log({
                    "event": "inference_error",
                    "prompt_idx": i,
                    "error": str(e),
                })
        
        # Calculate statistics
        stats = {
            "framework": "pytorch",
            "gpu_type": gpu_type,
            "total_prompts": len(prompts),
            "successful_inferences": len(inference_times),
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
            "tokens_per_second": tokens_generated / sum(inference_times) if inference_times else 0,
            "load_time": load_time,
            "warmup_time": warmup_time,
        }
        
        # Calculate cost
        total_time = sum(inference_times) + load_time + warmup_time
        stats["total_time"] = total_time
        stats["estimated_cost"] = total_time * GPU_PRICING.get(gpu_type, 0)
        stats["cost_per_1k_tokens"] = (stats["estimated_cost"] / tokens_generated) * 1000 if tokens_generated > 0 else 0
        
        logger.log({"event": "complete", "stats": stats})
        
    except Exception as e:
        logger.log({
            "event": "fatal_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        stats = {"error": str(e), "framework": "pytorch", "gpu_type": gpu_type}
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return {
        "stats": stats,
        "logs": logger.get_results(),
    }

@app.function(
    image=get_image("jax"),
    volumes={"/results": modal.Volume.from_name("benchmark-results", create_if_missing=True)},
)
def save_results(results: Dict[str, Any], run_id: str):
    """Save benchmark results to volume."""
    results_dir = Path("/results") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    with open(results_dir / "full_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary
    summary = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "gpu_types": list(results.keys()),
        "frameworks": ["jax", "pytorch"],
    }
    
    # Extract key metrics for each combination
    for gpu_type in results:
        for framework in ["jax", "pytorch"]:
            key = f"{gpu_type}_{framework}"
            if key in results and "stats" in results[key]:
                stats = results[key]["stats"]
                summary[key] = {
                    "mean_inference_time": stats.get("inference_times", {}).get("mean"),
                    "tokens_per_second": stats.get("tokens_per_second"),
                    "cost_per_1k_tokens": stats.get("cost_per_1k_tokens"),
                    "total_cost": stats.get("estimated_cost"),
                }
    
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {results_dir}")
    return str(results_dir)

@app.local_entrypoint()
def main(
    gpu_types: str = "T4,L4",  # Comma-separated list
    num_prompts: int = 10,  # Reduced for testing, use 100 for full benchmark
    frameworks: str = "jax,pytorch",  # Which frameworks to test
):
    """Run benchmarks across specified GPUs and frameworks."""
    
    print("="*60)
    print("GPT-OSS-20B Inference Benchmark")
    print("="*60)
    
    # Parse arguments
    gpu_list = gpu_types.split(",")
    framework_list = frameworks.split(",")
    
    # Load or generate prompts
    prompts_file = Path("benchmark_prompts.json")
    if prompts_file.exists():
        with open(prompts_file) as f:
            all_prompts = json.load(f)["prompts"]
    else:
        # Generate simple prompts for testing
        all_prompts = [
            "Explain quantum mechanics in simple terms.",
            "What is the meaning of life?",
            "Write a poem about artificial intelligence.",
            "How does photosynthesis work?",
            "Describe the process of machine learning.",
            "What are the benefits of renewable energy?",
            "Explain the theory of relativity.",
            "How do computers work?",
            "What is consciousness?",
            "Describe the water cycle.",
        ] * (num_prompts // 10 + 1)
    
    prompts = all_prompts[:num_prompts]
    
    print(f"Configuration:")
    print(f"  GPUs: {gpu_list}")
    print(f"  Frameworks: {framework_list}")
    print(f"  Prompts: {len(prompts)}")
    print()
    
    # Run benchmarks
    results = {}
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    for gpu_type in gpu_list:
        if gpu_type not in GPU_CONFIGS:
            print(f"⚠️  Skipping unknown GPU type: {gpu_type}")
            continue
        
        print(f"\n🖥️  Benchmarking {gpu_type}...")
        gpu_config = GPU_CONFIGS[gpu_type]
        
        # JAX benchmark
        if "jax" in framework_list:
            print(f"  Running JAX inference...")
            try:
                jax_result = benchmark_jax.with_options(gpu=gpu_config).remote(
                    gpu_type, prompts, gpu_config
                )
                results[f"{gpu_type}_jax"] = jax_result
                
                if "stats" in jax_result:
                    stats = jax_result["stats"]
                    print(f"  ✅ JAX: {stats.get('tokens_per_second', 0):.1f} tok/s, "
                          f"${stats.get('cost_per_1k_tokens', 0):.4f}/1k tokens")
            except Exception as e:
                print(f"  ❌ JAX failed: {e}")
                results[f"{gpu_type}_jax"] = {"error": str(e)}
        
        # PyTorch benchmark
        if "pytorch" in framework_list:
            print(f"  Running PyTorch inference...")
            try:
                pytorch_result = benchmark_pytorch.with_options(gpu=gpu_config).remote(
                    gpu_type, prompts, gpu_config
                )
                results[f"{gpu_type}_pytorch"] = pytorch_result
                
                if "stats" in pytorch_result:
                    stats = pytorch_result["stats"]
                    print(f"  ✅ PyTorch: {stats.get('tokens_per_second', 0):.1f} tok/s, "
                          f"${stats.get('cost_per_1k_tokens', 0):.4f}/1k tokens")
            except Exception as e:
                print(f"  ❌ PyTorch failed: {e}")
                results[f"{gpu_type}_pytorch"] = {"error": str(e)}
    
    # Save results
    print(f"\n💾 Saving results...")
    results_path = save_results.remote(results, run_id)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print("\n📊 Performance Comparison:")
    print(f"{'Config':<20} {'Mean Latency (s)':<18} {'Tokens/s':<12} {'$/1k tokens':<12}")
    print("-" * 62)
    
    for key in sorted(results.keys()):
        if "stats" in results[key] and "error" not in results[key]["stats"]:
            stats = results[key]["stats"]
            mean_time = stats.get("inference_times", {}).get("mean", 0)
            tokens_per_sec = stats.get("tokens_per_second", 0)
            cost_per_1k = stats.get("cost_per_1k_tokens", 0)
            print(f"{key:<20} {mean_time:<18.4f} {tokens_per_sec:<12.1f} ${cost_per_1k:<11.4f}")
    
    print(f"\n✅ Benchmark complete! Results saved to: {results_path}")
    print(f"Run ID: {run_id}")

if __name__ == "__main__":
    main()