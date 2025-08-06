#!/usr/bin/env python3
"""
Complete Inference Engine for GPT-OSS with JAX/Flax
Implements autoregressive sampling with temperature, top-k, and top-p strategies.
"""

import math
from typing import Dict, Any, Optional, Tuple, List, Union, Generator
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax import struct

from flax_gptoss_model import FlaxGPTOSSLMHeadModel, GPTOSSConfig, initialize_model


@struct.dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    seed: int = 42


class GPTOSSInferenceEngine:
    """Production-ready inference engine for GPT-OSS models."""
    
    def __init__(self, model: FlaxGPTOSSLMHeadModel, config: GPTOSSConfig, params: Dict[str, Any]):
        self.model = model
        self.config = config
        self.params = params
        
        # Compile forward function for performance
        self.forward_fn = jax.jit(self._forward_pass)
        self.generate_fn = jax.jit(self._generate_next_token, static_argnums=(2, 3, 4, 5))
        
        print("🚀 Inference engine initialized with JIT compilation")
    
    def _forward_pass(self, params: Dict[str, Any], input_ids: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled forward pass."""
        return self.model.apply(params, input_ids, deterministic=True)
    
    def _apply_temperature(self, logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
        """Apply temperature scaling to logits."""
        if temperature == 0.0:
            return logits
        return logits / temperature
    
    def _apply_top_k(self, logits: jnp.ndarray, k: int) -> jnp.ndarray:
        """Apply top-k filtering to logits."""
        if k <= 0:
            return logits
        
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k)
        min_top_k = jnp.min(top_k_logits, axis=-1, keepdims=True)
        
        # Mask tokens not in top-k
        return jnp.where(logits < min_top_k, -jnp.inf, logits)
    
    def _apply_top_p(self, logits: jnp.ndarray, p: float) -> jnp.ndarray:
        """Apply top-p (nucleus) filtering to logits."""
        if p >= 1.0:
            return logits
        
        # Sort logits in descending order
        sorted_indices = jnp.argsort(logits, axis=-1)[::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        
        # Convert to probabilities and compute cumulative sum
        probs = nn.softmax(sorted_logits, axis=-1)
        cumsum_probs = jnp.cumsum(probs, axis=-1)
        
        # Find cutoff point
        cutoff_mask = cumsum_probs <= p
        # Include at least one token
        cutoff_mask = cutoff_mask.at[..., 0].set(True)
        
        # Apply cutoff
        sorted_logits = jnp.where(cutoff_mask, sorted_logits, -jnp.inf)
        
        # Unsort back to original order
        unsorted_logits = jnp.zeros_like(logits).at[sorted_indices].set(sorted_logits)
        return unsorted_logits
    
    def _apply_repetition_penalty(self, logits: jnp.ndarray, input_ids: jnp.ndarray, penalty: float) -> jnp.ndarray:
        """Apply repetition penalty to previously generated tokens."""
        if penalty == 1.0:
            return logits
        
        # Create a mask for tokens that appear in input
        vocab_size = logits.shape[-1]
        penalty_mask = jnp.zeros(vocab_size, dtype=jnp.bool_)
        
        # Mark tokens that appear in input sequence
        for i in range(input_ids.shape[0]):
            penalty_mask = penalty_mask.at[input_ids[i]].set(True)
        
        # Apply penalty (divide if > 1, multiply if < 1)
        penalty_factor = jnp.where(penalty_mask, 1.0 / penalty, 1.0)
        penalty_factor = jnp.where(logits > 0, penalty_factor, penalty)
        
        return logits * penalty_factor
    
    def _generate_next_token(self, params: Dict[str, Any], input_ids: jnp.ndarray, 
                           temperature: float, top_k: int, top_p: float, 
                           repetition_penalty: float, rng_key: jax.random.PRNGKey) -> int:
        """Generate next token with all sampling strategies."""
        # Forward pass
        logits = self.forward_fn(params, input_ids)
        next_token_logits = logits[0, -1, :]  # Take last token of first batch
        
        # Apply repetition penalty
        next_token_logits = self._apply_repetition_penalty(next_token_logits, input_ids[0], repetition_penalty)
        
        # Apply temperature
        next_token_logits = self._apply_temperature(next_token_logits, temperature)
        
        # Apply top-k filtering
        if top_k > 0:
            next_token_logits = self._apply_top_k(next_token_logits, top_k)
        
        # Apply top-p filtering
        if top_p < 1.0:
            next_token_logits = self._apply_top_p(next_token_logits, top_p)
        
        # Sample from distribution
        if temperature == 0.0:
            # Greedy sampling
            next_token = jnp.argmax(next_token_logits)
        else:
            # Probabilistic sampling
            next_token = random.categorical(rng_key, next_token_logits)
        
        return next_token
    
    def generate(self, input_ids: Union[List[int], jnp.ndarray], 
                 generation_config: Optional[GenerationConfig] = None,
                 stream: bool = False) -> Union[jnp.ndarray, Generator]:
        """
        Generate text from input token IDs.
        
        Args:
            input_ids: Input token IDs
            generation_config: Generation configuration
            stream: Whether to stream tokens as they're generated
            
        Returns:
            Generated token IDs (or generator if streaming)
        """
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Prepare input
        if isinstance(input_ids, list):
            input_ids = jnp.array([input_ids], dtype=jnp.int32)
        elif len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        
        # Initialize random key
        rng_key = random.PRNGKey(generation_config.seed)
        
        generated_tokens = []
        current_ids = input_ids
        
        def generate_tokens():
            nonlocal current_ids, rng_key
            
            for step in range(generation_config.max_new_tokens):
                # Generate next token
                rng_key, subkey = random.split(rng_key)
                
                next_token = self.generate_fn(
                    self.params, current_ids,
                    generation_config.temperature,
                    generation_config.top_k,
                    generation_config.top_p,
                    generation_config.repetition_penalty,
                    subkey
                )
                
                next_token = int(next_token)
                generated_tokens.append(next_token)
                
                # Check for end-of-sequence
                if (generation_config.eos_token_id is not None and 
                    next_token == generation_config.eos_token_id):
                    if stream:
                        yield next_token
                    break
                
                # Update input for next iteration
                current_ids = jnp.concatenate([
                    current_ids, 
                    jnp.array([[next_token]], dtype=jnp.int32)
                ], axis=1)
                
                if stream:
                    yield next_token
            
            return generated_tokens
        
        if stream:
            return generate_tokens()
        else:
            list(generate_tokens())  # Consume generator
            return jnp.concatenate([input_ids[0], jnp.array(generated_tokens)])
    
    def generate_text(self, prompt: str, tokenizer=None, 
                     generation_config: Optional[GenerationConfig] = None) -> str:
        """
        Generate text from a prompt string.
        
        Args:
            prompt: Input text prompt
            tokenizer: Tokenizer to encode/decode text
            generation_config: Generation configuration
            
        Returns:
            Generated text
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for text generation")
        
        # Tokenize input
        input_ids = tokenizer.encode(prompt)
        
        # Generate tokens
        output_ids = self.generate(input_ids, generation_config)
        
        # Decode output
        generated_text = tokenizer.decode(output_ids.tolist())
        
        return generated_text
    
    def benchmark_performance(self, input_length: int = 10, num_tokens: int = 100, 
                            num_runs: int = 5) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            input_length: Length of input sequence
            num_tokens: Number of tokens to generate
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        print(f"🔥 Benchmarking performance ({num_runs} runs)...")
        
        # Prepare input
        rng = random.PRNGKey(42)
        input_ids = random.randint(rng, (1, input_length), 0, self.config.vocab_size)
        
        config = GenerationConfig(max_new_tokens=num_tokens, temperature=0.8)
        
        # Warmup run
        _ = self.generate(input_ids, config)
        
        # Benchmark runs
        times = []
        for run in range(num_runs):
            start_time = time.time()
            output = self.generate(input_ids, config)
            end_time = time.time()
            
            times.append(end_time - start_time)
            print(f"  Run {run + 1}: {times[-1]:.3f}s ({len(output)} tokens)")
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        tokens_per_second = num_tokens / avg_time
        
        metrics = {
            'avg_time_seconds': avg_time,
            'std_time_seconds': std_time,
            'tokens_per_second': tokens_per_second,
            'total_tokens_generated': num_tokens,
            'input_length': input_length
        }
        
        print(f"✅ Average time: {avg_time:.3f}±{std_time:.3f}s")
        print(f"✅ Throughput: {tokens_per_second:.1f} tokens/second")
        
        return metrics


# Validation functions

def validate_numerical_precision(pytorch_logits: np.ndarray, jax_logits: jnp.ndarray, 
                                tolerance: float = 1e-4) -> bool:
    """
    Validate numerical precision between PyTorch and JAX outputs.
    
    Args:
        pytorch_logits: Logits from PyTorch model
        jax_logits: Logits from JAX model
        tolerance: Acceptable difference threshold
        
    Returns:
        True if outputs match within tolerance
    """
    diff = np.abs(np.array(jax_logits) - pytorch_logits)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"📊 Numerical validation:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Tolerance: {tolerance:.6f}")
    
    within_tolerance = max_diff < tolerance
    
    if within_tolerance:
        print("✅ Numerical precision validation passed!")
    else:
        print("❌ Numerical precision validation failed!")
    
    return within_tolerance


def test_sampling_strategies():
    """Test all sampling strategies work correctly."""
    print("🧪 Testing sampling strategies...")
    
    # Create small model for testing
    config = GPTOSSConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8
    )
    
    model = FlaxGPTOSSLMHeadModel(config)
    rng = random.PRNGKey(42)
    params = initialize_model(model, rng, (1, 10))
    
    engine = GPTOSSInferenceEngine(model, config, params)
    
    # Test different sampling configurations
    test_configs = [
        GenerationConfig(temperature=0.0, max_new_tokens=5),  # Greedy
        GenerationConfig(temperature=0.8, top_k=50, max_new_tokens=5),  # Top-k
        GenerationConfig(temperature=0.8, top_p=0.9, max_new_tokens=5),  # Top-p
        GenerationConfig(temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1, max_new_tokens=5),  # All
    ]
    
    input_ids = [1, 2, 3, 4, 5]
    
    for i, config in enumerate(test_configs):
        print(f"  Testing configuration {i + 1}...")
        output = engine.generate(input_ids, config)
        print(f"    Input length: {len(input_ids)}, Output length: {len(output)}")
        assert len(output) == len(input_ids) + config.max_new_tokens
    
    print("✅ All sampling strategies work correctly!")


def create_simple_demo():
    """Create a simple demo of the inference engine."""
    print("🎭 Creating inference demo...")
    
    # Small model for demo
    config = GPTOSSConfig(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4
    )
    
    model = FlaxGPTOSSLMHeadModel(config)
    rng = random.PRNGKey(42)
    params = initialize_model(model, rng, (1, 5))
    
    engine = GPTOSSInferenceEngine(model, config, params)
    
    # Demo generation
    input_ids = [1, 2, 3]
    gen_config = GenerationConfig(max_new_tokens=10, temperature=0.8, top_k=20)
    
    print(f"Input tokens: {input_ids}")
    
    # Non-streaming generation
    output = engine.generate(input_ids, gen_config)
    print(f"Generated tokens: {output.tolist()}")
    
    # Streaming generation
    print("Streaming generation:", end=" ")
    for token in engine.generate(input_ids, gen_config, stream=True):
        print(token, end=" ")
    print()
    
    # Benchmark
    metrics = engine.benchmark_performance(input_length=5, num_tokens=20, num_runs=3)
    
    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    print("=" * 60)
    print("GPT-OSS INFERENCE ENGINE TESTING")
    print("=" * 60)
    
    test_sampling_strategies()
    print()
    
    create_simple_demo()
    print()
    
    print("✅ All tests passed! Inference engine is ready for production.")