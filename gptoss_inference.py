#!/usr/bin/env python3
"""
GPT-OSS Inference Pipeline
Handles text generation with various sampling strategies.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any, Generator
import time

from gptoss_model import GPTOSSModel, GPTOSSConfig, KVCache

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    HAS_JAX = True
except ImportError:
    jnp = np
    HAS_JAX = False
    
    # NumPy random fallback
    class random:
        @staticmethod
        def PRNGKey(seed):
            return np.random.RandomState(seed)
        
        @staticmethod
        def uniform(key, shape):
            if hasattr(key, 'uniform'):
                return key.uniform(0, 1, shape)
            return np.random.uniform(0, 1, shape)


class InferencePipeline:
    """Production-ready inference pipeline for GPT-OSS models."""
    
    def __init__(self, model: GPTOSSModel, config: GPTOSSConfig):
        self.model = model
        self.config = config
        
        # Compile model for better performance if using JAX
        if HAS_JAX:
            self.forward_fn = jax.jit(self._forward)
        else:
            self.forward_fn = self._forward
        
        # Initialize random key
        self.rng_seed = int(time.time() * 1000) % (2**32)
        self.rng = random.PRNGKey(self.rng_seed)
    
    def _forward(self, input_ids, attention_mask=None, position_ids=None, kv_caches=None, use_cache=False):
        """Forward pass through the model."""
        return self.model(input_ids, attention_mask, position_ids, kv_caches, use_cache)
    
    def _sample_token(
        self,
        logits: jnp.ndarray,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        past_tokens: Optional[jnp.ndarray] = None
    ) -> int:
        """
        Sample next token from logits with various sampling strategies.
        
        Args:
            logits: Logits for next token (vocab_size,)
            temperature: Temperature for sampling
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            past_tokens: Previously generated tokens for repetition penalty
        
        Returns:
            Sampled token ID
        """
        # Apply repetition penalty
        if repetition_penalty != 1.0 and past_tokens is not None:
            for token_id in np.unique(past_tokens):
                if logits[token_id] > 0:
                    logits = logits.at[token_id].divide(repetition_penalty)
                else:
                    logits = logits.at[token_id].multiply(repetition_penalty)
        
        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        # Top-k sampling
        if top_k > 0:
            if HAS_JAX:
                top_k_logits, top_k_indices = jax.lax.top_k(logits, min(top_k, len(logits)))
            else:
                top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
                top_k_logits = logits[top_k_indices]
            
            # Create masked logits
            masked_logits = jnp.full_like(logits, -jnp.inf)
            masked_logits = masked_logits.at[top_k_indices].set(top_k_logits)
            logits = masked_logits
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_indices = jnp.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            
            # Convert to probabilities
            probs = jnp.exp(sorted_logits - jnp.max(sorted_logits))
            probs = probs / jnp.sum(probs)
            
            # Find cutoff
            cumsum = jnp.cumsum(probs)
            cutoff_idx = jnp.searchsorted(cumsum, top_p) + 1
            
            # Mask tokens beyond cutoff
            masked_logits = jnp.full_like(logits, -jnp.inf)
            masked_logits = masked_logits.at[sorted_indices[:cutoff_idx]].set(
                sorted_logits[:cutoff_idx]
            )
            logits = masked_logits
        
        # Sample from distribution
        if temperature == 0:
            # Greedy sampling
            next_token = jnp.argmax(logits)
        else:
            # Probabilistic sampling
            probs = jnp.exp(logits - jnp.max(logits))
            probs = probs / jnp.sum(probs)
            
            if HAS_JAX:
                self.rng, subkey = random.split(self.rng)
                next_token = random.categorical(subkey, logits)
            else:
                # NumPy sampling
                next_token = np.random.choice(len(probs), p=np.array(probs))
        
        return int(next_token)
    
    def generate(
        self,
        input_ids: Union[List[int], jnp.ndarray],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        stream: bool = False
    ) -> Union[jnp.ndarray, Generator]:
        """
        Generate text from input IDs.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeated tokens
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            stream: Whether to stream tokens as they're generated
        
        Returns:
            Generated token IDs (or generator if streaming)
        """
        # Convert input to jnp array
        if isinstance(input_ids, list):
            input_ids = jnp.array([input_ids], dtype=jnp.int32)
        elif len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        
        batch_size = input_ids.shape[0]
        generated = input_ids
        past_tokens = input_ids.flatten()
        
        # Initialize KV cache if using
        kv_caches = None
        if self.config.use_kv_cache:
            kv_caches = []
            for _ in range(self.config.num_hidden_layers):
                kv_caches.append(KVCache(
                    batch_size, 
                    self.config.max_position_embeddings,
                    self.config.num_key_value_heads,
                    self.config.head_dim
                ))
        
        def generate_tokens():
            nonlocal generated, past_tokens, kv_caches
            
            for step in range(max_new_tokens):
                # Get model predictions
                if kv_caches and step > 0:
                    # Use only the last token as input when using cache
                    input_slice = generated[:, -1:]
                    position_ids = jnp.array([[generated.shape[1] - 1]])
                else:
                    input_slice = generated
                    position_ids = None
                
                logits, kv_caches = self.forward_fn(
                    input_slice,
                    position_ids=position_ids,
                    kv_caches=kv_caches,
                    use_cache=self.config.use_kv_cache
                )
                
                # Get logits for next token
                next_logits = logits[0, -1, :]
                
                # Sample next token
                next_token = self._sample_token(
                    next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    past_tokens=past_tokens
                )
                
                # Add to generated sequence
                next_token_array = jnp.array([[next_token]], dtype=jnp.int32)
                generated = jnp.concatenate([generated, next_token_array], axis=1)
                past_tokens = jnp.append(past_tokens, next_token)
                
                if stream:
                    yield next_token
                
                # Check for EOS
                if eos_token_id is not None and next_token == eos_token_id:
                    break
        
        if stream:
            return generate_tokens()
        else:
            # Generate all tokens
            for _ in generate_tokens():
                pass
            return generated
    
    def batch_generate(
        self,
        input_ids_batch: List[List[int]],
        max_new_tokens: int = 100,
        **kwargs
    ) -> List[jnp.ndarray]:
        """
        Generate text for multiple inputs in parallel.
        
        Args:
            input_ids_batch: List of input token ID sequences
            max_new_tokens: Maximum tokens to generate
            **kwargs: Other generation parameters
        
        Returns:
            List of generated sequences
        """
        results = []
        
        # Process each input
        # TODO: Implement true batch processing for efficiency
        for input_ids in input_ids_batch:
            generated = self.generate(input_ids, max_new_tokens, **kwargs)
            results.append(generated)
        
        return results
    
    def compute_perplexity(
        self,
        input_ids: jnp.ndarray,
        target_ids: Optional[jnp.ndarray] = None
    ) -> float:
        """
        Compute perplexity for a sequence.
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs (defaults to shifted input_ids)
        
        Returns:
            Perplexity value
        """
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        
        if target_ids is None:
            # Use shifted input as target
            target_ids = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
        
        # Get logits
        logits, _ = self.forward_fn(input_ids, use_cache=False)
        
        # Compute cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)
        
        # Compute log probabilities
        log_probs = jnp.log_softmax(logits_flat, axis=-1)
        
        # Get log probs for target tokens
        target_log_probs = log_probs[jnp.arange(len(targets_flat)), targets_flat]
        
        # Compute perplexity
        avg_nll = -jnp.mean(target_log_probs)
        perplexity = jnp.exp(avg_nll)
        
        return float(perplexity)


def create_generation_config(**kwargs) -> Dict[str, Any]:
    """
    Create a generation configuration dictionary.
    
    Common parameters:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_k: Keep only top k tokens for sampling
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
    
    Returns:
        Configuration dictionary
    """
    defaults = {
        'max_new_tokens': 100,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'repetition_penalty': 1.0,
        'eos_token_id': None,
        'pad_token_id': None,
        'stream': False
    }
    
    config = defaults.copy()
    config.update(kwargs)
    
    return config


def benchmark_inference(
    pipeline: InferencePipeline,
    test_sequences: List[List[int]],
    max_new_tokens: int = 50
) -> Dict[str, float]:
    """
    Benchmark inference performance.
    
    Args:
        pipeline: InferencePipeline instance
        test_sequences: List of test input sequences
        max_new_tokens: Tokens to generate per sequence
    
    Returns:
        Performance metrics
    """
    print("⚡ Benchmarking inference performance...")
    
    total_tokens = 0
    total_time = 0
    
    for i, input_ids in enumerate(test_sequences):
        start_time = time.time()
        
        generated = pipeline.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        tokens_generated = generated.shape[1] - len(input_ids)
        total_tokens += tokens_generated
        total_time += elapsed
        
        print(f"  Sequence {i+1}: {tokens_generated} tokens in {elapsed:.2f}s")
    
    # Calculate metrics
    metrics = {
        'total_tokens': total_tokens,
        'total_time': total_time,
        'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
        'avg_time_per_sequence': total_time / len(test_sequences),
        'avg_tokens_per_sequence': total_tokens / len(test_sequences)
    }
    
    print(f"\n📊 Benchmark Results:")
    print(f"  • Total tokens generated: {metrics['total_tokens']}")
    print(f"  • Total time: {metrics['total_time']:.2f}s")
    print(f"  • Throughput: {metrics['tokens_per_second']:.1f} tokens/s")
    print(f"  • Avg time per sequence: {metrics['avg_time_per_sequence']:.2f}s")
    print(f"  • Avg tokens per sequence: {metrics['avg_tokens_per_sequence']:.1f}")
    
    return metrics