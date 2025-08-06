#!/usr/bin/env python3
"""
Complete inference and text generation engine for Flax GPT-2.
Implements autoregressive generation with multiple sampling strategies.
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import numpy as np
from functools import partial
from dataclasses import dataclass

from flax_gpt2_model import FlaxGPT2Config, FlaxGPT2LMHeadModel


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    seed: Optional[int] = None


class FlaxTextGenerator:
    """High-performance text generation engine for Flax GPT-2."""
    
    def __init__(self, 
                 model: FlaxGPT2LMHeadModel,
                 params: Dict[str, Any],
                 config: FlaxGPT2Config):
        self.model = model
        self.params = params
        self.config = config
        
        # JIT compile model application only
        self._apply_model = jax.jit(self._apply_model_impl)

    def _apply_model_impl(self, 
                         params: Dict[str, Any], 
                         input_ids: jnp.ndarray) -> jnp.ndarray:
        """Apply model to get logits."""
        return self.model.apply(params, input_ids, deterministic=True)

    def _generate_step_impl(self,
                           params: Dict[str, Any],
                           input_ids: jnp.ndarray,
                           rng_key: jax.random.PRNGKey,
                           temperature: float = 1.0,
                           top_k: Optional[int] = None,
                           top_p: Optional[float] = None,
                           repetition_penalty: float = 1.0,
                           do_sample: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single generation step."""
        # Get logits for the last token
        logits = self._apply_model(params, input_ids)
        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        
        # Apply generation parameters
        next_token_logits = self._apply_generation_params(
            next_token_logits, input_ids, temperature, top_k, top_p, repetition_penalty
        )
        
        # Sample next token
        if do_sample and temperature > 0.0:
            next_token = self._sample_token(next_token_logits, rng_key)
        else:
            next_token = jnp.argmax(next_token_logits, axis=-1)
        
        # Append to sequence
        next_token = next_token[:, None]  # (batch_size, 1)
        new_input_ids = jnp.concatenate([input_ids, next_token], axis=1)
        
        return new_input_ids, next_token_logits

    def _apply_generation_params(self,
                                logits: jnp.ndarray,
                                input_ids: jnp.ndarray,
                                temperature: float = 1.0,
                                top_k: Optional[int] = None,
                                top_p: Optional[float] = None,
                                repetition_penalty: float = 1.0) -> jnp.ndarray:
        """Apply temperature, top-k, top-p, and repetition penalty."""
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, input_ids, repetition_penalty)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            logits = self._apply_top_k_filtering(logits, top_k)
        
        # Apply top-p (nucleus) filtering
        if top_p is not None:
            logits = self._apply_top_p_filtering(logits, top_p)
        
        return logits

    def _apply_repetition_penalty(self,
                                 logits: jnp.ndarray,
                                 input_ids: jnp.ndarray,
                                 penalty: float) -> jnp.ndarray:
        """Apply repetition penalty to reduce repetitive tokens."""
        batch_size, vocab_size = logits.shape
        seq_len = input_ids.shape[1]
        
        # Create penalty mask for tokens that have appeared
        penalty_mask = jnp.zeros((batch_size, vocab_size))
        
        # For each batch item
        for i in range(batch_size):
            unique_tokens = jnp.unique(input_ids[i])
            penalty_mask = penalty_mask.at[i, unique_tokens].set(1.0)
        
        # Apply penalty (decrease logits for repeated tokens)
        penalty_factor = jnp.where(penalty_mask, penalty, 1.0)
        adjusted_logits = jnp.where(
            logits > 0,
            logits / penalty_factor,
            logits * penalty_factor
        )
        
        return adjusted_logits

    def _apply_top_k_filtering(self, logits: jnp.ndarray, top_k: int) -> jnp.ndarray:
        """Keep only the top-k highest probability tokens."""
        # Get top-k values and indices
        top_k_logits, top_k_indices = lax.top_k(logits, top_k)
        
        # Create mask for top-k tokens
        mask = jnp.full_like(logits, -jnp.inf)
        mask = mask.at[jnp.arange(logits.shape[0])[:, None], top_k_indices].set(top_k_logits)
        
        return mask

    def _apply_top_p_filtering(self, logits: jnp.ndarray, top_p: float) -> jnp.ndarray:
        """Keep tokens with cumulative probability up to top_p (nucleus sampling)."""
        # Sort logits in descending order
        sorted_indices = jnp.argsort(logits, axis=-1)[..., ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        
        # Compute cumulative probabilities
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        
        # Find cutoff point
        cutoff_mask = cumulative_probs > top_p
        
        # Set logits to -inf for tokens beyond cutoff
        sorted_logits = jnp.where(cutoff_mask, -jnp.inf, sorted_logits)
        
        # Scatter back to original order
        filtered_logits = jnp.full_like(logits, -jnp.inf)
        filtered_logits = filtered_logits.at[
            jnp.arange(logits.shape[0])[:, None], sorted_indices
        ].set(sorted_logits)
        
        return filtered_logits

    def _sample_token(self,
                     logits: jnp.ndarray,
                     rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample next token from logits."""
        return jax.random.categorical(rng_key, logits)

    def generate(self,
                input_ids: Union[np.ndarray, jnp.ndarray, List[int]],
                generation_config: Optional[GenerationConfig] = None,
                rng_key: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Generate text from input prompt.
        
        Args:
            input_ids: Input token IDs (can be 1D or 2D)
            generation_config: Generation parameters
            rng_key: Random key for sampling
            
        Returns:
            Tuple of (generated_ids, generation_info)
        """
        # Default configuration
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Default random key
        if rng_key is None:
            seed = generation_config.seed or 0
            rng_key = jax.random.PRNGKey(seed)
        
        # Convert input to JAX array and ensure 2D
        if isinstance(input_ids, list):
            input_ids = jnp.array(input_ids)
        if isinstance(input_ids, np.ndarray):
            input_ids = jnp.array(input_ids)
        
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]  # Add batch dimension
        
        batch_size, input_length = input_ids.shape
        
        # Check sequence length
        max_length = input_length + generation_config.max_new_tokens
        if max_length > self.config.max_position_embeddings:
            print(f"Warning: Generated sequence length ({max_length}) exceeds model's max length ({self.config.max_position_embeddings})")
        
        # Generation loop
        current_ids = input_ids
        generated_tokens = []
        all_logits = []
        
        for step in range(generation_config.max_new_tokens):
            # Generate next token
            rng_key, step_key = jax.random.split(rng_key)
            current_ids, step_logits = self._generate_step_impl(
                self.params, current_ids, step_key,
                generation_config.temperature,
                generation_config.top_k,
                generation_config.top_p,
                generation_config.repetition_penalty,
                generation_config.do_sample
            )
            
            # Store information
            new_token = current_ids[:, -1]
            generated_tokens.append(new_token)
            all_logits.append(step_logits)
            
            # Check for early stopping
            if generation_config.eos_token_id is not None:
                if jnp.any(new_token == generation_config.eos_token_id):
                    break
        
        # Prepare output
        generated_ids = current_ids
        generation_info = {
            'input_length': input_length,
            'generated_length': generated_ids.shape[1] - input_length,
            'total_length': generated_ids.shape[1],
            'all_logits': jnp.stack(all_logits) if all_logits else None,
            'generated_tokens': jnp.stack(generated_tokens) if generated_tokens else None,
        }
        
        return generated_ids, generation_info

    def generate_batch_simple(self,
                             params: Dict[str, Any],
                             input_ids: jnp.ndarray,
                             max_new_tokens: int,
                             rng_key: jax.random.PRNGKey,
                             temperature: float = 1.0) -> jnp.ndarray:
        """
        Simple batch generation without JAX scan (for better compatibility).
        """
        current_ids = input_ids
        rng = rng_key
        
        for _ in range(max_new_tokens):
            rng, step_rng = jax.random.split(rng)
            
            # Get logits
            logits = self.model.apply(params, current_ids, deterministic=True)
            next_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            next_token = jax.random.categorical(step_rng, next_logits)
            next_token = next_token[:, None]
            
            # Append to sequence
            current_ids = jnp.concatenate([current_ids, next_token], axis=1)
        
        return current_ids


class TextGenerationPipeline:
    """High-level pipeline for text generation with tokenization."""
    
    def __init__(self,
                 model: FlaxGPT2LMHeadModel,
                 params: Dict[str, Any],
                 config: FlaxGPT2Config,
                 tokenizer: Optional[Any] = None):
        self.generator = FlaxTextGenerator(model, params, config)
        self.tokenizer = tokenizer
        self.config = config
        
        # Load tokenizer if not provided
        if self.tokenizer is None:
            self._load_default_tokenizer()

    def _load_default_tokenizer(self):
        """Load default GPT-2 tokenizer."""
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            print("Warning: transformers not available, tokenization disabled")
            self.tokenizer = None

    def generate_text(self,
                     prompt: str,
                     generation_config: Optional[GenerationConfig] = None,
                     return_full_text: bool = True) -> str:
        """
        Generate text from a string prompt.
        
        Args:
            prompt: Input text prompt
            generation_config: Generation parameters
            return_full_text: Whether to return prompt + generated text
            
        Returns:
            Generated text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available for text generation")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='np')
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        
        # Generate
        generated_ids, info = self.generator.generate(
            input_ids, generation_config
        )
        
        # Decode output
        if return_full_text:
            output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            # Only return newly generated tokens
            new_tokens = generated_ids[0, input_ids.shape[1]:]
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return output_text

    def generate_batch(self,
                      prompts: List[str],
                      generation_config: Optional[GenerationConfig] = None,
                      return_full_text: bool = True) -> List[str]:
        """Generate text for multiple prompts."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available for text generation")
        
        # Tokenize all prompts
        all_input_ids = []
        max_input_len = 0
        
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt)
            all_input_ids.append(input_ids)
            max_input_len = max(max_input_len, len(input_ids))
        
        # Pad to same length
        padded_ids = []
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        for input_ids in all_input_ids:
            padding_length = max_input_len - len(input_ids)
            padded = input_ids + [pad_token_id] * padding_length
            padded_ids.append(padded)
        
        batch_input_ids = jnp.array(padded_ids)
        
        # Generate
        generated_ids, info = self.generator.generate(
            batch_input_ids, generation_config
        )
        
        # Decode outputs
        outputs = []
        for i, prompt in enumerate(prompts):
            if return_full_text:
                output_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            else:
                original_length = len(all_input_ids[i])
                new_tokens = generated_ids[i, original_length:]
                output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(output_text)
        
        return outputs


# Utility functions

def create_generation_pipeline(model_path: str) -> TextGenerationPipeline:
    """Create a complete text generation pipeline from a model path."""
    from pytorch_to_flax_converter import load_converted_model
    from flax_gpt2_model import create_model
    
    # Load converted model
    flax_params, flax_config = load_converted_model(model_path)
    
    # Create model
    model = create_model(flax_config)
    
    # Create pipeline
    pipeline = TextGenerationPipeline(model, flax_params, flax_config)
    
    return pipeline


def benchmark_generation(pipeline: TextGenerationPipeline,
                        prompt: str = "The quick brown fox",
                        num_tokens: int = 50,
                        num_runs: int = 5) -> Dict[str, float]:
    """Benchmark text generation performance."""
    import time
    
    gen_config = GenerationConfig(
        max_new_tokens=num_tokens,
        do_sample=True,
        temperature=1.0
    )
    
    # Warmup
    pipeline.generate_text(prompt, gen_config)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        output = pipeline.generate_text(prompt, gen_config)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    tokens_per_second = num_tokens / avg_time
    
    return {
        'avg_time_seconds': avg_time,
        'tokens_per_second': tokens_per_second,
        'std_time_seconds': np.std(times),
        'num_runs': num_runs,
        'num_tokens': num_tokens
    }


if __name__ == "__main__":
    # Example usage
    print("Text Generation Engine Example")
    print("=" * 40)
    
    # This would work with a real converted model
    try:
        # Create a simple test
        from flax_gpt2_model import FlaxGPT2Config, create_model, init_model_params
        
        config = FlaxGPT2Config(
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12
        )
        
        model = create_model(config)
        rng = jax.random.PRNGKey(0)
        params = init_model_params(model, rng, (1, 10))
        
        generator = FlaxTextGenerator(model, params, config)
        
        # Test generation
        test_input = jnp.array([[1, 2, 3, 4, 5]])
        gen_config = GenerationConfig(max_new_tokens=5, temperature=1.0)
        
        output, info = generator.generate(test_input, gen_config)
        
        print(f"✅ Generation test successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Generated length: {info['generated_length']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()