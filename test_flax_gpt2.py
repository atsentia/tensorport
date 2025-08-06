#!/usr/bin/env python3
"""
Comprehensive test suite for the Flax GPT-2 implementation.
Tests model architecture, weight conversion, and text generation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Dict, Any, Tuple

# Import our modules
from flax_gpt2_model import (
    FlaxGPT2Config, FlaxGPT2LMHeadModel, FlaxAttention, FlaxMLP, 
    FlaxTransformerBlock, create_model, init_model_params, model_forward
)
from text_generation import (
    FlaxTextGenerator, GenerationConfig, TextGenerationPipeline
)


class TestFlaxGPT2Config:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FlaxGPT2Config()
        assert config.vocab_size == 50257
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.head_dim == 64  # 768 // 12
        assert config.intermediate_size == 3072  # 4 * 768

    def test_custom_config(self):
        """Test custom configuration."""
        config = FlaxGPT2Config(
            vocab_size=30000,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8
        )
        assert config.vocab_size == 30000
        assert config.hidden_size == 512
        assert config.head_dim == 64  # 512 // 8
        assert config.intermediate_size == 2048  # 4 * 512

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'vocab_size': 25000,
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'max_position_embeddings': 2048
        }
        config = FlaxGPT2Config.from_dict(config_dict)
        assert config.vocab_size == 25000
        assert config.hidden_size == 1024
        assert config.max_position_embeddings == 2048


class TestFlaxModelComponents:
    """Test individual model components."""

    @pytest.fixture
    def config(self):
        """Small test configuration."""
        return FlaxGPT2Config(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64
        )

    def test_attention_module(self, config):
        """Test attention module."""
        rng = jax.random.PRNGKey(0)
        attention = FlaxAttention(config)
        
        # Initialize parameters
        batch_size, seq_len = 2, 10
        hidden_states = jnp.ones((batch_size, seq_len, config.hidden_size))
        params = attention.init(rng, hidden_states)
        
        # Test forward pass
        output = attention.apply(params, hidden_states)
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert output.dtype == jnp.float32

    def test_mlp_module(self, config):
        """Test MLP module."""
        rng = jax.random.PRNGKey(0)
        mlp = FlaxMLP(config)
        
        # Initialize parameters
        batch_size, seq_len = 2, 10
        hidden_states = jnp.ones((batch_size, seq_len, config.hidden_size))
        params = mlp.init(rng, hidden_states)
        
        # Test forward pass
        output = mlp.apply(params, hidden_states)
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert output.dtype == jnp.float32

    def test_transformer_block(self, config):
        """Test transformer block."""
        rng = jax.random.PRNGKey(0)
        block = FlaxTransformerBlock(config)
        
        # Initialize parameters
        batch_size, seq_len = 2, 10
        hidden_states = jnp.ones((batch_size, seq_len, config.hidden_size))
        params = block.init(rng, hidden_states)
        
        # Test forward pass
        output = block.apply(params, hidden_states)
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert output.dtype == jnp.float32

    def test_full_model(self, config):
        """Test full model."""
        model = create_model(config)
        rng = jax.random.PRNGKey(0)
        
        # Initialize parameters
        batch_size, seq_len = 2, 10
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        params = model.init(rng, input_ids)
        
        # Test forward pass
        logits = model.apply(params, input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert logits.dtype == jnp.float32

    def test_model_with_attention_mask(self, config):
        """Test model with attention mask."""
        model = create_model(config)
        rng = jax.random.PRNGKey(0)
        
        batch_size, seq_len = 2, 10
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        attention_mask = jnp.ones((batch_size, seq_len))
        
        params = model.init(rng, input_ids, attention_mask)
        logits = model.apply(params, input_ids, attention_mask)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_parameter_count(self, config):
        """Test parameter counting."""
        model = create_model(config)
        rng = jax.random.PRNGKey(0)
        params = init_model_params(model, rng, (1, 10))
        
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        
        # Expected parameters for small model
        # Embeddings: vocab_size * hidden_size + max_pos * hidden_size
        # Transformer: layers * (4 * hidden_size^2 + intermediate * hidden_size + biases)
        # Layer norms: layers * 2 * hidden_size + hidden_size (final)
        expected_min = config.vocab_size * config.hidden_size  # At least embeddings
        assert param_count > expected_min


class TestTextGeneration:
    """Test text generation functionality."""

    @pytest.fixture
    def small_model_setup(self):
        """Setup small model for testing."""
        config = FlaxGPT2Config(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32
        )
        model = create_model(config)
        rng = jax.random.PRNGKey(42)
        params = init_model_params(model, rng, (1, 10))
        
        return model, params, config

    def test_generation_config(self):
        """Test generation configuration."""
        config = GenerationConfig(
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        assert config.max_new_tokens == 20
        assert config.temperature == 0.8
        assert config.top_k == 50
        assert config.top_p == 0.9

    def test_text_generator_creation(self, small_model_setup):
        """Test text generator creation."""
        model, params, config = small_model_setup
        generator = FlaxTextGenerator(model, params, config)
        
        assert generator.model == model
        assert generator.params == params
        assert generator.config == config

    def test_single_generation_step(self, small_model_setup):
        """Test single generation step."""
        model, params, config = small_model_setup
        generator = FlaxTextGenerator(model, params, config)
        
        batch_size, seq_len = 1, 5
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        rng_key = jax.random.PRNGKey(0)
        
        new_ids, logits = generator._generate_step_impl(
            params, input_ids, rng_key, temperature=1.0
        )
        
        assert new_ids.shape == (batch_size, seq_len + 1)
        assert logits.shape == (batch_size, config.vocab_size)

    def test_complete_generation(self, small_model_setup):
        """Test complete text generation."""
        model, params, config = small_model_setup
        generator = FlaxTextGenerator(model, params, config)
        
        input_ids = jnp.array([[1, 2, 3]])
        gen_config = GenerationConfig(max_new_tokens=5, temperature=1.0)
        
        output_ids, info = generator.generate(input_ids, gen_config)
        
        assert output_ids.shape[0] == 1  # batch size
        assert output_ids.shape[1] == 3 + 5  # input + generated
        assert info['input_length'] == 3
        assert info['generated_length'] == 5

    def test_batch_generation(self, small_model_setup):
        """Test batch generation."""
        model, params, config = small_model_setup
        generator = FlaxTextGenerator(model, params, config)
        
        batch_size = 3
        input_ids = jnp.array([[1, 2], [3, 4], [5, 6]])
        rng_key = jax.random.PRNGKey(0)
        
        output_ids = generator.generate_batch_simple(
            params, input_ids, max_new_tokens=3, rng_key=rng_key
        )
        
        assert output_ids.shape == (batch_size, 2 + 3)

    def test_different_sampling_strategies(self, small_model_setup):
        """Test different sampling strategies."""
        model, params, config = small_model_setup
        generator = FlaxTextGenerator(model, params, config)
        
        input_ids = jnp.array([[1, 2, 3]])
        
        # Greedy (temperature = 0)
        gen_config_greedy = GenerationConfig(max_new_tokens=3, temperature=0.0, do_sample=False)
        output_greedy, _ = generator.generate(input_ids, gen_config_greedy)
        
        # High temperature sampling
        gen_config_random = GenerationConfig(max_new_tokens=3, temperature=2.0, do_sample=True)
        output_random, _ = generator.generate(input_ids, gen_config_random)
        
        # Top-k sampling
        gen_config_topk = GenerationConfig(max_new_tokens=3, top_k=10, do_sample=True)
        output_topk, _ = generator.generate(input_ids, gen_config_topk)
        
        # All should have same output shape
        expected_shape = (1, 6)  # batch=1, seq_len=3+3
        assert output_greedy.shape == expected_shape
        assert output_random.shape == expected_shape
        assert output_topk.shape == expected_shape

    def test_generation_with_eos_token(self, small_model_setup):
        """Test generation with early stopping."""
        model, params, config = small_model_setup
        generator = FlaxTextGenerator(model, params, config)
        
        input_ids = jnp.array([[1, 2, 3]])
        gen_config = GenerationConfig(
            max_new_tokens=10, 
            eos_token_id=50,  # Assuming this token exists
            temperature=1.0
        )
        
        # Note: This test might not actually hit EOS with random model
        output_ids, info = generator.generate(input_ids, gen_config)
        
        # Should generate up to max_new_tokens if EOS not hit
        assert output_ids.shape[1] <= 3 + 10


class TestModelSerialization:
    """Test model saving and loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_config_serialization(self, temp_dir):
        """Test configuration saving and loading."""
        config = FlaxGPT2Config(
            vocab_size=30000,
            hidden_size=512,
            num_hidden_layers=6
        )
        
        # Save config
        config_file = temp_dir / "config.json"
        config_dict = {
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'num_hidden_layers': config.num_hidden_layers,
            'num_attention_heads': config.num_attention_heads,
            'head_dim': config.head_dim,
            'intermediate_size': config.intermediate_size,
            'hidden_act': config.hidden_act,
            'max_position_embeddings': config.max_position_embeddings,
            'layer_norm_epsilon': config.layer_norm_epsilon,
            'tie_word_embeddings': config.tie_word_embeddings
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f)
        
        # Load config
        with open(config_file) as f:
            loaded_dict = json.load(f)
        
        loaded_config = FlaxGPT2Config(**loaded_dict)
        
        assert loaded_config.vocab_size == config.vocab_size
        assert loaded_config.hidden_size == config.hidden_size
        assert loaded_config.num_hidden_layers == config.num_hidden_layers

    def test_parameter_serialization(self, temp_dir):
        """Test parameter saving and loading."""
        config = FlaxGPT2Config(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4
        )
        model = create_model(config)
        rng = jax.random.PRNGKey(0)
        params = init_model_params(model, rng, (1, 10))
        
        # Save parameters
        from flax import serialization
        params_file = temp_dir / "params.msgpack"
        with open(params_file, 'wb') as f:
            f.write(serialization.to_bytes(params))
        
        # Load parameters
        with open(params_file, 'rb') as f:
            loaded_params = serialization.from_bytes(None, f.read())
        
        # Compare parameter trees
        def compare_trees(tree1, tree2):
            leaves1 = jax.tree_util.tree_leaves(tree1)
            leaves2 = jax.tree_util.tree_leaves(tree2)
            
            assert len(leaves1) == len(leaves2)
            for leaf1, leaf2 in zip(leaves1, leaves2):
                assert leaf1.shape == leaf2.shape
                assert jnp.allclose(leaf1, leaf2)
        
        compare_trees(params, loaded_params)


class TestNumericalStability:
    """Test numerical properties and edge cases."""

    @pytest.fixture
    def config(self):
        return FlaxGPT2Config(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32
        )

    def test_no_nan_or_inf(self, config):
        """Test that model doesn't produce NaN or Inf values."""
        model = create_model(config)
        rng = jax.random.PRNGKey(0)
        params = init_model_params(model, rng, (1, 10))
        
        # Test with normal input
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        logits = model.apply(params, input_ids)
        
        assert jnp.isfinite(logits).all(), "Model produced NaN or Inf values"

    def test_gradient_computation(self, config):
        """Test that gradients can be computed."""
        model = create_model(config)
        rng = jax.random.PRNGKey(0)
        params = init_model_params(model, rng, (1, 10))
        
        def loss_fn(params, input_ids, targets):
            logits = model.apply(params, input_ids)
            # Simple cross-entropy loss
            return jnp.mean(jnp.sum(-jax.nn.log_softmax(logits) * jax.nn.one_hot(targets, config.vocab_size), axis=-1))
        
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        targets = jnp.array([[2, 3, 4, 5, 6]])
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, input_ids, targets)
        
        # Check that gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(grads)
        for grad in grad_leaves:
            assert jnp.isfinite(grad).all(), "Gradients contain NaN or Inf"

    def test_different_sequence_lengths(self, config):
        """Test model with different sequence lengths."""
        model = create_model(config)
        rng = jax.random.PRNGKey(0)
        params = init_model_params(model, rng, (1, 10))
        
        for seq_len in [1, 5, 10, 20]:
            if seq_len <= config.max_position_embeddings:
                input_ids = jnp.ones((1, seq_len), dtype=jnp.int32)
                logits = model.apply(params, input_ids)
                
                assert logits.shape == (1, seq_len, config.vocab_size)
                assert jnp.isfinite(logits).all()

    def test_large_vocabulary_indices(self, config):
        """Test with vocabulary indices near the boundary."""
        model = create_model(config)
        rng = jax.random.PRNGKey(0)
        params = init_model_params(model, rng, (1, 5))
        
        # Test with valid indices
        input_ids = jnp.array([[0, 1, config.vocab_size - 2, config.vocab_size - 1, 50]])
        logits = model.apply(params, input_ids)
        
        assert jnp.isfinite(logits).all()


class TestPerformance:
    """Test performance characteristics."""

    @pytest.fixture
    def medium_config(self):
        return FlaxGPT2Config(
            vocab_size=5000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            max_position_embeddings=128
        )

    def test_jit_compilation(self, medium_config):
        """Test that model can be JIT compiled."""
        model = create_model(medium_config)
        rng = jax.random.PRNGKey(0)
        params = init_model_params(model, rng, (1, 10))
        
        @jax.jit
        def jitted_forward(params, input_ids):
            return model.apply(params, input_ids)
        
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        
        # First call (compilation + execution)
        logits1 = jitted_forward(params, input_ids)
        
        # Second call (execution only)
        logits2 = jitted_forward(params, input_ids)
        
        # Results should be identical
        assert jnp.allclose(logits1, logits2)

    def test_batch_processing(self, medium_config):
        """Test efficient batch processing."""
        model = create_model(medium_config)
        rng = jax.random.PRNGKey(0)
        params = init_model_params(model, rng, (1, 10))
        
        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            input_ids = jnp.ones((batch_size, 10), dtype=jnp.int32)
            logits = model.apply(params, input_ids)
            
            assert logits.shape == (batch_size, 10, medium_config.vocab_size)
            assert jnp.isfinite(logits).all()


# Test runner
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])