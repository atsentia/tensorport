#!/usr/bin/env python3
"""
GPT-OSS Model Validation Suite
Comprehensive testing for numerical stability and correctness.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import json
from pathlib import Path

from gptoss_model import GPTOSSModel, GPTOSSConfig, dequantize_mxfp4, RMSNorm, RotaryEmbedding
from gptoss_inference import InferencePipeline

# JAX imports with fallback
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = np
    HAS_JAX = False


@dataclass
class ValidationResult:
    """Results from a validation test."""
    test_name: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'max_abs_error': float(self.max_abs_error),
            'max_rel_error': float(self.max_rel_error),
            'mean_abs_error': float(self.mean_abs_error),
            'details': self.details
        }


class ModelValidator:
    """Comprehensive validation framework for GPT-OSS models."""
    
    def __init__(self, rtol: float = 1e-4, atol: float = 1e-6):
        self.rtol = rtol
        self.atol = atol
        self.results: List[ValidationResult] = []
    
    def test_mxfp4_dequantization(self) -> ValidationResult:
        """Test MXFP4 dequantization accuracy."""
        print("🔧 Testing MXFP4 dequantization...")
        
        np.random.seed(42)
        rows, cols = 128, 64
        
        # Create test data
        original_weights = np.random.randn(rows, cols * 2).astype(np.float32)
        
        # Quantize
        scales = np.max(np.abs(original_weights), axis=1, keepdims=False) * 7.0
        quantized_int = np.round(original_weights / scales[:, np.newaxis] * 7.0).astype(np.int8)
        quantized_int = np.clip(quantized_int, -7, 7)
        
        # Pack into 4-bit format
        even_cols = quantized_int[:, ::2]
        odd_cols = quantized_int[:, 1::2]
        
        # Convert to unsigned and pack
        even_unsigned = (even_cols + 8).astype(np.uint8)
        odd_unsigned = (odd_cols + 8).astype(np.uint8)
        blocks = (even_unsigned << 4) | odd_unsigned
        
        try:
            # Dequantize
            dequantized = dequantize_mxfp4(blocks, scales)
            
            # Check shape
            if dequantized.shape != original_weights.shape:
                return ValidationResult(
                    test_name="MXFP4 Dequantization",
                    passed=False,
                    max_abs_error=float('inf'),
                    max_rel_error=float('inf'),
                    mean_abs_error=float('inf'),
                    details={"error": f"Shape mismatch: {dequantized.shape} vs {original_weights.shape}"}
                )
            
            # Compute errors
            abs_diff = np.abs(dequantized - original_weights)
            max_abs_error = np.max(abs_diff)
            mean_abs_error = np.mean(abs_diff)
            
            # Relative error
            rel_diff = abs_diff / (np.abs(original_weights) + self.atol)
            max_rel_error = np.max(rel_diff)
            
            # Quantization error is expected, so use relaxed threshold
            passed = max_rel_error < 0.2  # 20% error acceptable for 4-bit quantization
            
            result = ValidationResult(
                test_name="MXFP4 Dequantization",
                passed=passed,
                max_abs_error=max_abs_error,
                max_rel_error=max_rel_error,
                mean_abs_error=mean_abs_error,
                details={
                    "quantization_bits": 4,
                    "test_shape": original_weights.shape,
                    "expected_error": "~10-20% for 4-bit quantization"
                }
            )
            
            print(f"  {'✅' if passed else '❌'} Max relative error: {max_rel_error:.2%}")
            
        except Exception as e:
            result = ValidationResult(
                test_name="MXFP4 Dequantization",
                passed=False,
                max_abs_error=float('inf'),
                max_rel_error=float('inf'),
                mean_abs_error=float('inf'),
                details={"error": str(e)}
            )
            print(f"  ❌ Test failed: {e}")
        
        self.results.append(result)
        return result
    
    def test_numerical_stability(self) -> ValidationResult:
        """Test numerical stability of model components."""
        print("🧮 Testing numerical stability...")
        
        all_stable = True
        details = {}
        
        # Test RMSNorm stability
        print("  Testing RMSNorm...")
        test_cases = [
            ("normal", jnp.ones((2, 10, 128))),
            ("small", jnp.ones((2, 10, 128)) * 1e-6),
            ("large", jnp.ones((2, 10, 128)) * 1e6),
            ("zeros", jnp.zeros((2, 10, 128))),
        ]
        
        weight = jnp.ones(128)
        rms_norm = RMSNorm(weight, eps=1e-6)
        
        for name, x in test_cases:
            try:
                output = rms_norm(x)
                if jnp.any(jnp.isnan(output)) or jnp.any(jnp.isinf(output)):
                    details[f"rms_norm_{name}"] = "unstable"
                    all_stable = False
                    print(f"    ❌ RMSNorm unstable for {name}")
                else:
                    details[f"rms_norm_{name}"] = "stable"
                    print(f"    ✅ RMSNorm stable for {name}")
            except Exception as e:
                details[f"rms_norm_{name}"] = f"error: {e}"
                all_stable = False
                print(f"    ❌ RMSNorm failed for {name}: {e}")
        
        # Test RoPE stability
        print("  Testing RotaryEmbedding...")
        rope = RotaryEmbedding(dim=64, max_position_embeddings=2048, theta=10000.0)
        
        position_cases = [
            ("start", jnp.array([[0, 1, 2, 3]])),
            ("middle", jnp.array([[1000, 1001, 1002]])),
            ("end", jnp.array([[2045, 2046, 2047]])),
        ]
        
        for name, pos_ids in position_cases:
            try:
                x = jnp.ones((1, pos_ids.shape[1], 16, 64))
                output = rope(x, pos_ids)
                if jnp.any(jnp.isnan(output)) or jnp.any(jnp.isinf(output)):
                    details[f"rope_{name}"] = "unstable"
                    all_stable = False
                    print(f"    ❌ RoPE unstable for {name}")
                else:
                    details[f"rope_{name}"] = "stable"
                    print(f"    ✅ RoPE stable for {name}")
            except Exception as e:
                details[f"rope_{name}"] = f"error: {e}"
                all_stable = False
                print(f"    ❌ RoPE failed for {name}: {e}")
        
        result = ValidationResult(
            test_name="Numerical Stability",
            passed=all_stable,
            max_abs_error=0.0,
            max_rel_error=0.0,
            mean_abs_error=0.0,
            details=details
        )
        
        self.results.append(result)
        return result
    
    def test_model_output(self, model: GPTOSSModel, config: GPTOSSConfig) -> ValidationResult:
        """Test model output validity."""
        print("🔍 Testing model output...")
        
        try:
            # Create test input
            batch_size, seq_len = 2, 10
            input_ids = jnp.array(np.random.randint(0, 1000, (batch_size, seq_len)), dtype=jnp.int32)
            
            # Forward pass
            logits, _ = model(input_ids, use_cache=False)
            
            # Check output shape
            expected_shape = (batch_size, seq_len, config.vocab_size)
            if logits.shape != expected_shape:
                return ValidationResult(
                    test_name="Model Output",
                    passed=False,
                    max_abs_error=float('inf'),
                    max_rel_error=float('inf'),
                    mean_abs_error=float('inf'),
                    details={"error": f"Shape mismatch: {logits.shape} vs {expected_shape}"}
                )
            
            # Check for NaN/Inf
            has_nan = jnp.any(jnp.isnan(logits))
            has_inf = jnp.any(jnp.isinf(logits))
            
            if has_nan or has_inf:
                return ValidationResult(
                    test_name="Model Output",
                    passed=False,
                    max_abs_error=float('inf'),
                    max_rel_error=float('inf'),
                    mean_abs_error=float('inf'),
                    details={"has_nan": bool(has_nan), "has_inf": bool(has_inf)}
                )
            
            # Check value ranges
            logits_np = np.array(logits)
            details = {
                "output_shape": logits.shape,
                "min_value": float(np.min(logits_np)),
                "max_value": float(np.max(logits_np)),
                "mean_value": float(np.mean(logits_np)),
                "std_value": float(np.std(logits_np))
            }
            
            # Basic sanity checks
            passed = (
                np.isfinite(details["min_value"]) and
                np.isfinite(details["max_value"]) and
                np.isfinite(details["mean_value"]) and
                details["std_value"] > 0  # Should have some variation
            )
            
            result = ValidationResult(
                test_name="Model Output",
                passed=passed,
                max_abs_error=0.0,
                max_rel_error=0.0,
                mean_abs_error=0.0,
                details=details
            )
            
            print(f"  {'✅' if passed else '❌'} Output range: [{details['min_value']:.2f}, {details['max_value']:.2f}]")
            print(f"  Mean: {details['mean_value']:.2f}, Std: {details['std_value']:.2f}")
            
        except Exception as e:
            result = ValidationResult(
                test_name="Model Output",
                passed=False,
                max_abs_error=float('inf'),
                max_rel_error=float('inf'),
                mean_abs_error=float('inf'),
                details={"error": str(e)}
            )
            print(f"  ❌ Test failed: {e}")
        
        self.results.append(result)
        return result
    
    def test_generation(self, pipeline: InferencePipeline) -> ValidationResult:
        """Test text generation capability."""
        print("🎯 Testing generation...")
        
        try:
            # Test input
            input_ids = [1, 2, 3, 4, 5]  # Dummy tokens
            
            # Generate
            start_time = time.time()
            generated = pipeline.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50
            )
            generation_time = time.time() - start_time
            
            # Check output
            generated_np = np.array(generated)
            num_generated = generated_np.shape[1] - len(input_ids)
            
            details = {
                "input_length": len(input_ids),
                "output_length": generated_np.shape[1],
                "tokens_generated": num_generated,
                "generation_time": generation_time,
                "tokens_per_second": num_generated / generation_time if generation_time > 0 else 0
            }
            
            passed = num_generated > 0
            
            result = ValidationResult(
                test_name="Generation",
                passed=passed,
                max_abs_error=0.0,
                max_rel_error=0.0,
                mean_abs_error=0.0,
                details=details
            )
            
            print(f"  {'✅' if passed else '❌'} Generated {num_generated} tokens in {generation_time:.2f}s")
            print(f"  Throughput: {details['tokens_per_second']:.1f} tokens/s")
            
        except Exception as e:
            result = ValidationResult(
                test_name="Generation",
                passed=False,
                max_abs_error=float('inf'),
                max_rel_error=float('inf'),
                mean_abs_error=float('inf'),
                details={"error": str(e)}
            )
            print(f"  ❌ Test failed: {e}")
        
        self.results.append(result)
        return result
    
    def run_all_tests(self, model: GPTOSSModel, config: GPTOSSConfig) -> Dict[str, Any]:
        """Run all validation tests."""
        print("\n" + "="*60)
        print("RUNNING VALIDATION SUITE")
        print("="*60)
        
        # Run tests
        self.test_mxfp4_dequantization()
        self.test_numerical_stability()
        self.test_model_output(model, config)
        
        # Test generation
        pipeline = InferencePipeline(model, config)
        self.test_generation(pipeline)
        
        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        # Show failed tests
        failed = [r for r in self.results if not r.passed]
        if failed:
            print("\n❌ Failed tests:")
            for r in failed:
                print(f"  • {r.test_name}")
                if "error" in r.details:
                    print(f"    Error: {r.details['error']}")
        
        # Overall assessment
        all_passed = all(r.passed for r in self.results)
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED - Model ready for production!")
        else:
            print("\n⚠️  Some tests failed - review results before deployment")
        
        return {
            "all_passed": all_passed,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests/total_tests,
            "results": [r.to_dict() for r in self.results]
        }
    
    def save_results(self, save_path: Path) -> None:
        """Save validation results to JSON."""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.passed),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📝 Results saved to {save_path}")


def validate_model(model: GPTOSSModel, config: GPTOSSConfig, save_results: bool = True) -> Dict[str, Any]:
    """
    Run complete validation suite on a model.
    
    Args:
        model: GPTOSSModel instance
        config: Model configuration
        save_results: Whether to save results to file
    
    Returns:
        Validation results dictionary
    """
    validator = ModelValidator()
    results = validator.run_all_tests(model, config)
    
    if save_results:
        results_file = Path("validation_results.json")
        validator.save_results(results_file)
    
    return results