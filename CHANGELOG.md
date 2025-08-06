# Changelog

All notable changes to TensorPort will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-08-06

### Added
- Complete JAX inference pipeline with working forward pass
- Tokenization support with realistic text generation examples
- Three comprehensive test scripts for JAX integration
- Simple inference demonstration showing input/output flow
- Executive summary documentation for stakeholders
- Project status documentation with detailed metrics

### Changed
- Updated README with JAX inference examples and code snippets
- Improved project completion status to 92%
- Enhanced documentation with performance metrics

### Fixed
- Rust compiler warnings for unused imports
- Attention mechanism dimensions for proper JAX tensor operations

### Verified
- End-to-end pipeline with GPT-OSS-20B (21.5B parameters)
- 459 tensor files successfully loadable by JAX
- Inference performance: ~1s for 10 tokens on CPU
- Memory usage remains under 2GB during conversion

## [0.1.0] - 2025-08-05

### Added
- Initial TensorPort implementation
- Custom bfloat16 parsing for safetensors
- MXFP4 quantization support
- Memory-efficient streaming conversion
- Multiple output formats (numpy-direct, msgpack, jax-pickle)
- Resume capability for interrupted conversions
- Parallel processing with configurable workers
- Git LFS-compatible sharding (<2GB per shard)

### Features
- Handles models that standard libraries cannot process
- Direct byte-level parsing of bfloat16 weights
- Zero-copy memory-mapped file access
- Progress tracking with real-time statistics

---

For more information, see the [README](README.md) and [PROJECT_STATUS](PROJECT_STATUS.md).