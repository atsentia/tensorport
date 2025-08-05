# TensorPort 🚀

Fast, memory-efficient tensor format conversion with custom bfloat16 support. Convert PyTorch safetensors models to JAX-compatible sharded formats.

## Features

- ⚡ **Fast Conversion**: Written in Rust for maximum performance
- 🧠 **Custom bfloat16 Support**: Handles bfloat16 tensors that standard libraries can't read
- 💾 **Memory Efficient**: Streaming conversion without loading entire models into memory
- 📦 **Smart Sharding**: Automatically splits large models into Git LFS-compatible chunks
- 🔍 **Verification**: Built-in integrity checking for converted models
- 🎯 **Precision Control**: Choose between float16 and float32 output precision

## Installation

```bash
cargo install tensorport
```

Or build from source:

```bash
git clone https://github.com/atsentia/tensorport.git
cd tensorport
cargo build --release
```

## Usage

### Convert a Model

```bash
tensorport convert \
    --input path/to/safetensors/model \
    --output path/to/output/directory \
    --shard-size 1.8 \
    --precision float16
```

### Verify Conversion

```bash
tensorport verify --model path/to/converted/model --verbose
```

## Options

- `--shard-size`: Maximum size per shard in GB (default: 1.8 for Git LFS compatibility)
- `--precision`: Target precision (`float16` or `float32`, default: `float16`)
- `--workers`: Number of parallel workers (default: CPU count)
- `--skip-verify`: Skip verification after conversion

## Supported Formats

**Input:**
- Safetensors format with model.safetensors.index.json
- All tensor types: BF16 (bfloat16), F32, F16, U8, I64
- MXFP4 quantized MoE expert weights (as U8)

**Output:**
- MessagePack sharded format optimized for JAX
- Configurable precision (float16/float32)
- JSON manifest with metadata

## Why TensorPort?

TensorPort was created to solve the challenge of converting large language models with bfloat16 weights that standard libraries couldn't handle. It implements custom bfloat16 parsing by reading safetensors files at the byte level, bypassing library limitations.

### Key Innovation: Custom bfloat16 Parsing

```rust
// bfloat16 format: 1 sign + 8 exponent + 7 mantissa bits
// Conversion: shift bfloat16 to upper 16 bits of u32, interpret as f32
let bf16_bits = cursor.read_u16::<LittleEndian>()?;
let f32_bits = (bf16_bits as u32) << 16;
let f32_value = f32::from_bits(f32_bits);
```

This allows TensorPort to extract ALL parameters from models like GPT-OSS-20B (21B parameters) where other tools fail.

## Performance

TensorPort is designed for large-scale model conversion:

- **Memory Efficient**: Streaming processing with configurable memory boundaries
- **Fast I/O**: Memory-mapped file access with zero-copy reads
- **Progress Tracking**: Real-time progress bars and statistics
- **Robust**: Comprehensive error handling and validation

## Examples

### Converting GPT-OSS-20B

```bash
# Convert the full 21B parameter model
tensorport convert \
    --input models/gpt-oss-20b \
    --output models/gpt-oss-20b-jax \
    --precision float16

# Verify the conversion
tensorport verify --model models/gpt-oss-20b-jax --verbose
```

### Custom Shard Size

```bash
# Use larger shards for internal use (not Git LFS)
tensorport convert \
    --input large-model \
    --output large-model-jax \
    --shard-size 5.0 \
    --precision float32
```

## Architecture

TensorPort consists of several key modules:

- **`formats`**: Safetensors file parsing with memory mapping
- **`tensor`**: Custom tensor type conversion (especially bfloat16)
- **`shard`**: Memory-efficient streaming shard writer
- **`converter`**: Main orchestration with progress tracking
- **`verify`**: Integrity verification for converted models

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see our contributing guidelines.

---

**TensorPort**: Because tensor conversion shouldn't be a roadblock to AI research. 🚀