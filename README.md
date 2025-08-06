# TensorPort 🚀

Fast, memory-efficient conversion from Safetensors to JAX-compatible formats with MXFP4 quantization support.

## Features

- ⚡ **Direct Rust → JAX**: Convert directly to NumPy arrays (.npy) that JAX can load natively
- 🎯 **MXFP4 Support**: Correctly handles 4-bit quantized models (21.5B params for GPT-OSS-20B)
- 🧠 **Custom bfloat16 Support**: Handles bfloat16 tensors that standard libraries can't read
- 💾 **Memory Efficient**: Streaming conversion without loading entire models into memory
- 📦 **Smart Sharding**: Automatically splits large models into Git LFS-compatible chunks (<2GB)
- 🔄 **Resume Capability**: Continue interrupted conversions with --resume flag
- 🎯 **Multiple Formats**: NumPy direct, MessagePack, or JAX pickle output

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

### Convert to NumPy for JAX (Recommended)

```bash
tensorport convert \
    --input path/to/safetensors/model \
    --output path/to/output/directory \
    --format numpy-direct \
    --shard-size 1.8 \
    --precision float16
```

### Loading in JAX

```python
import jax.numpy as jnp
import json

# Load manifest
with open('output/manifest.json') as f:
    manifest = json.load(f)

# Load a tensor directly
tensor = jnp.load('output/shard_000/tensor_name.npy')
```

### Complete JAX Inference Example

```python
# Load converted model parameters
from pathlib import Path
import jax.numpy as jnp
import numpy as np

def load_tensor(base_path, tensor_name):
    """Load a single tensor from sharded numpy files."""
    file_name = tensor_name.replace('.', '_') + '.npy'
    for shard_dir in sorted(base_path.glob('shard_*')):
        tensor_path = shard_dir / file_name
        if tensor_path.exists():
            return jnp.array(np.load(tensor_path))
    return None

# Load model weights
model_path = Path('jax-numpy-model')
embeddings = load_tensor(model_path, 'model.embed_tokens.weight')
q_proj = load_tensor(model_path, 'model.layers.0.self_attn.q_proj.weight')

# Run inference
input_ids = jnp.array([[1, 2, 3, 4, 5]])
hidden_states = embeddings[input_ids]
query = jnp.matmul(hidden_states, q_proj.T)
```

✅ **Verified**: Successfully tested with GPT-OSS-20B (21.5B parameters)

## Latest Release

**v0.2.0** - Full JAX Integration
- Complete JAX inference pipeline
- Tokenization support with example generations
- 459 tensor files directly loadable by JAX
- Tested end-to-end with GPT-OSS-20B model
- Performance: ~1s/10 tokens on CPU, expect 100x faster on TPU

## Options

- `--format`: Output format (`numpy-direct`, `msgpack`, `jax-pickle`)
- `--shard-size`: Maximum size per shard in GB (default: 1.8 for Git LFS)
- `--precision`: Target precision (`float16` or `float32`)
- `--resume`: Resume interrupted conversion
- `--workers`: Number of parallel workers (default: CPU count)

## Supported Formats

**Input:**
- Safetensors format with model.safetensors.index.json
- All tensor types: BF16 (bfloat16), F32, F16, U8, I64
- MXFP4 quantized weights (4-bit packed in U8)

**Output Formats:**
- **numpy-direct**: NumPy .npy files - JAX loads directly with `jnp.load()`
- **msgpack**: MessagePack sharded format
- **jax-pickle**: Direct pickle format (slower for large models)

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

### Converting GPT-OSS-20B (MXFP4 Quantized)

```bash
# Convert the full 21.5B parameter model
./target/release/tensorport convert \
    --input models/gpt-oss-20b \
    --output models/gpt-oss-20b-jax \
    --format numpy-direct \
    --precision float16

# Output:
# ✅ NumPy array conversion complete!
#    Total shards: 8
#    Total parameters: 21.51B
#    Output: models/gpt-oss-20b-jax
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