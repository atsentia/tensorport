# TensorPort Development Instructions

**ALWAYS FOLLOW THESE INSTRUCTIONS FIRST**. Only fall back to additional search and context gathering if the information here is incomplete or found to be in error.

## Overview

TensorPort is a high-performance Rust library for converting Safetensors models to JAX-compatible formats with MXFP4 quantization support. The project includes:
- **Rust core** (`tensorport`): Safetensors to JAX conversion with custom bfloat16 parsing
- **Python library** (`jax-mxfp4/`): 4-bit quantization library for JAX
- **Python scripts**: Inference, testing, and Modal.com deployment scripts

## Critical Build Requirements

### **NEVER CANCEL BUILDS OR LONG-RUNNING COMMANDS**
- Rust builds take **~1 minute 20 seconds**. NEVER CANCEL. Set timeout to **3+ minutes**.
- Rust tests take **~28 seconds**. NEVER CANCEL. Set timeout to **1+ minute**.
- Python package installation takes **~1-2 minutes**. NEVER CANCEL. Set timeout to **5+ minutes**.

## Bootstrap and Build Process

Run these commands in exact order:

### 1. Build Rust Project
```bash
cd /path/to/tensorport
cargo build --release
```
**NEVER CANCEL: Takes ~1 minute 20 seconds. Set timeout to 3+ minutes.**

### 2. Test Rust Project
```bash
cargo test
```
**NEVER CANCEL: Takes ~28 seconds. Set timeout to 1+ minute.**
- Runs 8 tests in both lib and bin targets
- All tests should pass with some warnings (warnings are expected)

### 3. Install Python Dependencies
```bash
python3 -m pip install -e jax-mxfp4/
```
**NEVER CANCEL: Takes 1-2 minutes. Set timeout to 5+ minutes.**
- Installs JAX, Flax, NumPy, and other ML dependencies
- Required for Python inference scripts

### 4. Install Python Development Tools
```bash
python3 -m pip install pytest ruff mypy black isort
```
**NEVER CANCEL: Takes 1 minute. Set timeout to 3+ minutes.**

## Working with the Codebase

### TensorPort CLI Usage
After building, the CLI is available at `./target/release/tensorport`:

```bash
# Get help
./target/release/tensorport --help
./target/release/tensorport convert --help

# Convert a model (example - requires actual model data)
./target/release/tensorport convert \
    --input /path/to/safetensors/model \
    --output /path/to/output \
    --format numpy-direct \
    --precision float16
```

### Python Linting and Code Quality
**ALWAYS run these before committing changes:**

```bash
cd jax-mxfp4/
ruff check .        # Lint Python code
black .             # Format Python code  
isort .             # Sort imports
mypy .              # Type checking
```

### Rust Linting
Rust warnings are expected and acceptable. No additional linting is required.

## Testing and Validation

### Required Validation After Changes
**ALWAYS run these validation steps after making changes:**

1. **Build validation:**
   ```bash
   cargo build --release  # NEVER CANCEL - 3+ minute timeout
   ```

2. **Test validation:**
   ```bash
   cargo test             # NEVER CANCEL - 1+ minute timeout
   ```

3. **Python import validation:**
   ```bash
   python3 -c "import jax; import jax_mxfp4; print('✅ JAX imports successful')"
   ```

4. **Basic functionality validation:**
   ```bash
   python3 -c "
   from jax_mxfp4.layers import MXFP4Linear
   layer = MXFP4Linear(features=128, block_size=32)
   print('✅ JAX-MXFP4 layer creation successful')
   "
   ```

5. **CLI validation:**
   ```bash
   ./target/release/tensorport --help
   ```

### Python Testing
Currently no automated test suite. Validation is manual through:
- Import tests (above)
- Basic functionality tests (above)
- Python syntax validation: `python3 -m py_compile script.py`

## Project Structure

### Key Directories
```
tensorport/
├── src/                           # Rust source code
│   ├── main.rs                   # CLI entry point
│   ├── converter.rs              # Main conversion logic
│   ├── tensor.rs                 # Tensor type handling (bfloat16)
│   ├── formats/                  # Output format implementations
│   └── ...
├── jax-mxfp4/                    # Python 4-bit quantization library
│   ├── jax_mxfp4/               
│   │   ├── quantize.py          # MXFP4 quantization functions
│   │   ├── layers.py            # Flax/JAX layers
│   │   ├── kernels.py           # Optimized kernels
│   │   └── __init__.py          # Main exports
│   └── pyproject.toml           # Python package config
├── Cargo.toml                    # Rust package config
├── *.py                         # Inference and testing scripts
└── README.md                    # Main documentation
```

### Important Files to Check After Changes
- **Rust changes**: Always check `Cargo.toml` dependencies
- **Python changes**: Always check `jax-mxfp4/pyproject.toml` dependencies
- **JAX-MXFP4 changes**: Always test imports after modifying `jax_mxfp4/__init__.py`
- **CLI changes**: Always test `./target/release/tensorport --help` after changes to `src/main.rs`

## Common Tasks

### Adding New Rust Dependencies
1. Edit `Cargo.toml`
2. Run `cargo build --release` (NEVER CANCEL - 3+ minute timeout)
3. Run `cargo test` (NEVER CANCEL - 1+ minute timeout)

### Adding New Python Dependencies  
1. Edit `jax-mxfp4/pyproject.toml`
2. Run `python3 -m pip install -e jax-mxfp4/` (NEVER CANCEL - 5+ minute timeout)
3. Test imports: `python3 -c "import jax_mxfp4; print('OK')"`

### Working with Conversions
TensorPort converts Safetensors models to JAX formats. Key points:
- Input: Safetensors format with `.safetensors.index.json`
- Output: NumPy `.npy` files that JAX can load directly
- Supports custom bfloat16 parsing for models other libraries can't handle
- MXFP4 quantization reduces model size by ~75%

### Performance Expectations
- **Model size**: GPT-OSS-20B: 86GB → 13GB (MXFP4 quantized)
- **Conversion time**: Large models may take 10+ minutes (this is normal)
- **Memory usage**: Streaming architecture keeps RAM usage low (<2GB)

## Known Issues and Limitations

### JAX-MXFP4 Library Status
- ✅ **Working**: Basic imports, layer creation, quantization functions
- ⚠️ **Partially working**: Some quantization functions have JAX tracing issues
- ❌ **Missing**: `model.py` and `training.py` modules (in development)

### Expected Warnings
- **Rust**: Dead code warnings are normal and acceptable
- **Python**: Ruff linting will show formatting issues that should be fixed

### Missing Components
- No GitHub Actions CI/CD pipeline (manual testing required)
- No automated Python test suite (manual validation required)
- Model conversion requires actual Safetensors model data

## Troubleshooting

### Build Failures
- **Rust build fails**: Check `Cargo.toml` for dependency conflicts
- **Python install fails**: Check `jax-mxfp4/pyproject.toml` for version conflicts
- **Import failures**: Check that JAX-MXFP4 was installed correctly

### Performance Issues
- **Slow builds**: Normal for first build (~1m 20s), incremental builds are faster
- **Memory issues**: Use streaming conversion with `--shard-size` parameter
- **Large models**: May require 10+ minutes to convert (this is expected)

### JAX/Python Issues
- **JAX tracing errors**: Some quantization functions have limitations
- **Import errors**: Ensure all dependencies installed via `pip install -e jax-mxfp4/`
- **GPU issues**: JAX defaults to CPU, GPU setup requires additional configuration

## Development Workflow

1. **Make changes** to Rust or Python code
2. **Build and test** using commands above (respect timeouts)
3. **Run validation** steps to ensure everything works
4. **Run linting** tools (especially for Python)
5. **Test manually** with actual workflows when possible
6. **Commit changes** only after validation passes

Remember: **NEVER CANCEL long-running builds or tests**. They are working correctly even if they take several minutes.