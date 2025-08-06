# Modal.com Deployment: GPT-OSS-20B JAX vs PyTorch Benchmark

Comprehensive benchmarking suite for comparing JAX (via TensorPort) and PyTorch inference performance across multiple GPU types on Modal.com.

## 🎯 Overview

This deployment benchmarks GPT-OSS-20B (21.5B parameters) inference comparing:
- **Frameworks**: JAX (with TensorPort-converted weights) vs PyTorch (HuggingFace)
- **GPUs**: H100, A100-40GB, L4, T4
- **Metrics**: Throughput, latency, cost-efficiency
- **Scale**: 100 diverse LinkedIn-friendly prompts

## 📁 Files

- `modal_setup.py` - One-time setup to download and convert weights
- `modal_benchmark.py` - Main benchmarking script with proper warmup
- `benchmark_prompts.json` - 100 diverse professional prompts
- `analyze_results.py` - Results analysis and report generation

## 🚀 Quick Start

### Prerequisites

1. Install Modal CLI:
```bash
pip install modal
modal setup
```

2. Ensure you have Modal credits (benchmarks cost ~$10-20)

### Step 1: Setup Weights (One-time)

```bash
# Download GPT-OSS-20B and convert to JAX format
modal run modal_setup.py

# This will:
# - Download ~13GB from HuggingFace
# - Convert to numpy-direct format using TensorPort
# - Store in Modal volume "gpt-oss-20b-jax"
# - Takes ~30-60 minutes
```

### Step 2: Run Benchmarks

```bash
# Quick test (10 prompts on T4 only)
modal run modal_benchmark.py --gpu-types T4 --num-prompts 10

# Full benchmark (100 prompts on all GPUs)
modal run modal_benchmark.py \
    --gpu-types H100,A100-40GB,L4,T4 \
    --num-prompts 100 \
    --frameworks jax,pytorch

# JAX only benchmark
modal run modal_benchmark.py \
    --gpu-types L4,T4 \
    --num-prompts 100 \
    --frameworks jax
```

### Step 3: Analyze Results

```bash
# Generate markdown report and CSV
python analyze_results.py full_results.json --format all

# This creates:
# - benchmark_report_TIMESTAMP.md
# - benchmark_results_TIMESTAMP.csv
```

## 📊 Benchmark Methodology

### Warmup Strategy

Both frameworks undergo comprehensive warmup to ensure fair comparison:

#### JAX Warmup
- 5 iterations × 5 sequence lengths (10, 20, 30, 40, 50 tokens)
- Additional 5 iterations with typical input shape
- Ensures JIT compilation is complete
- Uses `block_until_ready()` to guarantee compilation

#### PyTorch Warmup
- 5 iterations × 5 different prompt lengths
- CUDA kernel compilation and memory allocation
- Cache cleared after warmup
- Uses `torch.cuda.synchronize()` for accurate timing

### Measurement Protocol
1. **Excluded measurements**: First 5 inferences after warmup are excluded
2. **Timing precision**: GPU synchronization enforced before timing
3. **Statistical validity**: Reports mean, median, p95, p99, std deviation
4. **Cost calculation**: Based on Modal's published per-second pricing

## 💰 Cost Estimates

| Component | Estimated Cost | Duration |
|-----------|---------------|----------|
| Setup (one-time) | $0.50 | 30-60 min |
| Benchmark (per GPU) | $2-5 | 10-15 min |
| Full suite (4 GPUs) | $10-20 | 40-60 min |
| Storage (monthly) | $0.023 | Ongoing |

## 📈 Expected Results

Based on typical performance patterns:

### Throughput (tokens/second)
- **H100**: 500-800 tok/s
- **A100-40GB**: 300-500 tok/s
- **L4**: 100-200 tok/s
- **T4**: 50-100 tok/s

### JAX vs PyTorch
- JAX typically 1.5-2x faster after JIT compilation
- Lower memory usage with JAX
- Faster model loading with pre-converted weights

### Cost Efficiency
- **Best throughput**: H100
- **Best value**: L4 (tokens per dollar)
- **Budget option**: T4

## 🔧 Configuration Options

### GPU Selection
```python
GPU_CONFIGS = {
    "H100": modal.gpu.H100(),           # $0.001097/s
    "A100-40GB": modal.gpu.A100("40GB"), # $0.000583/s
    "L4": modal.gpu.L4(),                # $0.000222/s
    "T4": modal.gpu.T4(),                # $0.000164/s
}
```

### Customizing Prompts

Edit `benchmark_prompts.json` to add your own prompts:
```json
{
  "prompts": [
    "Your custom prompt here",
    "Another test prompt"
  ]
}
```

## 📊 Sample Output

```
BENCHMARK SUMMARY
================================================================

📊 Performance Rankings (by throughput):
1. H100        + jax      :    650.3 tok/s | $0.0017/1k
2. H100        + pytorch  :    425.1 tok/s | $0.0026/1k
3. A100-40GB   + jax      :    380.2 tok/s | $0.0015/1k
4. A100-40GB   + pytorch  :    245.8 tok/s | $0.0024/1k

💰 Cost Efficiency Rankings:
1. L4          + jax      : $0.0011/1k tokens
2. T4          + jax      : $0.0016/1k tokens
3. A100-40GB   + jax      : $0.0015/1k tokens
4. H100        + jax      : $0.0017/1k tokens
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size in benchmark script
   - Use smaller GPU (counterintuitively can help with memory management)
   - Clear cache more frequently

2. **Volume Not Found**
   - Ensure setup script completed successfully
   - Check volume exists: `modal volume list`

3. **Slow First Run**
   - Normal - Docker images are being built
   - Subsequent runs will be faster

4. **Import Errors**
   - Modal handles dependencies automatically
   - If issues persist, check image definitions in script

## 🔍 Advanced Usage

### Custom Analysis

```python
# Load and analyze results programmatically
import json
import pandas as pd

with open("full_results.json") as f:
    results = json.load(f)

# Extract specific metrics
for config, data in results.items():
    if "stats" in data:
        print(f"{config}: {data['stats']['tokens_per_second']:.1f} tok/s")
```

### Monitoring During Run

Modal provides real-time logs:
```bash
# Watch logs in real-time
modal run modal_benchmark.py --gpu-types T4 --num-prompts 10

# View in Modal dashboard
# https://modal.com/apps
```

## 📝 Notes

- **Warmup is critical**: JAX requires JIT compilation, PyTorch needs CUDA kernel compilation
- **Fair comparison**: Both frameworks use same prompts, generation length, and hardware
- **Production considerations**: Add proper tokenization, error handling, and monitoring
- **Cost optimization**: Use spot instances for non-critical benchmarks

## 🤝 Contributing

To add new benchmarks or improve the suite:

1. Add new GPU configs to `GPU_CONFIGS`
2. Extend prompts in `benchmark_prompts.json`
3. Add metrics to `analyze_results.py`
4. Submit PR with benchmark results

## 📄 License

MIT - See LICENSE file

## 🙏 Acknowledgments

- Modal.com for serverless GPU infrastructure
- HuggingFace for model hosting
- TensorPort for efficient weight conversion
- OpenAI for GPT-OSS-20B model

---

**Last Updated**: 2025-08-06
**Version**: 1.0.0
**Status**: Production Ready