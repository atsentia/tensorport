#!/usr/bin/env python3
"""
Comprehensive benchmark script for TensorPort on ARM64 Neoverse-N1
"""
import subprocess
import time
import json
import psutil
import os
from pathlib import Path
from datetime import datetime

def get_system_info():
    """Gather comprehensive system information"""
    import platform
    
    # CPU info from lscpu
    lscpu_output = subprocess.check_output(['lscpu'], text=True)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version()
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "model": "ARM Neoverse-N1",
            "architecture": "aarch64"
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "percent_used": psutil.virtual_memory().percent
        }
    }

def benchmark_tensorport_operations():
    """Run various TensorPort operations and measure performance"""
    results = []
    
    # Test 1: Version check
    start = time.perf_counter()
    version_output = subprocess.run(
        ['./target/release/tensorport', '--version'],
        capture_output=True, text=True
    )
    version_time = time.perf_counter() - start
    results.append({
        "operation": "version_check",
        "duration_ms": round(version_time * 1000, 2),
        "output": version_output.stdout.strip()
    })
    
    # Test 2: Help command
    start = time.perf_counter()
    help_output = subprocess.run(
        ['./target/release/tensorport', '--help'],
        capture_output=True, text=True
    )
    help_time = time.perf_counter() - start
    results.append({
        "operation": "help_display",
        "duration_ms": round(help_time * 1000, 2),
        "lines_output": len(help_output.stdout.splitlines())
    })
    
    # Test 3: Check if we have sample data to convert
    sample_path = Path("/root/tensorport/jax-numpy-model")
    if sample_path.exists():
        # Test loading manifest
        manifest_path = sample_path / "manifest.json"
        if manifest_path.exists():
            start = time.perf_counter()
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            load_time = time.perf_counter() - start
            results.append({
                "operation": "manifest_load",
                "duration_ms": round(load_time * 1000, 2),
                "tensor_count": len(manifest.get("tensors", []))
            })
            
            # Count numpy files
            npy_files = list(sample_path.rglob("*.npy"))
            results.append({
                "operation": "numpy_file_scan",
                "file_count": len(npy_files),
                "total_size_mb": round(sum(f.stat().st_size for f in npy_files) / (1024**2), 2)
            })
    
    return results

def memory_benchmark():
    """Benchmark memory usage patterns"""
    import gc
    
    gc.collect()
    initial_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
    
    # Simulate loading operations
    memory_tests = []
    
    # Test 1: Allocate and deallocate arrays
    import numpy as np
    
    for size_mb in [10, 50, 100, 200]:
        gc.collect()
        before = psutil.Process().memory_info().rss / (1024**2)
        
        # Allocate array
        elements = int(size_mb * 1024 * 1024 / 4)  # float32
        arr = np.random.randn(elements).astype(np.float32)
        
        after_alloc = psutil.Process().memory_info().rss / (1024**2)
        
        # Process (simulate conversion)
        arr_processed = arr.astype(np.float16)
        after_process = psutil.Process().memory_info().rss / (1024**2)
        
        # Cleanup
        del arr, arr_processed
        gc.collect()
        after_cleanup = psutil.Process().memory_info().rss / (1024**2)
        
        memory_tests.append({
            "test_size_mb": size_mb,
            "baseline_mb": round(before, 2),
            "after_alloc_mb": round(after_alloc, 2),
            "after_process_mb": round(after_process, 2),
            "after_cleanup_mb": round(after_cleanup, 2),
            "peak_usage_mb": round(after_process - before, 2)
        })
    
    return {
        "initial_memory_mb": round(initial_memory, 2),
        "tests": memory_tests
    }

def io_benchmark():
    """Benchmark I/O operations"""
    import tempfile
    import numpy as np
    
    results = []
    test_sizes = [1, 10, 50, 100]  # MB
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        for size_mb in test_sizes:
            elements = int(size_mb * 1024 * 1024 / 4)
            data = np.random.randn(elements).astype(np.float32)
            
            # Write benchmark
            write_path = tmppath / f"test_{size_mb}mb.npy"
            start = time.perf_counter()
            np.save(write_path, data)
            write_time = time.perf_counter() - start
            
            # Read benchmark
            start = time.perf_counter()
            loaded = np.load(write_path)
            read_time = time.perf_counter() - start
            
            # Calculate throughput
            write_throughput = size_mb / write_time  # MB/s
            read_throughput = size_mb / read_time    # MB/s
            
            results.append({
                "size_mb": size_mb,
                "write_ms": round(write_time * 1000, 2),
                "read_ms": round(read_time * 1000, 2),
                "write_throughput_mbps": round(write_throughput, 2),
                "read_throughput_mbps": round(read_throughput, 2)
            })
    
    return results

def run_comprehensive_benchmark():
    """Run all benchmarks and compile results"""
    print("Starting comprehensive benchmark on ARM64 Neoverse-N1...")
    
    results = {
        "system_info": get_system_info(),
        "tensorport_operations": benchmark_tensorport_operations(),
        "memory_benchmark": memory_benchmark(),
        "io_benchmark": io_benchmark()
    }
    
    # Generate summary statistics
    results["summary"] = {
        "total_tests_run": len(results["tensorport_operations"]) + 
                          len(results["memory_benchmark"]["tests"]) + 
                          len(results["io_benchmark"]),
        "average_io_read_throughput_mbps": round(
            sum(t["read_throughput_mbps"] for t in results["io_benchmark"]) / 
            len(results["io_benchmark"]), 2
        ),
        "average_io_write_throughput_mbps": round(
            sum(t["write_throughput_mbps"] for t in results["io_benchmark"]) / 
            len(results["io_benchmark"]), 2
        ),
        "peak_memory_test_mb": max(
            t["peak_usage_mb"] for t in results["memory_benchmark"]["tests"]
        )
    }
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    
    # Save results to JSON
    output_path = Path("benchmark_results_arm64_neoverse_n1.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark complete! Results saved to {output_path}")
    print(f"\nSystem: {results['system_info']['cpu']['model']} ({results['system_info']['cpu']['architecture']})")
    print(f"Cores: {results['system_info']['cpu']['physical_cores']} physical, {results['system_info']['cpu']['logical_cores']} logical")
    print(f"Memory: {results['system_info']['memory']['total_gb']} GB total")
    print(f"Total tests run: {results['summary']['total_tests_run']}")