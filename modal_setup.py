#!/usr/bin/env python3
"""
Modal setup script to download GPT-OSS-20B from HuggingFace,
convert to JAX format using TensorPort, and store in Modal volume.

Run once to prepare weights for benchmarking.
"""

import modal
import os
import subprocess
import json
from pathlib import Path
import time

# Modal app configuration
app = modal.App("gpt-oss-jax-setup")

# Define volume for storing converted weights
volume = modal.Volume.from_name("gpt-oss-20b-jax", create_if_missing=True)

# Docker image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "cargo", "rustc", "build-essential", "curl"])
    .pip_install([
        "huggingface_hub",
        "safetensors",
        "numpy",
        "tqdm",
        "requests",
    ])
    .run_commands(
        # Clone and build TensorPort
        "git clone https://github.com/atsentia/tensorport.git /tensorport",
        "cd /tensorport && cargo build --release",
    )
)

@app.function(
    image=image,
    volumes={"/cache": volume},
    timeout=7200,  # 2 hour timeout for download and conversion
    cpu=8.0,
    memory=32768,  # 32GB RAM for conversion
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def setup_weights():
    """Download and convert GPT-OSS-20B weights to JAX format."""
    
    print("="*60)
    print("GPT-OSS-20B JAX Weight Setup")
    print("="*60)
    
    # Check if weights already exist
    cache_dir = Path("/cache/gpt-oss-20b-jax")
    manifest_path = cache_dir / "manifest.json"
    
    if manifest_path.exists():
        print("✅ Weights already converted and cached!")
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"   Found {len(manifest.get('shards', []))} shards")
        return {"status": "already_exists", "path": str(cache_dir)}
    
    # Create directories
    download_dir = Path("/tmp/gpt-oss-20b")
    download_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download from HuggingFace
    print("\n📥 Downloading GPT-OSS-20B from HuggingFace...")
    start_time = time.time()
    
    from huggingface_hub import snapshot_download
    
    try:
        snapshot_download(
            repo_id="allen-ai/gpt-oss-20b",
            local_dir=download_dir,
            cache_dir="/tmp/hf_cache",
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )
        download_time = time.time() - start_time
        print(f"✅ Download complete in {download_time:.1f} seconds")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise
    
    # Verify download
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "model-00000-of-00002.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    
    for file in required_files:
        if not (download_dir / file).exists():
            raise FileNotFoundError(f"Missing required file: {file}")
    
    print(f"✅ All required files present")
    
    # Step 2: Convert using TensorPort
    print("\n🔄 Converting to JAX format using TensorPort...")
    start_time = time.time()
    
    tensorport_cmd = [
        "/tensorport/target/release/tensorport",
        "convert",
        "--input", str(download_dir),
        "--output", str(cache_dir),
        "--format", "numpy-direct",
        "--shard-size", "2.0",
        "--precision", "float16",
        "--workers", "8",
        "--skip-verify"
    ]
    
    try:
        result = subprocess.run(
            tensorport_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        conversion_time = time.time() - start_time
        print(f"✅ Conversion complete in {conversion_time:.1f} seconds")
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print(f"Stderr: {e.stderr}")
        raise
    
    # Step 3: Verify conversion
    print("\n🔍 Verifying converted weights...")
    
    if not manifest_path.exists():
        raise FileNotFoundError("Manifest file not created")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Count tensor files
    tensor_count = 0
    total_size = 0
    for shard_dir in sorted(cache_dir.glob("shard_*")):
        npy_files = list(shard_dir.glob("*.npy"))
        tensor_count += len(npy_files)
        for file in npy_files:
            total_size += file.stat().st_size
    
    print(f"✅ Verification complete:")
    print(f"   • Shards: {len(list(cache_dir.glob('shard_*')))}")
    print(f"   • Tensors: {tensor_count}")
    print(f"   • Total size: {total_size / (1024**3):.2f} GB")
    print(f"   • Config: {manifest['config']['num_hidden_layers']} layers, "
          f"{manifest['config']['hidden_size']} hidden size")
    
    # Clean up download directory to save space
    print("\n🧹 Cleaning up temporary files...")
    import shutil
    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree("/tmp/hf_cache", ignore_errors=True)
    
    print("\n" + "="*60)
    print("✨ Setup Complete!")
    print("="*60)
    print(f"Weights stored in volume: /cache/gpt-oss-20b-jax")
    print(f"Ready for benchmarking!")
    
    return {
        "status": "success",
        "path": str(cache_dir),
        "tensor_count": tensor_count,
        "size_gb": total_size / (1024**3),
        "download_time": download_time,
        "conversion_time": conversion_time,
    }

@app.local_entrypoint()
def main():
    """Run the setup process."""
    print("Starting GPT-OSS-20B weight setup on Modal...")
    
    try:
        result = setup_weights.remote()
        
        print("\n📊 Setup Results:")
        print(json.dumps(result, indent=2))
        
        if result["status"] == "already_exists":
            print("\n⚠️  Weights already exist. Delete the volume to re-convert.")
        else:
            print(f"\n✅ Successfully prepared weights!")
            print(f"   Download: {result['download_time']:.1f}s")
            print(f"   Conversion: {result['conversion_time']:.1f}s")
            print(f"   Size: {result['size_gb']:.2f} GB")
            
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()