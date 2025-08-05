use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod converter;
mod error;
mod formats;
mod shard;
mod tensor;
mod verify;

use crate::converter::TensorportConverter;
use crate::error::TensorportResult;

#[derive(Parser)]
#[command(name = "tensorport")]
#[command(about = "Fast, memory-efficient tensor format conversion with custom bfloat16 support")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert safetensors model to JAX-compatible sharded format
    Convert {
        /// Path to input safetensors model directory
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output directory for converted shards
        #[arg(short, long)]
        output: PathBuf,
        
        /// Maximum size per shard in GB (default: 1.8 for Git LFS compatibility)
        #[arg(short, long, default_value = "1.8")]
        shard_size: f64,
        
        /// Number of parallel workers (default: CPU count)
        #[arg(short, long)]
        workers: Option<usize>,
        
        /// Target precision for converted weights
        #[arg(short, long, default_value = "float16")]
        precision: String,
        
        /// Output format (jax, numpy, msgpack, safetensors)
        #[arg(short, long, default_value = "jax")]
        format: String,
        
        /// Skip verification after conversion
        #[arg(long)]
        skip_verify: bool,
        
        /// Resume interrupted conversion
        #[arg(long)]
        resume: bool,
    },
    
    /// Verify converted model integrity
    Verify {
        /// Path to converted model directory
        #[arg(short, long)]
        model: PathBuf,
        
        /// Show detailed tensor information
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() -> TensorportResult<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Convert {
            input,
            output,
            shard_size,
            workers,
            precision,
            format,
            skip_verify,
            resume,
        } => {
            let converter = TensorportConverter::new(
                input,
                output,
                shard_size,
                workers,
                precision,
                format,
                resume,
            )?;
            
            println!("🚀 TensorPort: Fast Tensor Conversion");
            println!("=====================================");
            
            let result = converter.convert()?;
            
            println!("\n🎉 Conversion Complete!");
            println!("📊 Total parameters: {} ({:.2}B)", result.total_params, result.total_params as f64 / 1e9);
            println!("📦 Shards created: {}", result.shard_count);
            println!("💾 Total size: {:.2}GB", result.total_size_gb);
            
            if !skip_verify {
                println!("\n🔍 Verifying conversion...");
                crate::verify::verify_conversion(&result.output_path, false)?;
                println!("✅ Verification passed!");
            }
        }
        
        Commands::Verify { model, verbose } => {
            println!("🔍 TensorPort: Model Verification");
            println!("=================================");
            crate::verify::verify_conversion(&model, verbose)?;
        }
    }

    Ok(())
}