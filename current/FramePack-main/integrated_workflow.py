#!/usr/bin/env python3
"""
Integrated workflow script for FramePack
Automatically triggers CUDA processing after AMD ROCm latent generation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def trigger_cuda_processing(
    latent_file: str,
    conda_env: str = "hunyuan3d_21",
    device: str = "cuda:0",
    outputs_dir: str = "./outputs",
    enable_slicing: bool = False,
    disable_tiling: bool = False,
    quiet: bool = False,
) -> bool:
    """
    Trigger CUDA processing of a latent file in a separate conda environment.

    Args:
        latent_file: Path to the latent file
        conda_env: Name of the conda environment with CUDA setup
        device: CUDA device to use
        outputs_dir: Output directory for the video
        enable_slicing: Enable VAE slicing
        disable_tiling: Disable VAE tiling
        quiet: Suppress output

    Returns:
        True if successful, False otherwise
    """
    latent_path = Path(latent_file)
    if not latent_path.exists():
        print(f"Error: Latent file not found: {latent_file}")
        return False

    print()
    print("=" * 70)
    print("TRIGGERING CUDA PROCESSING")
    print("=" * 70)
    print(f"Latent file: {latent_path.name}")
    print(f"Conda env: {conda_env}")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    # Build the command
    cmd_parts = [
        f"source /root/miniconda3/etc/profile.d/conda.sh",
        f"conda activate {conda_env}",
        f"python process_saved_latents.py",
        f"--latents {latent_file}",
        f"--device {device}",
        f"--output-dir {outputs_dir}",
    ]

    if enable_slicing:
        cmd_parts.append("--enable-slicing")
    if disable_tiling:
        cmd_parts.append("--disable-tiling")
    if quiet:
        cmd_parts.append("--quiet")

    full_cmd = " && ".join(cmd_parts)

    try:
        # Run in bash shell
        result = subprocess.run(
            ["bash", "-c", full_cmd],
            check=True,
            capture_output=quiet,
            text=True,
        )

        job_id = latent_path.stem.replace('_latents', '')
        output_video = Path(outputs_dir) / f"{job_id}_resume.mp4"

        print()
        print("=" * 70)
        print("✓ CUDA PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Output video: {output_video}")
        print("=" * 70)
        print()

        return True

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print(f"✗ CUDA PROCESSING FAILED (exit code: {e.returncode})")
        print("=" * 70)
        if quiet and e.stderr:
            print(e.stderr)
        print()
        return False

    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ UNEXPECTED ERROR: {e}")
        print("=" * 70)
        print()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Integrated workflow: AMD latent generation → CUDA video decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script is designed to be called automatically after demo_gradio.py finishes
generating latents on AMD ROCm. It will switch to a CUDA conda environment and
process the latents to generate the final video.

Examples:
  # Process a specific latent file
  python integrated_workflow.py outputs/20250101_123456_latents.pt

  # Use specific GPU with slicing
  python integrated_workflow.py outputs/20250101_123456_latents.pt \\
      --device cuda:1 --enable-slicing

  # Use custom conda environment
  python integrated_workflow.py outputs/20250101_123456_latents.pt \\
      --conda-env my_cuda_env
        """
    )

    parser.add_argument(
        "latent_file",
        type=str,
        help="Path to the latent file to process"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="hunyuan3d_21",
        help="Conda environment name for CUDA processing (default: hunyuan3d_21)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="./outputs",
        help="Output directory for the video (default: ./outputs)"
    )
    parser.add_argument(
        "--enable-slicing",
        action="store_true",
        help="Enable VAE slicing for lower VRAM usage"
    )
    parser.add_argument(
        "--disable-tiling",
        action="store_true",
        help="Disable VAE tiling (not recommended)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress processing output"
    )

    args = parser.parse_args()

    success = trigger_cuda_processing(
        latent_file=args.latent_file,
        conda_env=args.conda_env,
        device=args.device,
        outputs_dir=args.outputs_dir,
        enable_slicing=args.enable_slicing,
        disable_tiling=args.disable_tiling,
        quiet=args.quiet,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
