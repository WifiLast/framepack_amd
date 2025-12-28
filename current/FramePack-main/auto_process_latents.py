#!/usr/bin/env python3
"""
Auto-process latents workflow script for FramePack
Monitors for new latent files from demo_gradio.py (AMD ROCm)
and automatically processes them using process_saved_latents.py (NVIDIA CUDA)
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Set, Optional


class LatentFileMonitor:
    """Monitor and auto-process latent files."""

    def __init__(
        self,
        outputs_dir: str = "./outputs",
        conda_env: str = "hunyuan3d_21",
        device: str = "cuda:0",
        enable_slicing: bool = False,
        disable_tiling: bool = False,
        clear_cache: bool = False,
    ):
        self.outputs_dir = Path(outputs_dir)
        self.conda_env = conda_env
        self.device = device
        self.enable_slicing = enable_slicing
        self.disable_tiling = disable_tiling
        self.clear_cache = clear_cache
        self.processed_files: Set[Path] = set()

        # Verify outputs directory exists
        if not self.outputs_dir.exists():
            raise FileNotFoundError(f"Outputs directory not found: {self.outputs_dir}")

    def find_latest_latent_file(self) -> Optional[Path]:
        """Find the most recently modified latent file."""
        latent_files = list(self.outputs_dir.glob("*_latents.pt"))
        if not latent_files:
            return None
        return max(latent_files, key=lambda p: p.stat().st_mtime)

    def clear_pytorch_cache(self):
        """Clear PyTorch compilation caches to prevent cross-environment interference."""
        if not self.clear_cache:
            return

        print("Clearing PyTorch caches...")
        cache_paths = [
            Path.home() / ".triton" / "cache",
            Path("/tmp") / "__pycache__",
            Path("/tmp") / "torch_extensions",
        ]

        for cache_path in cache_paths:
            if cache_path.exists():
                try:
                    subprocess.run(["rm", "-rf", str(cache_path / "*")], shell=True)
                    print(f"  Cleared: {cache_path}")
                except Exception as e:
                    print(f"  Warning: Could not clear {cache_path}: {e}")

        print("Cache clearing complete\n")

    def process_latent_file(self, latent_file: Path) -> bool:
        """Process a single latent file using the CUDA environment."""
        print("=" * 60)
        print(f"Processing: {latent_file.name}")
        print(f"Job ID: {latent_file.stem.replace('_latents', '')}")
        print("=" * 60)

        # Clear caches before processing if requested
        if self.clear_cache:
            self.clear_pytorch_cache()

        # Build command to run in conda environment
        cmd = [
            "bash", "-c",
            f"source /root/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            f"python process_saved_latents.py "
            f"--latents {latent_file} "
            f"--device {self.device} "
            f"--output-dir {self.outputs_dir}"
        ]

        if self.enable_slicing:
            cmd[-1] += " --enable-slicing"
        if self.disable_tiling:
            cmd[-1] += " --disable-tiling"

        print(f"Running: {' '.join(cmd[-1].split('&&')[-1].strip().split())}")
        print()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
            )

            job_id = latent_file.stem.replace('_latents', '')
            output_video = self.outputs_dir / f"{job_id}_resume.mp4"

            print()
            print(f"✓ Successfully processed: {output_video.name}")
            print(f"Output saved to: {output_video}")

            # Clear caches after processing if requested
            if self.clear_cache:
                self.clear_pytorch_cache()

            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing {latent_file.name} (exit code: {e.returncode})")
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
        finally:
            print()

    def process_latest_once(self) -> bool:
        """Process the latest latent file once and exit."""
        latest_file = self.find_latest_latent_file()

        if latest_file is None:
            print(f"No latent files found in {self.outputs_dir}")
            return False

        return self.process_latent_file(latest_file)

    def watch_and_process(self, check_interval: float = 2.0):
        """Continuously watch for new latent files and process them."""
        print("=== FramePack Auto-Processing Monitor ===")
        print(f"Outputs directory: {self.outputs_dir}")
        print(f"CUDA device: {self.device}")
        print(f"Conda environment: {self.conda_env}")
        print(f"Check interval: {check_interval}s")
        print()
        print("Watching for new latent files...")
        print("Press Ctrl+C to stop")
        print()

        # Mark existing files as already processed
        existing_files = list(self.outputs_dir.glob("*_latents.pt"))
        for file in existing_files:
            self.processed_files.add(file)
            print(f"Skipping existing file: {file.name}")

        if existing_files:
            print()

        try:
            while True:
                # Check for new latent files
                current_files = set(self.outputs_dir.glob("*_latents.pt"))
                new_files = current_files - self.processed_files

                for new_file in sorted(new_files, key=lambda p: p.stat().st_mtime):
                    print(f"New latent file detected: {new_file.name}")
                    time.sleep(1)  # Wait a bit to ensure file is fully written
                    self.process_latent_file(new_file)
                    self.processed_files.add(new_file)

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print()
            print("Monitoring stopped by user")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-process FramePack latent files using CUDA environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch for new latent files continuously
  python auto_process_latents.py --mode watch

  # Process the latest latent file once
  python auto_process_latents.py --mode once

  # Process a specific latent file
  python auto_process_latents.py --mode file --latents outputs/20250101_123456_latents.pt

  # Use specific GPU with slicing enabled
  python auto_process_latents.py --mode watch --device cuda:1 --enable-slicing
        """
    )

    parser.add_argument(
        "--mode",
        choices=["watch", "once", "file"],
        default="watch",
        help="Operating mode: watch (continuous), once (latest file), or file (specific file)"
    )
    parser.add_argument(
        "--latents",
        type=str,
        help="Specific latent file to process (required when mode=file)"
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="./outputs",
        help="Directory containing latent files (default: ./outputs)"
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
        "--check-interval",
        type=float,
        default=2.0,
        help="Check interval in seconds for watch mode (default: 2.0)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear PyTorch caches before and after processing to prevent AMD/NVIDIA interference"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "file" and not args.latents:
        parser.error("--latents is required when mode=file")

    # Create monitor
    monitor = LatentFileMonitor(
        outputs_dir=args.outputs_dir,
        conda_env=args.conda_env,
        device=args.device,
        enable_slicing=args.enable_slicing,
        disable_tiling=args.disable_tiling,
        clear_cache=args.clear_cache,
    )

    # Execute based on mode
    if args.mode == "watch":
        monitor.watch_and_process(check_interval=args.check_interval)
    elif args.mode == "once":
        success = monitor.process_latest_once()
        sys.exit(0 if success else 1)
    elif args.mode == "file":
        latent_file = Path(args.latents)
        if not latent_file.exists():
            print(f"Error: Latent file not found: {latent_file}")
            sys.exit(1)
        success = monitor.process_latent_file(latent_file)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
