#!/bin/bash
# Auto-process latents workflow script
# This script monitors for new latent files from demo_gradio.py (AMD ROCm)
# and automatically processes them using process_saved_latents.py (NVIDIA CUDA)

set -e  # Exit on error

OUTPUTS_DIR="./outputs"
CONDA_ENV="hunyuan3d_21"
CONDA_PATH="/root/miniconda3/envs/${CONDA_ENV}"
WATCH_MODE="${1:-watch}"  # Default to watch mode, or use 'once' for single run

echo "=== FramePack Auto-Processing Script ==="
echo "Outputs directory: ${OUTPUTS_DIR}"
echo "CUDA Conda environment: ${CONDA_ENV}"
echo "Mode: ${WATCH_MODE}"
echo ""

# Function to process a latent file
process_latent_file() {
    local latent_file="$1"
    local job_id=$(basename "${latent_file}" _latents.pt)

    echo "========================================="
    echo "Processing: ${latent_file}"
    echo "Job ID: ${job_id}"
    echo "========================================="

    # Activate conda environment and run process_saved_latents.py
    echo "Activating conda environment: ${CONDA_ENV}..."
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV}"

    echo "Running VAE decode on CUDA GPU..."
    python process_saved_latents.py \
        --latents "${latent_file}" \
        --device cuda:0 \
        --output-dir "${OUTPUTS_DIR}"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully processed: ${job_id}_resume.mp4"
        echo "Output saved to: ${OUTPUTS_DIR}/${job_id}_resume.mp4"
    else
        echo "✗ Error processing ${latent_file} (exit code: ${exit_code})"
        return $exit_code
    fi

    conda deactivate
    echo ""
}

# Function to find and process the latest latent file
process_latest() {
    local latest_file=$(find "${OUTPUTS_DIR}" -name "*_latents.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "${latest_file}" ]; then
        echo "No latent files found in ${OUTPUTS_DIR}"
        return 1
    fi

    process_latent_file "${latest_file}"
}

# Function to watch for new latent files
watch_for_latents() {
    echo "Watching for new latent files in ${OUTPUTS_DIR}..."
    echo "Press Ctrl+C to stop"
    echo ""

    # Keep track of already processed files
    declare -A processed_files

    # Mark existing files as already processed
    for file in "${OUTPUTS_DIR}"/*_latents.pt; do
        if [ -f "$file" ]; then
            processed_files["$file"]=1
            echo "Skipping existing file: $(basename "$file")"
        fi
    done
    echo ""

    # Watch for new files
    while true; do
        for file in "${OUTPUTS_DIR}"/*_latents.pt; do
            if [ -f "$file" ] && [ -z "${processed_files[$file]}" ]; then
                echo "New latent file detected: $(basename "$file")"
                sleep 2  # Wait a bit to ensure file is fully written
                process_latent_file "$file"
                processed_files["$file"]=1
            fi
        done
        sleep 2  # Check every 2 seconds
    done
}

# Main execution
if [ "$WATCH_MODE" = "once" ]; then
    # Process the latest latent file once and exit
    process_latest
elif [ "$WATCH_MODE" = "watch" ]; then
    # Watch for new latent files continuously
    watch_for_latents
elif [ -f "$WATCH_MODE" ]; then
    # Process a specific file
    process_latent_file "$WATCH_MODE"
else
    echo "Usage: $0 [watch|once|<latent_file_path>]"
    echo ""
    echo "Modes:"
    echo "  watch          - Continuously watch for new latent files (default)"
    echo "  once           - Process the latest latent file and exit"
    echo "  <file_path>    - Process a specific latent file"
    exit 1
fi
