#!/bin/bash

# Define input and output directories
SOURCE_DIR="../../../clips_extended"
OUTPUT_DIR="../../../clips_extended_output"

# Loop over all .mp4 files in the source directory
for file in "$SOURCE_DIR"/*.mp4; do
    # Extract filename without path
    filename=$(basename "$file")

    # Construct output file path
    output_file="$OUTPUT_DIR/$filename"

    # Run the Python command
    python main.py --source_video_path "$file" --target_video_path "$output_file" --device cuda --mode PLAYER_TRACKING
done