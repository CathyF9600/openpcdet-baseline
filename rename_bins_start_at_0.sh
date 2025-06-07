#!/bin/bash

# Use this script to rename a set of files so that the numbering starts at 0 instead of 1

# Specify the directory containing the files
DIRECTORY="/home/perception/MLWorkspace/kevin/R2Y3/datasets/2023-11-18_3dod_data_collection_4_sample_cvat_data/bin_files"

cd "$DIRECTORY"

if [ $? -eq 0 ]; then
    for file in *.bin; do
      # Extract the number from filename
      num=$(echo $file | sed 's/^0*//;s/.bin$//')

      # Subtract 1 from the number
      new_num=$((num - 1))

      # Format the new number with leading zeros (total len 6) and add the extension
      new_file=$(printf "%06d.bin" $new_num)

      # Rename
      mv "$file" "$new_file"
    done
else
    echo "Failed to navigate to directory $DIRECTORY. Please check the path and try again."
fi
