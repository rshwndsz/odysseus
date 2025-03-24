#!/bin/bash

# Get the model file from the first CLI argument
MODEL_FILE="$1"

# Check if the argument is provided
if [[ -z "$MODEL_FILE" ]]; then
  echo "Error: No model file provided. Usage: $0 <model_file>"
  exit 1
fi

# Check if the file exists
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Error: $MODEL_FILE does not exist."
  exit 1
fi

# Loop through each line in the file
while IFS= read -r model || [[ -n "$model" ]]; do
  # Skip empty lines
  if [[ -z "$model" ]]; then
    continue
  fi

  echo "Downloading model: $model"
  huggingface-cli download "$model"

  # Check if the download command succeeded
  if [[ $? -ne 0 ]]; then
    echo "Error: Failed to download model: $model"
  fi
done < "$MODEL_FILE"
