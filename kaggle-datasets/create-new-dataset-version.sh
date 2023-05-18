#!/bin/bash

# Automatic creation of new dataset version for https://www.kaggle.com/datasets/vollernom/inference
# To run this script, make it executable first: chmod +x create-new-dataset-version.sh
# Then run with `python create-new-dataset-version.bash`

# Ensure the script is executed in the directory where it is located
cd "$(dirname "$0")"

# Remove existing models
rm ./inference/*.pt

# Copy the latest model, label_encoder.joblib from ../audio-classification to ./inference
latest_model=$(ls -t ../audio-classification/5s*.pt | head -n 1)
cp "$latest_model" ./inference
cp ../audio-classification/label_encoder.joblib ./inference

# Copy the latest version of utils.py from ../utils to ./inference 
cp ../utils.py ./inference

# Create new dataset version
cd inference
kaggle datasets version -m "newest versions of utils.py, model and label encoder"
