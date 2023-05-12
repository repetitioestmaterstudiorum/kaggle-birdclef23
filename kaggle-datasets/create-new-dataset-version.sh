#!/bin/bash

# Automatic creation of new dataset version for https://www.kaggle.com/datasets/vollernom/inference
# To run this script, make it executable first: chmod +x create-new-dataset-version.sh
# Then run with `python create-new-dataset-version.bash`

# ensure the script is executed in the directory where it is located
cd "$(dirname "$0")"

# move best_model.pt and label_encoder.joblib from ../audio-classification to ./inference
cp ../audio-classification/best_model.pt ./inference
cp ../audio-classification/label_encoder.joblib ./inference
cp ../utils.py ./inference

# test
cd inference
kaggle datasets version -m "newest versions of utils.py, best_model.pt and label_encoder.joblib"
