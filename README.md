# kaggle-birdclef23

A solution to the BirdCLEF 2023 Kaggle challenge: https://www.kaggle.com/competitions/birdclef-2023/overview/description

## Team

Project team:

- https://github.com/repetitioestmaterstudiorum
- https://github.com/VaronLaStrauss
- https://github.com/tyroklan

This is also our final project of the Data Science and Machine Learning course at HSLU for the Master of Science in IT, Digitalization, and Sustainability.

## Requirements (Kaggle)

- Pytorch (not Tensorflow)
- CPU, not GPU
- Kaggle notebook runtime < 120min

## Interesting Public Notebooks

- https://www.kaggle.com/code/haydenismith/deep-audio-classification-birdclef-2023
- https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-infer#Methodology--%F0%9F%8E%AF
- https://www.kaggle.com/code/mattop/birdclef-2023-eda

## More Data

Could be found here:

- https://xeno-canto.org/explore/region

## Kaggle CLI

If the notebook is running in an environment other than on Kaggle, the kaggle cli is required to download the **data**. This guide can be followed (with reasonable exceptions): https://www.kaggle.com/general/74235

## Folder/Notebooks Guide

- additional_data: additional data for visualization purposes (https://github.com/tyroklan)
- audio-classification: contains the notebooks for the audio classification (https://github.com/repetitioestmaterstudiorum)
- audio-classification-yans: contains contains more notebooks for the audio classification (https://github.com/VaronLaStrauss)
- kaggle-dataset: automation to create new datasets (model weights and label encoder) on kaggle more quickly (https://github.com/repetitioestmaterstudiorum)
- csv-classification.ipynb: contains the notebooks for the csv classification (https://github.com/VaronLaStrauss)
- data_visualization.ipynb: contains the notebooks for the data visualization (https://github.com/tyroklan)
- example-nb-torchified.ipynb: sample notebook from Kaggle, with some functions changed from TF to Torch (https://github.com/repetitioestmaterstudiorum)
- utils.py: contains some utility functions (https://github.com/repetitioestmaterstudiorum)

**Most interesting files to look at:**

Understanding the data:

- data_visualization.ipynb
- audio-classification/data-analysis.ipynb

Preprocessing:

- audio-classification/spectrogram-playground.ipynb

Training Logs:

- audio-classification/experiment-log.md
- audio-classification-yans/experiment-log.md

Training Notebooks:

- RNN approach: audio-classification/rnn-audio-as-vector.ipynb
- Custom CNN: audio-classification/cnn-custom-audio-as-spectrogram.ipynb
- Audio Spectrogram Transformer: audio-classification/audio-spectrogram-transformer.ipynb
- MobileNet: audio-classification-yans/training_mobilenet_wavelets.ipynb
- EfficientNet: audio-classification/cnn-efficientnetV2S-audio-as-spectrogram.ipynb
- ResNet: audio-classification/training-resnet18.ipynb
- RegNet: audio-classification/training-regnet.ipynb (best results)
- Multi-channel approach: audio-classification-yans/training-regnet-wavelet-melspec-default-fc.ipynb
- Wavelet approach: audio-classification-yans/Yans2dMelWavelet.ipynb
- Multi approach: audio-classification-yans/Yans2dMulti.ipynb

Submission Notebooks:

- 4 channel (melcpec, mfcc, wavelet low & high): audio-classification/submission-regnet-4C-test.ipynb
- 1 channel: audio-classification/submission-regnet.ipynb
