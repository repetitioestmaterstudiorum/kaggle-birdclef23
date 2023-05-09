# Manual Experiment Log

## Format (Template)

### Question

Experiment: ...

(Constraints: ...)

Results: ...

## Experiments

### What's the best architecture for our feature extractors?

Experiment: Try out various NN architectures (self-made), pretrained CNN and transformer models.

- Self-made RNN
- Self-made CNN
- ResNet18
- ResNet50
- Swin Transformer V2
- EfficientNet V2
- Regnet Y800MF

Constraints: Models with few than ~50M parameters

Results:

- Best results were obtained with EfficientNet, then RegNet, then the ResNets
- From the manual approaches, the RNN was better than a basic, self-made CNN, but with more work invested (and more articles read), a custom CNN was better

### What's the ideal audio sequence length?

Experiment: Run training with different length audio sequences, all else equal (batch_size 8, all training data, 5 epochs, 128 n_mels, 0.00008 lr)

Results:

- 8s: 37.36% accuracy (validation set)
- 10s: % accuracy (validation set)
- 15s: % accuracy (validation set)
- 20s: 49.73% accuracy (validation set)
- 24s (median): 48.35% accuracy (validation set)

Interestingly, the length of the audio file affects training time disproportionately. Meaning 20s files doesn't mean twice the training time compared to 10s. In fact training with 20s files resulted in ~13min per epoch, and 10s files in ~10min per epoch.

### NAS: Dropout after classifier layers?

Experiment: Run training with with and without dropout of 0.1, all else equal (batch_size 16, all training data, 10 epochs, 128 n_mels, 0.0002 lr)

Results:

- with: 51.01% accuracy (validation set)
- without: 55.46% accuracy (validation set)

### Hyperparameters: Batch size of 4, 8, 16, 32 with RegNet?

Experiment: Run training with different batch sizes and different learning rates and compare.

Results:

batch_size 32
0.00009 3.25% 0.00007 2.19% 0.00008 1.83% 0.0001 2.42% 0.0005 0.68%

batch_size 16
0.00009 2.95% 0.00007 3.57% 0.00008 3.25% 0.0001 3.31% 0.0005 4.48% 0.001 2.98%

batch_size 8
0.00009 5.32% 0.00007 4.05% 0.00008 5.79% 0.0001 4.90% 0.0005 2.89% 0.001 2.51%

batch_size 4
0.00009 5.40% 0.00007 5.76% 0.00008 5.91% 0.0001 4.13% 0.0005 2.95% 0.001 3.01%

It seems that lower batch sizes (4 and 8) yield best results.

### Optimizer: Weight decay or not?

Experiment: Run training with and without dropout of 0.1, all else equal (batch_size 16, all training data, 10 epochs, 128 n_mels, 0.0002 lr)

Results:

- with: 52.81% accuracy (validation set)
- without: 55.46% accuracy (validation set)

### Preprocessing of data: Pad audio files with copy of the audio or with 0s

(Some audio files are just a few seconds long. We need to pad these audio files (and truncate others) such that all audio files are of equal length for training with non-recurrent NNs.)

Experiment: Run training with padding options (all else equal: 20s, batch_size 8, 5 epochs, 128 n_mels, lr 0.00008):

- wrap: copy the audio until the desired length is reached
- constant 0: fill 0s until the desired length is reached

Results:

- wrap: 49.73% accuracy (validation set)
- constant 0: 46.93% accuracy (validation set)

### NAS: 2 or 3 hidden layers in the fully-connected classifier layers after the CNN?

Experiment: Run training all else equal (seconds: 10, batch_size: 8, data_percentage: 1, num_epochs: 3, n_mels: 128, learning_rate: 8e-05) with 2 or 3 layers.

Results:

- 2 layers:
- 3 layers:

### Will mirroring the audio files (e.g. 10s -> 20s with first 10s in reverse) improve results?

Experiment: Run training all else equal with 20s vs 10s mirrored.

Results:

- 20s:
- 10s mirrored:

### Will applying preprocessing in the model as opposed to in the dataset speed up training?

Experiment: Run training all else equal with the two approaches.

Results:

- In the dataset:
- In the model:
