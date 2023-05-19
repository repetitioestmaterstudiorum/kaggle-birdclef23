# Manual Experiment Log

## Experiments

### What's the best architecture for our feature extractors?

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
- 10s: 42.85% accuracy (validation set)
- 15s: 46.60% accuracy (validation set)
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

Also interesting is that ideal learning rates are quote low for an adam optimizer (~0.00008).

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

### NAS: 2 or 3 hidden layers in the fully-connected classifier layers after the CNN?

Experiment: Run training all else equal (seconds: 10, batch_size: 8, data_percentage: 1, num_epochs: 5, n_mels: 128, learning_rate: 8e-05) with 2 or 3 layers.

Results:

- 2 layers:
  nn.Linear(self.regnet.fc.in_features, 1024),
  nn.BatchNorm1d(1024),
  nn.PReLU(),

  nn.Linear(1024, 512),
  nn.BatchNorm1d(512),
  nn.PReLU(),

  nn.Linear(512, num_classes)

- 3 layers: Train Accuracy: 52.77%, Valid Accuracy: 39.99%
  nn.Linear(self.regnet.fc.in_features, 1024),
  nn.BatchNorm1d(1024),
  nn.PReLU(),

  nn.Linear(1024, 1024),
  nn.BatchNorm1d(1024),
  nn.PReLU(),

  nn.Linear(1024, 512),
  nn.BatchNorm1d(512),
  nn.PReLU(),

  nn.Linear(512, num_classes),

### Will mirroring the audio files (e.g. 10s -> 20s with first 10s in reverse) improve results?

Experiment: Run training all else equal (batch_size 8, all training data, 5 epochs, 128 n_mels, 0.00008 lr) with 20s vs 10s mirrored.

Results:

- 20s: 49.73% accuracy (validation set)
- 10s mirrored: 45.78% accuracy (validation set)
- 15s mirrored: 46.31% accuracy (validation set)
- 20s mirrored: 52.60% accuracy (validation set)

Experiment: Run training with pad reflect, which mirrors the vector only for audio files that aren't long enough. (batch_size 8, 50% of training data, 2 epochs, 128 n_mels, 0.00008 lr)

Results:

- 10s wrap: 9% accuracy (validation set)
- 10s reflect: 11% accuracy (validation set)
- 20s wrap: 13% accuracy (validation set)
- 20s reflect: 12% accuracy (validation set)

Interesting results. Reflect is better for short cap sizes, wrap for larger ones.

### ~~Will applying preprocessing in the model as opposed to in the dataset speed up training?~~

~~Experiment: Run training all else equal with the two approaches.~~

~~Results:~~

~~- In the dataset:~~
~~- In the model:~~

### After realizing we just need to run inference within 120min and not training: efficientnet or regnet?

Experiment: Research.

Results:

- According to https://towardsdatascience.com/regnet-the-most-flexible-network-architecture-for-computer-vision-2fd757f9c5cd#:~:text=First%2C%20RegNets%20efficient%20performance%20versus%20the%20EfficientNet%20architecture.&text=Impressively%2C%20for%20all%20comparisons%2C%20RegNet,especially%20towards%20the%20lower%20end., RegNet is more efficient than EfficientNet. (Relevant graphic:
  https://miro.medium.com/v2/resize:fit:1400/format:webp/1*pKBw2ox6UT86t1tyn5lViQ.png)
- According to https://github.com/facebookresearch/pycls/issues/116, RegNets do not outperform Efficientnets
- According to https://medium.com/syncedreview/facebook-ai-regnet-models-outperform-efficientnet-models-run-5x-faster-on-gpus-7bdc3ea577ae, RegNets outperform Efficientnets _ON GPUS_

### Why does the new train notebook not learn?

Details:

- Training and validation functions are fine (verified in other notebook)
- 5s audios train on the other notebook
- Loading a well-trained model from the other notebook in this one spits out bad accuracy as well (~1.5%), but good (~43%) in other notebook
- Highly suspect: the new torchaudio melspec, power to db, and normalization functions

Experiment:

1. Train a model with 5s audios in an old, working notebook (so far, only loaded good models trained on longer sequences)
2. Verify low accuracy in new notebook
3. If still bad accuracy, use old librosa functions for preprocessing
4. Verify better accuracy in new notebook

_Optimize later_

- By default turn 5s audios into 10s audios with reflection
- Or even more with melspec to MFCC conversion (or others)

Results:

1. This worked, achieved ~35% accuracy (but overfitted)
2. Indeed, an initial validation set validation accuracy of just 2.2% was achieved
3. Instead of using librosa functions, I first adjusted torchaudio functions to librosa default settings. This didn't improvde results. I then adapted the old notebook such that only the Dataset class remained as it was before, meaning that the Dataset is the only effective difference (using Librosa in the old NB, Torchaudio in the new one). This notebook can still train a model. Subsequently, I decided to switch back to Librosa for preprocessing in the main (new) notebook.
4. Nothing changed!

After pondering about what the remaining differences in the code are I tested the following: setting the variable melspecs_per_audio to 1 which avoids the code that flattens the audio crops per audio recording. Suddenly, the model learned as well as in the old notebook!
To verify, I'm running the new notebook with torchaudio instead of librosa with melspecs_per_audio set to 1. Result: The model learns.
The torchaudio implementation runs much faster than the librosa one (~3 times faster), so this one's the winner.

### Does normalization of mel spectrograms/MFCCs lead to better results? (and do MFCCs lead to better results than mel spectrograms?)

Experiment: Run training using mel spectrograms all else equal with and without normalization.
Hyperparameters: batch_size: 8, data_percentage: 1, num_epochs: 4, n_mels: 128, learning_rate: 0.0001, pin_memory: True,
validate_on_train: True, device: mps, pad_method: wrap, validate_train_pct: 0.33

Results mel spectrograms:

- Without normalization: Train Loss: 0.5240, Valid Loss: 0.5496, Train Accuracy: 24.31%, Valid Accuracy: 21.90%
- With normalization: Train Loss: 0.5141, Valid Loss: 0.5358, Train Accuracy: 25.48%, Valid Accuracy: 24.71%

Experiment: Run training using MFCCs all else equal with and without normalization.

Results MFCCs:

- Without normalization: Train Loss: 0.5966, Valid Loss: 0.6272, Train Accuracy: 11.78%, Valid Accuracy: 10.11%
- With normalization: Train Loss: 0.6095, Valid Loss: 0.6293, Train Accuracy: 10.50%, Valid Accuracy: 9.49%

### When concatenating MFCCs and mel spectrograms, does normalizing mel spectrograms but not MFCCs lead to better results than normalizing both?

Experiment: Run training with normalized mel spectrograms, with and without normalization of MFCCs.

Hyperparameters: batch_size: 8, data_percentage: 1, num_epochs: 4, n_mels: 128, learning_rate: 0.0001, pin_memory: True,
validate_on_train: True, device: mps, pad_method: wrap, validate_train_pct: 0.33

Results (mel spectrograms normalized):

- Without MFCC normalization: Train Loss: 0.5274, Valid Loss: 0.5497, Train Accuracy: 23.45%, Valid Accuracy: 21.79%
- With MFCC normalization: Train Loss: 0.5204, Valid Loss: 0.5374, Train Accuracy: 27.19%, Valid Accuracy: 25.94%
- Normalizing both after concatenation: Train Loss: 0.5923, Valid Loss: 0.6020, Train Accuracy: 14.04%, Valid Accuracy: 13.86%

Clearly, normalizing separately was the best option. However, compared to just normalized mel spectrograms (Train Loss: 0.5141, Valid Loss: 0.5358, Train Accuracy: 25.48%, Valid Accuracy: 24.71%), the results are almost identical, so it's questionable whether the extra computations are worth it. Probably not, because with MFCCs there is a slight bit more overfitting.

### Does mirroring the audio before converting it to a mel spectrogram lead to better results?

Experiment: Run training with and without mirroring the audio before converting it to a mel spectrogram.

Hyperparameters: batch_size: 8, data_percentage: 1, num_epochs: 4, n_mels: 128, learning_rate: 0.0001, pin_memory: True,
validate_on_train: True, device: mps, pad_method: wrap, validate_train_pct: 0.33

Results:

- Without mirroring: Train Loss: 0.5141, Valid Loss: 0.5358, Train Accuracy: 25.48%, Valid Accuracy: 24.71%
- With mirroring: Train Loss: 0.5341, Valid Loss: 0.5508, Train Accuracy: 24.73%, Valid Accuracy: 22.67%
- With mirroring before melspec creation: Train Loss: 0.5177, Valid Loss: 0.5394, Train Accuracy: 26.54%, Valid Accuracy: 24.59%

... without mirroring results in the best accuracy.

### Compare different padding methods (again)

Experiment: Run training with wrap and with zeros padding.

Hyperparameters: batch_size: 8, data_percentage: 1, num_epochs: 4, n_mels: 128, learning_rate: 0.0001, pin_memory: True,
validate_on_train: True, device: mps, pad_method: wrap, validate_train_pct: 0.33

Results:

- With wrap padding: Train Loss: 0.5141, Valid Loss: 0.5358, Train Accuracy: 25.48%, Valid Accuracy: 24.71%
- With zeros padding: Train Loss: 0.5293, Valid Loss: 0.5451, Train Accuracy: 26.42%, Valid Accuracy: 25.59%

### Feature extractor search notes

Inference times **per audio** on trained models:

- RegNet Y 800MF (~6M params): ~4s
- EfficientNet V2 S (~22M params): ~10s
- ResNet 18 (~11M): ~4s

Accuracy after many epochs (when the model has converged):

- RegNet Y 800MF (~7M params): 55% valid, 63% train -> overfitted
- EfficientNet V2 S (~22M params): 21% (20 epochs)
- ResNet 18 (~11M): 40% valid, 45% train -> overfitted
- EVA-02 (eva02_base_patch14_448.mim_in22k_ft_in22k_in1k): 3% and decreasing, 1 epoch takes almost ~2h to train with validation
- EVA Tiny (eva02_tiny_patch14_336.mim_in22k_ft_in1k): 3% but increasing, but not very promising after 3 epochs
