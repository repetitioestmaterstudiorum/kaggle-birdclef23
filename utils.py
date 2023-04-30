import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# ---

def get_is_in_kaggle_env():
    is_in_kaggle_env = False
    run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
    
    if run_type:
        is_in_kaggle_env = True
        if run_type == 'Interactive':
            print("We are running a Kaggle Notebook/Script - Interactive Mode")
        elif run_type == 'Batch':
            print("We are running a Kaggle Notebook/Script - Batch Mode")
        else:
            print("We are running a Kaggle Notebook/Script - Could be Interactive or Batch Mode")
    else:
        print("We are running code on Localhost")
    
    return is_in_kaggle_env


def determine_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" # Apple M1

    print(f"We are using device: {device}")
    return device


def normalize_spectrogram(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    if max_val - min_val == 0:
        return spectrogram
    else:
        return (spectrogram - min_val) / (max_val - min_val)
    

def plot_train_and_valid_loss_and_accuracy(train_losses, valid_losses, train_accuracies, valid_accuracies):
    train_accuracies = [x.cpu() for x in train_accuracies]
    valid_accuracies = [x.cpu() for x in valid_accuracies]
    train_losses = [x.cpu() for x in train_losses]
    valid_losses = [x.cpu() for x in valid_losses]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    if len(train_losses) == 1 and len(valid_losses) == 1:
        ax1.scatter(0, train_losses[0], label='Training Loss')
        ax1.scatter(0, valid_losses[0], label='Validation Loss')
    else:
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(valid_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    if len(train_accuracies) == 1 and len(valid_accuracies) == 1:
        ax2.scatter(0, train_accuracies[0], label='Training Accuracy')
        ax2.scatter(0, valid_accuracies[0], label='Validation Accuracy')
    else:
        ax2.plot(train_accuracies, label='Training Accuracy')
        ax2.plot(valid_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()


def plot_minibatch_loss(minibatch_loss):
    plt.plot(range(len(minibatch_loss)), minibatch_loss)
    plt.title('Minibatch Loss')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Minibatch')
    plt.grid()
    plt.show()
