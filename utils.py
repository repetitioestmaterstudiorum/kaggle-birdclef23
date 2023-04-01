import os
import torch

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
