# test.py

import os
import sys

import argparse

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import transformation, file_paths, get_dataloaders
from models.model_factory import load_model
from test_utils import compute_overall_statistics, test

paths = file_paths()
PROJECT_ROOT = paths["PROJECT_ROOT"]

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Test individual models for CIFAR-100")
    
    parser.add_argument('--m', type=str, required=True, choices=['alexnet', 'vgg16', 'resnet18'], 
                        help="Model type to test (alexnet, vgg16, resnet18)")
    
    parser.add_argument('--w', type=str, required=True,
                        help="Path to the saved model weights file (.pth)")
    
    return parser.parse_args()

# Main function
def main():
    args = parse_args()

    args.w = os.path.abspath(os.path.join(PROJECT_ROOT, args.w))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transformation()
    
    _, dataloader = get_dataloaders(transform, 1, dataset_name="CIFAR100")

    # Load the model
    model = load_model(args.m, args.w, device)
    model.eval()

    # Compute overall statistics
    compute_overall_statistics(model, dataloader, device, ensemble=False)

    classes = dataloader.dataset.classes
    test(model, dataloader, classes, device, ensemble=False)

if __name__ == "__main__":
    main()
