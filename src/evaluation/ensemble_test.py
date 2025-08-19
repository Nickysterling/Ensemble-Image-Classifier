# ensemble_test.py

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
    parser = argparse.ArgumentParser(description="Test ensemble methods for CIFAR-100")
    parser.add_argument('--an', type=str, required=True,
                        help="Path to the saved weights file for AlexNet")
    
    parser.add_argument('--vgg', type=str, required=True,
                        help="Path to the saved weights file for VGG16")
    
    parser.add_argument('--res', type=str, required=True,
                        help="Path to the saved weights file for ResNet18")

    parser.add_argument('--m', type=str, required=True, choices=["max_prob", "avg_prob", "majority_vote"],
                        help="Ensemble method to use (max_prob, avg_prob, majority_vote)")
    return parser.parse_args()

def main():
    args = parse_args()

    args.an = os.path.abspath(os.path.join(PROJECT_ROOT, args.an))
    args.vgg = os.path.abspath(os.path.join(PROJECT_ROOT, args.vgg))
    args.res = os.path.abspath(os.path.join(PROJECT_ROOT, args.res))

    # Set up the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the CIFAR-100 dataset
    transform = transformation()
    
    _, dataloader = get_dataloaders(transform=transform, batch_size=1, dataset_name="CIFAR100")

    # Load the models individually
    print(f"\nLoading AlexNet model from {args.an}...")
    alexnet = load_model("alexnet", args.an, device)

    print(f"Loading VGG16 model from {args.vgg}...")
    vgg16 = load_model("vgg16", args.vgg, device)

    print(f"Loading ResNet18 model from {args.res}...")
    resnet18 = load_model("resnet18", args.res, device)

    # List of models for ensemble methods
    models = [alexnet, vgg16, resnet18]

    # Get the class names
    classes = dataloader.dataset.classes

    # Compute overall statistics for the ensemble
    compute_overall_statistics(models=models, dataloader=dataloader, device=device, ensemble=True, method=args.m)

    classes = dataloader.dataset.classes
    test(models=models, dataloader=dataloader, classes=classes, device=device, ensemble=True, method=args.m)


if __name__ == "__main__":
    main()
