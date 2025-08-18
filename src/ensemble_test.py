# ensemble_test.py

import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms, datasets, models

# Load the model and its weights
def load_model(model_type, weights_path, device):
    if model_type == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 100)
        )
    elif model_type == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 100)
        )
    elif model_type == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 100)
    else:
        raise ValueError("Invalid model type")  

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Ensemble methods
def ensemble_predictions(outputs, method="max_prob"):
    if method == "max_prob":
        ensemble_output = torch.max(torch.stack(outputs), dim=0)[0]
    
    elif method == "avg_prob":
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
    
    elif method == "majority_vote":
        # Class predictions (votes) from each model
        votes = torch.stack([torch.argmax(output, dim=1) for output in outputs])
        
        # Compute the most frequent prediction (mode) for each sample
        ensemble_output = torch.mode(votes, dim=0).values
        
        # Convert to one-hot encoding to match other methods' output shape
        num_classes = outputs[0].shape[1]
        one_hot = torch.zeros(votes.size(1), num_classes, device=votes.device)
        one_hot.scatter_(1, ensemble_output.unsqueeze(1), 1)
        return one_hot
    else:
        raise ValueError("Invalid ensemble method")
    return ensemble_output

# Test a single image using ensemble
def test_single_image_ensemble(models, dataloader, classes, device, index, method):
    img, lbl = dataloader.dataset[index]
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = [model(img) for model in models]
        ensemble_output = ensemble_predictions(outputs, method=method)

    predictions = torch.topk(ensemble_output, k=5, dim=1).indices[0]
    top1_correct = predictions[0] == lbl
    top5_correct = lbl in predictions

    print(f"\nImage Index: {index}")
    print(f"True Label: {classes[lbl]}")
    print(f"Top-1 Prediction: {classes[predictions[0]]} ({'Correct' if top1_correct else 'Incorrect'})")
    print("Top-5 Predictions:", ", ".join([classes[p] for p in predictions]))
    print(f"Top-5 Accuracy: {'Correct' if top5_correct else 'Incorrect'}")

    # Visualize the image with predictions
    img_np = img.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title(f"True Label: {classes[lbl]}\nTop-1: {classes[predictions[0]]}")
    plt.axis('off')
    plt.show()

# Compute overall statistics for ensemble
def compute_overall_statistics_ensemble(models, dataloader, device, method):
    total_top1, total_top5 = 0, 0
    total_images = len(dataloader.dataset)
    all_classes = dataloader.dataset.classes

    with torch.no_grad():
        for img, lbl in dataloader:
            img = img.to(device)
            lbl = lbl.to(device)

            outputs = [model(img) for model in models]
            ensemble_output = ensemble_predictions(outputs, method=method)
            predictions = torch.topk(ensemble_output, k=5, dim=1).indices

            total_top1 += (predictions[:, 0] == lbl).sum().item()
            total_top5 += (predictions == lbl.unsqueeze(1)).sum().item()

    print("\nOverall Statistics:")
    print(f"Top-1 Accuracy: {total_top1 / total_images * 100:.2f}%")
    print(f"Top-5 Accuracy: {total_top5 / total_images * 100:.2f}%")

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Test ensemble methods for CIFAR-100")
    parser.add_argument('--alexnet_weights', type=str, required=True,
                        help="Path to the saved weights file for AlexNet")
    parser.add_argument('--vgg16_weights', type=str, required=True,
                        help="Path to the saved weights file for VGG16")
    parser.add_argument('--resnet18_weights', type=str, required=True,
                        help="Path to the saved weights file for ResNet18")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the CIFAR-100 dataset directory")
    parser.add_argument('--method', type=str, required=True, choices=["max_prob", "avg_prob", "majority_vote"],
                        help="Ensemble method to use (max_prob, avg_prob, majority_vote)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set up the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the CIFAR-100 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the models individually
    print(f"Loading AlexNet model from {args.alexnet_weights}...")
    alexnet = load_model("alexnet", args.alexnet_weights, device)

    print(f"Loading VGG16 model from {args.vgg16_weights}...")
    vgg16 = load_model("vgg16", args.vgg16_weights, device)

    print(f"Loading ResNet18 model from {args.resnet18_weights}...")
    resnet18 = load_model("resnet18", args.resnet18_weights, device)

    # List of models for ensemble methods
    models = [alexnet, vgg16, resnet18]

    # Get the class names
    classes = dataset.classes

    # Compute overall statistics for the ensemble
    compute_overall_statistics_ensemble(models, dataloader, device, args.method)

    # Interactive single image testing
    total_images = len(dataset)
    while True:
        try:
            idx = int(input(f"\nEnter image index (0 to {total_images - 1}): "))
            if 0 <= idx < total_images:
                test_single_image_ensemble(models, dataloader, classes, device, idx, args.method)
            else:
                print("Invalid index, try again.")
        except ValueError:
            print("Please enter a valid number.")



if __name__ == "__main__":
    main()
