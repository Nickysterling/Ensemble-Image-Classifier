# test.py

import torch
from torch import nn
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
import argparse

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

# Get the Top-1 and Top-5 predictions
def get_top_predictions(output, classes, top_k=5):
    _, top_indices = torch.topk(output, top_k, dim=1)
    return [(classes[idx], output[0, idx].item()) for idx in top_indices[0]]

# Test a single image by index
def test_single_image(model, dataloader, classes, device, index):
    img, lbl = dataloader.dataset[index]
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
        predictions = get_top_predictions(output, classes)

    top1_correct = predictions[0][0] == classes[lbl]
    top5_correct = any(pred[0] == classes[lbl] for pred in predictions)

    print(f"\nImage Index: {index}")
    print(f"True Label: {classes[lbl]}")
    print(f"Top-1 Prediction: {predictions[0][0]} ({'Correct' if top1_correct else 'Incorrect'})")
    print("Top-5 Predictions:", ", ".join([pred[0] for pred in predictions]))
    print(f"Top-5 Accuracy: {'Correct' if top5_correct else 'Incorrect'}")

    # Visualize the image with predictions
    img_np = img.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title(f"True Label: {classes[lbl]}\nTop-1: {predictions[0][0]}")
    plt.axis('off')
    plt.show()

# Compute overall accuracy statistics
def compute_overall_statistics(model, dataloader, device):
    total_top1, total_top5 = 0, 0
    total_images = len(dataloader.dataset)
    all_classes = dataloader.dataset.classes

    with torch.no_grad():
        for img, lbl in dataloader:
            img = img.to(device)
            lbl = lbl.to(device)
            output = model(img)
            predictions = torch.topk(output, k=5, dim=1).indices
            total_top1 += (predictions[:, 0] == lbl).sum().item()
            total_top5 += (predictions == lbl.unsqueeze(1)).sum().item()

    print("\nOverall Statistics:")
    print(f"Top-1 Accuracy: {total_top1 / total_images * 100:.2f}%")
    print(f"Top-5 Accuracy: {total_top5 / total_images * 100:.2f}%")

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Test individual models for CIFAR-100")
    parser.add_argument('--model_type', type=str, required=True, choices=['alexnet', 'vgg16', 'resnet18'],
                        help="Model type to test (alexnet, vgg16, resnet18)")
    parser.add_argument('--weights_path', type=str, required=True,
                        help="Path to the saved model weights file (.pth)")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the CIFAR-100 dataset directory")
    return parser.parse_args()

# Main function
def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CIFAR-100 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the model
    model = load_model(args.model_type, args.weights_path, device)
    classes = dataset.classes

    # Compute overall statistics
    compute_overall_statistics(model, dataloader, device)

    # Allow user to select an image to test
    total_images = len(dataset)
    while True:
        try:
            idx = int(input(f"\nEnter image index (0 to {total_images - 1}): "))
            if 0 <= idx < total_images:
                test_single_image(model, dataloader, classes, device, idx)
            else:
                print("Invalid index, try again.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
