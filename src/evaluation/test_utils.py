# test_utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt


# Show an image with its true and predicted labels
def visualize_image(img_tensor, true_label, predicted_label, classes):
    img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    plt.imshow(img_np)
    plt.title(f"True Label: {classes[true_label]}\nTop-1: {classes[predicted_label]}")
    plt.axis('off')
    plt.show()


# Combine predictions from multiple models using the specified ensemble method
def ensemble_predictions(outputs, method="max_prob"):
    if method == "max_prob":
        return torch.max(torch.stack(outputs), dim=0)[0]
    elif method == "avg_prob":
        return torch.mean(torch.stack(outputs), dim=0)
    elif method == "majority_vote":
        votes = torch.stack([torch.argmax(output, dim=1) for output in outputs])
        ensemble_output = torch.mode(votes, dim=0).values
        # Convert to one-hot to match other methods
        num_classes = outputs[0].shape[1]
        one_hot = torch.zeros(votes.size(1), num_classes, device=votes.device)
        one_hot.scatter_(1, ensemble_output.unsqueeze(1), 1)
        return one_hot
    else:
        raise ValueError(f"Invalid ensemble method: {method}")


# Predict a single image and display Top-1 and Top-5 results
def get_predictions(models, dataloader, classes, device, index, ensemble=False, method="max_prob"):
    img, lbl = dataloader.dataset[index]
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        if ensemble:
            outputs = [model(img) for model in models]
            output = ensemble_predictions(outputs, method=method)
        else:
            output = models(img)

    # Top-5 predictions
    predictions = torch.topk(output, k=5, dim=1).indices[0]
    top1_correct = predictions[0] == lbl
    top5_correct = lbl in predictions

    # Print results
    print(f"\nImage Index: {index}")
    print(f"True Label: {classes[lbl]}")
    print(f"Top-1 Accuracy: {'Correct' if top1_correct else 'Incorrect'}")
    print(f"Top-5 Accuracy: {'Correct' if top5_correct else 'Incorrect'}")
    print(f"Top-1 Prediction: {classes[predictions[0]]}")
    print("Top-5 Predictions:", ", ".join([classes[p] for p in predictions]))

    visualize_image(img, lbl, predictions[0], classes)


# Compute Top-1 and Top-5 accuracy for whole dataset
def compute_overall_statistics(models, dataloader, device, ensemble=False, method="max_prob"):
    total_top1, total_top5 = 0, 0
    total_images = len(dataloader.dataset)

    with torch.no_grad():
        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)

            if ensemble:
                outputs = [model(img) for model in models]
                output = ensemble_predictions(outputs, method=method)
            else:
                output = models(img)

            predictions = torch.topk(output, k=5, dim=1).indices
            total_top1 += (predictions[:, 0] == lbl).sum().item()
            total_top5 += (predictions == lbl.unsqueeze(1)).sum().item()

    top1_acc = total_top1 / total_images * 100
    top5_acc = total_top5 / total_images * 100
    print(f"Overall Statistics:")
    print(f"Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%")
    return top1_acc, top5_acc


# Interactive loop to test images by index
def test(models, dataloader, classes, device, ensemble=False, method="max_prob"):
    total_images = len(dataloader.dataset)
    while True:
        try:
            idx = int(input(f"\nEnter image index (0 to {total_images - 1}): "))
            if 0 <= idx < total_images:
                get_predictions(models, dataloader, classes, device, idx, ensemble, method)
            else:
                print("Invalid index, try again.")
        except ValueError:
            print("Please enter a valid number.")
