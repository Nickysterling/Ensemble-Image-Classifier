# visualize_dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Function to display images with labels
def show_images(images, labels):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for idx, (img, label) in enumerate(zip(images, labels)):
        img = img.permute(1, 2, 0).numpy()  # Change channel order for plotting
        axes[idx].imshow(img)
        axes[idx].set_title(classes[label])
        axes[idx].axis('off')
    plt.show()

if __name__ == '__main__':
    # Convert PIL images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR-100 dataset
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Define CIFAR-100 label names for display
    classes = dataset.classes

    # Iterate over all batches in the training set
    for images, labels in data_loader:
        show_images(images, labels)