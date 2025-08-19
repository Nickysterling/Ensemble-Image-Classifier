# train.py

import os
import sys

import time
import datetime
import argparse

import matplotlib.pyplot as plt

import torch
from torch.amp import autocast, GradScaler
from torchsummary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import transformation, file_paths, get_dataloaders, setup_components
from models.model_factory import load_model

# -----------------------------
# Default configuration
# -----------------------------
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2


paths = file_paths()
DATA_DIR = paths["DATA_DIR"]
OUTPUT_DIR = paths["OUTPUT_DIR"]
MODEL_DIR = paths["MODEL_DIR"]
PLOT_DIR = paths["PLOT_DIR"]
LOG_DIR = paths["LOG_DIR"]


# Initialize the GradScaler
scaler = GradScaler()

# ------------------------------
# Utility Functions
# ------------------------------

def log_details(log_path, message):
    with open(log_path, "a") as f:
        f.write(message + "\n")
    print(f"Saved log to: {log_path}")


def save_model(model, path, best_val_loss, current_val_loss):
    if current_val_loss < best_val_loss:
        torch.save(model.state_dict(), path)
        print(f"Saved model to: {path}")
        return current_val_loss
    return best_val_loss


# Plot train and val loss
def plot_losses(train_losses, val_losses, total_val_images, total_training_time, 
                top1_acc, top5_acc, plot_path):
    plt.figure(figsize=(12, 7))
    plt.clf()

    # Plot avg train and val losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)

    minutes, seconds = divmod(total_training_time, 60)
    avg_time_per_image = (total_training_time / total_val_images) * 1000

    # Add stats at bottom of plot
    stats_text = (f"Total Runtime: {int(minutes)} min, {int(seconds)} sec, Avg Time/Test Image: {avg_time_per_image:.2f} ms\n"
                    f"Avg Train Loss: {train_losses[-1]:.4f}, Avg Val Loss: {val_losses[-1]:.4f}\n"
                    f"Top-1 Acc: {top1_acc:.2f}%, Top-5 Acc: {top5_acc:.2f}%")

    # Position text at bottom of figure
    plt.figtext(0.5, -0.05, stats_text, wrap=True, horizontalalignment='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5))

    # Add title
    plt.title(f"Training Results\nLoss Plot: {plot_path}")
    
    print('Saving plot to:', plot_path, '\n')
    plt.savefig(plot_path, bbox_inches="tight")


# ------------------------------
# Training Loop
# ------------------------------

def train_model(model, optimizer, scheduler, loss_fn, train_loader, val_loader,
                device='cpu', n_epochs=30, model_path=None, plot_path=None, log_path=None):
    
    # Lists to hold training and validation losses for each epoch
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    # Start timer for the whole training process
    training_start_time = time.time()

    # Loop through total num of epochs
    for epoch in range(1, n_epochs + 1):
        timestamp = datetime.datetime.now().strftime("%I:%M:%S %p")
        print(f"\n[{timestamp}] Epoch {epoch}/{n_epochs}")

        # Epoch variables
        running_train_loss = 0.0
        running_val_loss = 0.0
        
        # Error rate tracking
        top1_correct = 0
        top5_correct = 0
        total = 0
        
        # Training loop
        model.train()
        for data in train_loader:
            img, lbl = data
            img = img.float().to(device=device)
            lbl = lbl.long().to(device=device)

            optimizer.zero_grad()  # Clear previous gradients

            # Compute loss
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                outputs = model(img)
                loss = loss_fn(outputs, lbl)

            # Backward pass with scaled loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

        # Validation loop
        model.eval()
        with torch.no_grad(), autocast(device_type="cuda", enabled=(device=="cuda")):
            for data in val_loader:
                img, lbl = data
                img = img.float().to(device=device)
                lbl = lbl.long().to(device=device)
                
                outputs = model(img)             # Forward pass for validation
                loss = loss_fn(outputs, lbl)     # Compute validation loss
                running_val_loss += loss.item()  # Accumulate validation loss

                # Compute Top-1 and Top-5
                _, top1_pred = outputs.max(1)
                top1_correct += (top1_pred == lbl).sum().item()

                _, top5_pred = outputs.topk(5, dim=1)
                top5_correct += (lbl.unsqueeze(1) == top5_pred).sum().item()

                total += lbl.size(0)

        # End timer
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time

        train_losses.append(running_train_loss / len(train_loader))
        val_losses.append(running_val_loss / len(val_loader))

        # Calculate error rates
        top1_acc = 100 * top1_correct / total
        top5_acc = 100 * top5_correct / total

        # Update learning rate 
        scheduler.step(val_losses[-1])

        timestamp = datetime.datetime.now().strftime("%I:%M:%S %p")

        log_message = (
            f"[{timestamp}] Epoch {epoch}/{n_epochs}, "
            f"Train Loss: {train_losses[-1]:.4f}, "
            f"Val Loss: {val_losses[-1]:.4f}, "
            f"Top-1 Accuracy: {top1_acc:.2f}%, "
            f"Top-5 Accuracy: {top5_acc:.2f}%, "
            f"Total Time: {total_training_time:.2f}s\n"
        )
        print(log_message.strip())

        # Save model
        best_val_loss = save_model(model, model_path, best_val_loss, val_losses[-1])
        
        # Log details
        log_details(log_path, log_message)

        # Total number of val images
        total_val_images = len(val_loader.dataset)

        # Plot loss graph
        if plot_path is not None:
            plot_losses(train_losses, val_losses, total_val_images, total_training_time, 
                        top1_acc, top5_acc, plot_path)


# ------------------------------
# Command-line Argument Parsing
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # Command-line arguments
    parser.add_argument("--m", type=str, choices=["alexnet", "vgg16", "resnet18"], required=True, help="Model to Train")
    parser.add_argument('--e', type=int, default=DEFAULT_EPOCHS, help="Number of Epochs to Train")
    parser.add_argument('--b', type=int, default=DEFAULT_BATCH_SIZE, help="Batch Size")
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help="Learning Rate")
    parser.add_argument('--wd', type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight Decay")

    return parser.parse_args()


# ------------------------------
# Main
# ------------------------------

def main():
    # Parse command line arguments
    args = parse_args()

    # Print the training configuration with bold text
    print(f"\033[1mRunning main with the following parameters:\033[0m\n"
          f"Model: {args.m}\nEpochs: {args.e}\nBatch Size: {args.b}\n"
          f"Learning Rate: {args.lr}\nWeight Decay: {args.wd}\n")
    
    # Set device to GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running Model using Device: ', device)

    # Load model
    model = load_model(model_type=args.m, device=device, num_classes=100)

    optimizer, scheduler, loss_fn = setup_components(model, args)

    # Model summary
    summary(model, input_size=(3, 224, 224))

    # Data transformations
    transform = transformation()

    # Data loaders
    train_loader, val_loader = get_dataloaders(transform, args.b)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # File paths
    model_path = os.path.join(MODEL_DIR, f"{args.m}_weights_E{args.e}_B{args.b}.pth")
    plot_path = os.path.join(PLOT_DIR, f"{args.m}_loss_E{args.e}_B{args.b}.png")
    log_path  = os.path.join(LOG_DIR, f"{args.m}_E{args.e}_B{args.b}_{timestamp}.txt")

    # Start training
    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=args.e,
        model_path=model_path,
        plot_path=plot_path,
        log_path=log_path,
    )

# Entry point of the script
if __name__ == '__main__':
    main()