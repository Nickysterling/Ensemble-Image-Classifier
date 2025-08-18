# train.py

import argparse
import time
import datetime
import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from torchvision import transforms, datasets, models
from torchsummary import summary
import matplotlib.pyplot as plt

# Default values for the command-line arguments
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2

# Initialize the GradScaler
scaler = GradScaler()

# Function to train the model
def train(model, n_epochs, device, optimizer, scheduler, loss_fn, train_loader, test_loader, model_file, plot, log, args):
    print(f"\nTraining Model...")

# Lists to hold losses for each epoch
    average_loss_train = []
    average_loss_val = []

    # Start timer for the whole training process
    training_start_time = time.time()

    # Loop through the number of epochs
    for epoch in range(1, n_epochs + 1):
        current_time = datetime.datetime.now()
        
        # Format to show hour, minute, second
        formatted_time = current_time.strftime("%I:%M:%S %p")

        print(formatted_time, 'Epoch', epoch)
        loss_train = 0.0  # Accumulates training loss
        loss_validation = 0.0  # Accumulates validation loss
        
        # Training loop
        model.train()  # Set model to training mode
        for data in train_loader:
            img, lbl = data
            img = img.float().to(device=device)  # Move images to the specified device (CPU/GPU)
            lbl = lbl.long().to(device=device)  # Convert labels to LongTensor and move to the device

            optimizer.zero_grad()  # Clear previous gradients

            # Mixed precision forward pass
            with autocast(device_type="cuda"):
                outputs = model(img)
                loss = loss_fn(outputs, lbl)

            # Backward pass with scaled loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_train += loss.item()  # Accumulate training loss

        # Validation loop (model in eval mode, no gradient computation)
        model.eval()
        
        # Error rate tracking
        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad(), autocast(device_type="cuda"):  # Disable gradient computation for validation
            for img, lbl in test_loader:
                img = img.to(device=device)  # Move validation images to device
                lbl = lbl.to(device=device)  # Convert labels to LongTensor and move to the device
                outputs = model(img)  # Forward pass for validation

                # Compute the loss and validation distances
                loss = loss_fn(outputs, lbl)  # Compute validation loss
                loss_validation += loss.item()  # Accumulate validation loss

                # Compute Top-1 and Top-5
                _, top1_pred = outputs.max(1)
                top1_correct += (top1_pred == lbl).sum().item()

                _, top5_pred = outputs.topk(5, dim=1)
                top5_correct += (lbl.unsqueeze(1) == top5_pred).sum().item()

                total += lbl.size(0)

        # Calculate error rates
        top1_error_rate = 100 - (100 * top1_correct / total)
        top5_error_rate = 100 - (100 * top5_correct / total)

        # Append losses and error rates to their respective lists
        average_loss_train.append(loss_train / len(train_loader))
        average_loss_val.append(loss_validation / len(test_loader))

        # Format time to show hour, minute, second
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%I:%M:%S %p")

        # Generate the log message
        log_message = (f'{formatted_time} Epoch {epoch}, Batch Size: {args.b}, Learning Rate: {args.lr}, Weight Decay: {args.wd}\n'
                       f'Training loss: {loss_train / len(train_loader):.4f}, '
                       f'Validation loss: {loss_validation / len(test_loader):.4f}\n'
                       f'Top-1 Error Rate: {top1_error_rate:.2f}%, Top-5 Error Rate: {top5_error_rate:.2f}%\n')

        # Print to the console
        print(log_message)

        # Save to a text file at the 5th epoch and the final epoch
        if epoch == 5 or epoch == n_epochs:
            # Write the log message to the file
            with open(log, "a") as log_file:
                log_file.write(log_message + "\n")
            
            print(f"Saved log message to log file: {log}")

        # Save model after 5 epochs
        if epoch == 5:
            torch.save(model.state_dict(), f'{model_file}_epoch_05.pth')
            print(f"Saved model checkpoint after 5 epochs: {model_file}_epoch_05.pth\n")

        # Update the learning rate scheduler based on training loss
        scheduler.step(loss_validation)

        # Save model checkpoint if a file path is provided
        if model_file is not None:
            model_file += ".pth"
            torch.save(model.state_dict(), model_file)
        print('Saving model to:', model_file,)

        # End timer for the whole training process
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        minutes, seconds = divmod(total_training_time, 60)

        # Calculate the total number of test images (validation images)
        total_test_images = len(test_loader.dataset)

        # Calculate the average time per test image (total time divided by number of test images)
        avg_time_per_image = (total_training_time / total_test_images) * 1000  # Convert to milliseconds

        # Plot and save the loss graph
        if plot is not None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()

            # Plot average train and validation losses
            plt.plot(average_loss_train, label='Train Loss')
            plt.plot(average_loss_val, label='Validation Loss')

            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc=1)

            # Add statistics as text at the bottom of the plot (below the epoch label)
            stats_text = (f"Total Runtime: {int(minutes)} min, {int(seconds)} sec, Avg Time/Test Image: {avg_time_per_image:.2f} ms\n"
                          f"Avg Train Loss: {average_loss_train[-1]:.1f}, Avg Val Loss: {average_loss_val[-1]:.1f}\n"
                          f"Top-1 Error Rate: {top1_error_rate:.2f}%, Top-5 Error Rate: {top5_error_rate:.2f}%")

            # Position the text at the bottom of the figure
            plt.figtext(0.5, -0.05, stats_text, wrap=True, horizontalalignment='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.5))

            # Add a title to the plot
            plt.title(f"Training Results\nLoss Plot: {plot}")
            print('Saving plot to:', plot, '\n')
            plt.savefig(plot, bbox_inches="tight")

# Function to initialize the model weights (Xavier Uniform Initialization)
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)  # Apply Xavier initialization for weights
        if m.bias is not None:
            m.bias.data.fill_(0.01)  # Initialize bias with a small value (0.01)

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # Command-line arguments
    parser.add_argument("--model_type", type=str, choices=["alexnet", "vgg16", "resnet18"], required=True, help="Model to Train")
    parser.add_argument('--e', type=int, default=DEFAULT_EPOCHS, help="Number of Epochs to Train")
    parser.add_argument('--b', type=int, default=DEFAULT_BATCH_SIZE, help="Batch Size")
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help="Learning Rate")
    parser.add_argument('--wd', type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight Decay")

    return parser.parse_args()

# Main function to run the training process
def main():
    # Parse command line arguments
    args = parse_args()

    # Print the training configuration with bold text
    print(f"\033[1mRunning main with the following parameters:\033[0m\n"
          f"Model: {args.model_type}\nEpochs: {args.e}\nBatch Size: {args.b}\n"
          f"Learning Rate: {args.lr}\nWeight Decay: {args.wd}\n")
    
    # Set device to GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print('Running Model using Device: ', device, "\n")

    # Load the model using args
    if args.model_type == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier = nn.Sequential(
        nn.Dropout(p=0.7),  # Increased dropout rate
        nn.Linear(256 * 6 * 6, 2048),  # Reduced size from 4096 to 2048
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.7),
        nn.Linear(2048, 1024),  # Reduced size from 4096 to 1024
        nn.ReLU(inplace=True),
        nn.Linear(1024, 100)  # Output layer for CIFAR-100
        )
    elif args.model_type == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.7),  # Increased dropout rate
            nn.Linear(512 * 7 * 7, 2048),  # Reduced size from 4096 to 2048
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(2048, 1024),  # Reduced size from 4096 to 1024
            nn.ReLU(inplace=True),
            nn.Linear(1024, 100)  # Output layer for CIFAR-100
        )
    elif args.model_type == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust for CIFAR-100

    model.to(device)
    model.apply(init_weights)

    # Print the model summary
    summary(model, input_size=(3, 224, 224))

    # Transformations applied to the train and validation datasets
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])

    # Load training and validation datasets
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Create data loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.b, shuffle=False)

    # Set up optimizer, learning rate scheduler, and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    loss_fn = nn.CrossEntropyLoss()

    # Generate dynamic filenames for model and plot based on epochs and batch size
    model_file = f'outputs/model_weights/{args.model_type}_weights_E{args.e}_B{args.b}'
    plot_file = f'outputs/plots/{args.model_type}_loss_E{args.e}_B{args.b}.png'
    log_file = f'outputs/logs/{args.model_type}_loss_E{args.e}_B{args.b}.txt'

    # Start the training process
    train(
        model=model,
        n_epochs=args.e,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        model_file=model_file,
        plot=plot_file,
        log=log_file,
        args=args
    )

# Entry point of the script
if __name__ == '__main__':
    main()