# utils.py

import os

import torch
from torch import nn, optim
from torchvision import transforms, datasets

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def transformation():
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])

    return transform

def file_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, "outputs")
    
    paths = {
        "PROJECT_ROOT": project_root,
        "DATA_DIR": os.path.join(project_root, "data"),
        "OUTPUT_DIR": output_dir,
        "MODEL_DIR": os.path.join(output_dir, "model_weights"),
        "PLOT_DIR": os.path.join(output_dir, "plots"),
        "LOG_DIR": os.path.join(output_dir, "logs")
    }
    
    # Create directories if they don't exist
    for path in ["OUTPUT_DIR", "MODEL_DIR", "PLOT_DIR", "LOG_DIR"]:
        os.makedirs(paths[path], exist_ok=True)
    
    return paths

def get_dataloaders(transform, batch_size, dataset_name="CIFAR100"):
    paths = file_paths()
    DATA_DIR = paths["DATA_DIR"]

    DatasetClass = getattr(datasets, dataset_name)

    # Load datasets
    train_dataset = DatasetClass(root=DATA_DIR, train=True, download=True, transform=transform)
    val_dataset = DatasetClass(root=DATA_DIR, train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def setup_components(model, args):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    loss_fn = nn.CrossEntropyLoss()
    return optimizer, scheduler, loss_fn