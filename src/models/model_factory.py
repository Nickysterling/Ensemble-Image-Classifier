# model_factory.py

import torch
import torch.nn as nn
import torchvision.models as models

def init_layers(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)

# ---------- MODEL BUILDERS ----------

def build_alexnet(num_classes=100):
    model = models.alexnet(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(256 * 6 * 6, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.6),
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, num_classes)
    )
    model.classifier.apply(init_layers)
    return model

def build_vgg16(num_classes=100):
    model = models.vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.7),
        nn.Linear(512 * 7 * 7, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.7),
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, num_classes)
    )
    model.classifier.apply(init_layers)
    return model

def build_resnet18(num_classes=100):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.apply(init_layers)
    return model

# ---------- FACTORY REGISTRY ----------
MODEL_FACTORY = {
    "alexnet": build_alexnet,
    "vgg16": build_vgg16,
    "resnet18": build_resnet18,
}

# ---------- FACTORY LOADER ----------
def load_model(model_type, weights_path=None, device="cpu", num_classes=100):
    if model_type not in MODEL_FACTORY:
        raise ValueError(f"Invalid model type '{model_type}'. Available: {list(MODEL_FACTORY.keys())}")

    model = MODEL_FACTORY[model_type](num_classes=num_classes)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))

    model.to(device)

    return model
