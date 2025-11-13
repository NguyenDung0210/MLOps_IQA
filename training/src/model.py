import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet18, mobilenet_v2

def build_model(model_name="efficientnet_b0"):
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights="DEFAULT")
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, 1)

    elif model_name == "resnet18":
        model = resnet18(weights="DEFAULT")
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, 1)

    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(weights="DEFAULT")
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, 1)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
