import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet18, mobilenet_v2

def build_model(model_name="efficientnet_b0", hidden_dim=128, dropout=0.2):
    """
    Build a model for IQA regression.
    
    Args:
        model_name (str): "efficientnet_b0", "resnet18", or "mobilenet_v2"
        hidden_dim (int): hidden layer size
        dropout (float): dropout probability
    Returns:
        nn.Module
    """
    
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights="DEFAULT")
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    elif model_name == "resnet18":
        model = resnet18(weights="DEFAULT")
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(weights="DEFAULT")
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
