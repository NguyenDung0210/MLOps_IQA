import os
import torch
import mlflow.pytorch

mlflow.set_tracking_uri("https://mlops-mlflow-server-586303961329.us-central1.run.app")

MODEL_NAME = "iqa_efficientnet_b0"
MODEL_ALIAS = "staging"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pytorch.load_model(
        model_uri, 
        map_location=device # Map checkpoint tensors to the device in use to avoid CUDA/CPU mismatch
    )
    return model.to(device)
