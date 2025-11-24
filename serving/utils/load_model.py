import os
import torch
import mlflow.pytorch

mlflow.set_tracking_uri("https://mlops-mlflow-server-586303961329.us-central1.run.app")

MODEL_NAME = "koniq_iqa_model"
MODEL_ALIAS = "staging"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pytorch.load_model(model_uri)
    return model.to(device)
