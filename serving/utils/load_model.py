import torch
import mlflow.pytorch


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "koniq_iqa_model"
MODEL_ALIAS = "staging"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pytorch.load_model(model_uri)
    return model.to(device)
