import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import numpy as np

from dataset import KonIQDataset
from model import build_model

CSV = "training/data/koniq10k_distributions_sets.csv"
IMG = "training/data/koniq10k_512x384/"

EPOCHS = 3
BATCH = 16
LR = 1e-4
MODEL_NAME = "efficientnet_b0"  # CHANGE HERE

def compute_metrics(preds, targets):
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    sp, _ = spearmanr(preds, targets)
    pr, _ = pearsonr(preds, targets)

    return mae, rmse, sp, pr

def train():
    mlflow.set_experiment("koniq_iqa")

    run_name = f"{MODEL_NAME}_lr{LR}_bs{BATCH}_ep{EPOCHS}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH)
        mlflow.log_param("lr", LR)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}\n")
        mlflow.log_param("device", device)
        model = build_model(MODEL_NAME).to(device)

        train_ds = KonIQDataset(CSV, IMG, split="training")
        train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for ep in range(EPOCHS):
            model.train()
            epoch_loss = 0

            all_preds = []
            all_targets = []

            for imgs, mos in tqdm(train_dl, desc=f"Epoch {ep+1}"):
                imgs, mos = imgs.to(device), mos.to(device)

                optimizer.zero_grad()
                pred = model(imgs)

                loss = criterion(pred, mos)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                all_preds.extend(pred.detach().cpu().numpy())
                all_targets.extend(mos.cpu().numpy())

            avg_loss = epoch_loss / len(train_dl)

            mae, rmse, sp, pr = compute_metrics(all_preds, all_targets)

            mlflow.log_metric("loss", avg_loss, step=ep)
            mlflow.log_metric("mae", mae, step=ep)
            mlflow.log_metric("rmse", rmse, step=ep)
            mlflow.log_metric("spearman", sp, step=ep)
            mlflow.log_metric("pearson", pr, step=ep)

            print(f"Epoch {ep+1}: loss={avg_loss:.4f}, mae={mae:.4f}")

        mlflow.pytorch.log_model(model, name="model")

if __name__ == "__main__":
    train()
