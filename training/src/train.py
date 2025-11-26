import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import numpy as np

from dataset import KonIQDataset
from model import build_model


mlflow.set_tracking_uri("https://mlops-mlflow-server-586303961329.us-central1.run.app")


CSV = "training/data/koniq10k_distributions_sets.csv"
IMG = "training/data/koniq10k_512x384/"


EPOCHS = 100
BATCH = 16
LR = 1e-4
WEIGHT_DECAY = 1e-5
MODEL_NAME = "resnet18"  # efficientnet_b0 / resnet18 / mobilenet_v2


def compute_metrics(preds, targets):
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    sp = spearmanr(preds, targets).correlation
    if np.isnan(sp):
        sp = 0.0

    pr = pearsonr(preds, targets)[0]
    if np.isnan(pr):
        pr = 0.0

    return mae, rmse, sp, pr


def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, mos in dataloader:
            imgs = imgs.to(device, dtype=torch.float32)
            mos = mos.to(device)

            pred = model(imgs).squeeze(-1)

            preds.extend(pred.cpu().numpy())
            targets.extend(mos.cpu().numpy())

    return compute_metrics(preds, targets)


def train():
    mlflow.set_experiment("koniq_iqa")

    run_name = f"{MODEL_NAME}_lr{LR}_bs{BATCH}_ep{EPOCHS}_wc{WEIGHT_DECAY}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY
        })

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}\n")
        mlflow.log_param("device", device)

        model = build_model(MODEL_NAME).to(device)

        train_ds = KonIQDataset(CSV, IMG, split="training")
        val_ds   = KonIQDataset(CSV, IMG, split="validation")
        test_ds  = KonIQDataset(CSV, IMG, split="test")

        train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
        val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
        test_dl  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for ep in range(EPOCHS):
            model.train()
            epoch_loss = 0

            all_preds = []
            all_targets = []

            for imgs, mos in tqdm(train_dl, desc=f"Epoch {ep+1}"):
                imgs = imgs.to(device, dtype=torch.float32)
                mos = mos.to(device)

                optimizer.zero_grad()
                pred = model(imgs)
                pred = pred.squeeze(-1) # (B,1) -> (B,)

                loss = criterion(pred, mos)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                all_preds.extend(pred.detach().cpu().numpy())
                all_targets.extend(mos.cpu().numpy())

            train_loss = epoch_loss / len(train_dl)
            train_mae, train_rmse, train_sp, train_pr = compute_metrics(all_preds, all_targets)

            mlflow.log_metric("train_loss", train_loss, step=ep)
            mlflow.log_metric("train_mae", train_mae, step=ep)
            mlflow.log_metric("train_rmse", train_rmse, step=ep)
            mlflow.log_metric("train_spearman", train_sp, step=ep)
            mlflow.log_metric("train_pearson", train_pr, step=ep)

            val_mae, val_rmse, val_sp, val_pr = evaluate(model, val_dl, device)

            mlflow.log_metric("val_mae", val_mae, step=ep)
            mlflow.log_metric("val_rmse", val_rmse, step=ep)
            mlflow.log_metric("val_spearman", val_sp, step=ep)
            mlflow.log_metric("val_pearson", val_pr, step=ep)

            print(
                f"Epoch {ep+1}: "
                f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f} | "
                f"val_mae={val_mae:.4f}, val_rmse={val_rmse:.4f}"
            )
        
        test_mae, test_rmse, test_sp, test_pr = evaluate(model, test_dl, device)

        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_spearman", test_sp)
        mlflow.log_metric("test_pearson", test_pr)

        print(f"\n[TEST] mae={test_mae:.4f}, rmse={test_rmse:.4f}, sp={test_sp:.4f}")

        mlflow.pytorch.log_model(model, name="model")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_registry_name = "iqa_efficientnet_b0"

        client = MlflowClient()
        try:
            client.create_registered_model(model_registry_name)
        except Exception:
            pass

        mv = client.create_model_version(
            name=model_registry_name,
            source=model_uri,
            run_id=run_id
        )

        mv = client.update_model_version(
            name=model_registry_name,
            version=mv.version,
            description="KONIQ10k IQA model, ready for staging"
        )

        client.set_registered_model_alias(
            name=model_registry_name,
            alias="staging",
            version=mv.version
        )

        print(f"Registered model {model_registry_name} version {mv.version} with alias 'staging'")

        
if __name__ == "__main__":
    train()
