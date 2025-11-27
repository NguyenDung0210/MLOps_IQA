# Image Quality Assessment (IQA) MLOps Project

This project implements an end-to-end **Image Quality Assessment (IQA)** pipeline using modern **MLOps practices**. It includes:

- A **training pipeline** built with PyTorch and MLflow
- Automated **experiment tracking** and **model registry**
- A **FastAPI inference service** with both REST API and UI
- Deployment of both MLflow and FastAPI to **Google Cloud Run**
- Full **CI/CD** integration using GitHub Actions

The IQA models are trained on the **KONIQ-10k dataset** and support the following backbones:
- EfficientNet-B0
- ResNet-18
- MobileNetV2

---

## 🚀 Architecture Overview
```mermaid
graph TD;
    A[Training Script (PyTorch)] -->|Log Metrics + Model| B(MLflow Tracking Server);
    B --> C(Model Registry);
    C -->|Alias: staging/production| D(FastAPI Inference Service);
    D --> E(Cloud Run - FastAPI);
    B --> F(Cloud Run - MLflow);
    subgraph GCP
        E
        F
    end
```

---

## 📂 Project Structure
```
├── infra/
│   ├── Dockerfile.fastapi
│   ├── Dockerfile.mlflow
│   └── server.sh
├── serving/
│   ├── api.py
│   ├── config.py
│   ├── routers/
│   ├── utils/
│   ├── static/
│   └── templates/
├── training/
│   ├── src/
│   │   ├── train.py
│   │   ├── model.py
│   │   └── dataset.py
├── tests/
├── requirements.txt
├── README.md
└── .github/workflows/
    ├── test.yaml
    └── deploy.yaml
```

---

## 🧠 Training Pipeline
Training is performed using PyTorch with MLflow for tracking.

### Run training locally
```bash
python training/src/train.py
```

Training automatically logs:
- Hyperparameters (lr, batch size, epochs…)
- Metrics: MAE, RMSE, Spearman, Pearson
- Test set metrics
- The trained model as an MLflow artifact

It then:
1. Registers the model in MLflow Model Registry
2. Creates or updates alias **staging**

---

## 🖥️ Inference Service (FastAPI)
The FastAPI service loads the latest model using an MLflow model alias, e.g.:
```
models:/iqa_efficientnet_b0@staging
```

### Lazy loading
The model is only loaded once on first request.

### API Endpoints
| Endpoint | Method | Description |
|---------|--------|-------------|
| `/health` | GET | Health check |
| `/` | GET | Home page |
| `/predict-ui` | GET | Web UI for image upload |
| `/predict` | POST | Predict IQA score for an uploaded image |

### Example Prediction
```bash
curl -X POST -F "file=@image.jpg" https://your-fastapi-url/predict
```

Response:
```json
{ "quality": 78.54 }
```

---

## 🧪 CI Pipeline (GitHub Actions)
The CI pipeline (`test.yaml`) runs automatically for pull requests to `main` and executes:
- Python 3.11 setup
- Dependency installation
- `pytest` for all tests in `tests/`

---

## 🚀 CD Pipeline – Deploy to Google Cloud Run
Deployment is performed via manual trigger using `deploy.yaml`. It consists of two jobs:
- Deploy MLflow
- Deploy FastAPI

Deployment requires correct setup of GCP services.

---

# ☁️ GCP Setup Guide (Required Before Deployment)
This section describes what any user must configure before they can successfully deploy this project.

## 1. MLflow Cloud Infrastructure
MLflow uses:
- **Cloud SQL (PostgreSQL 17, private IP)** → backend store
- **Cloud Storage bucket** → artifact store
- **Artifact Registry** → Docker images
- **VPC Network + Serverless VPC Connector** → allow Cloud Run to access private Cloud SQL
- **Secret Manager** → store DB URL, bucket URL, allowed hosts

### Required IAM roles for MLflow Service Account
```
Artifact Registry Administrator
Cloud Run Admin
Cloud SQL Client
Secret Manager Admin
Serverless VPC Access User
Service Account User
Storage Admin
```

### Secrets to store in Secret Manager
Example names:
- `DB_url`
- `bucket_url`
- `mlflow_allowed_hosts`

These secrets are passed into Cloud Run during deployment using:
```
--update-secrets=MLFLOW_ALLOWED_HOSTS=mlflow_allowed_hosts:latest
--update-secrets=POSTGRESQL_URL=DB_url:latest
--update-secrets=STORAGE_URL=bucket_url:latest
```

### Artifact Registry
Create a repository for MLflow Docker images.

---

## 2. FastAPI Cloud Infrastructure
FastAPI uses:
- Artifact Registry → Docker image
- Cloud Run → inference service

### Required IAM roles for FastAPI Service Account
```
Artifact Registry Administrator
Cloud Run Admin
Service Account User
```

### Environment variables
FastAPI only needs:
- model alias + tracking URI (already in code)

---

## 3. GitHub Actions Secrets
Add the following to GitHub repository:

### Service account JSON keys
- `ML_SA_KEY` → MLflow deployer
- `FA_SA_KEY` → FastAPI deployer

### Deployment passcode
- `DEPLOY_PASS`

### MLflow secrets (optional depending on workflow)
- Database URL
- Bucket URL
- Allowed hosts

These must match the names used in `deploy.yaml`.

---

## 4. Modify `deploy.yaml`
Update the following ENV variables inside the workflow:
- `REGION`
- `PROJECT_ID`
- Docker repository names
- Cloud Run service names
- VPC connector name
- Secret names

If you changed any secret names, update:
```
--update-secrets=MLFLOW_ALLOWED_HOSTS=...
```

---

## 5. Trigger Deployment
1. Push the repository to your GitHub account
2. Go to **GitHub → Actions → CD**
3. Click **Run workflow**
4. Enter your passcode
5. The workflow will:
   - Build & push MLflow Docker image
   - Deploy MLflow Cloud Run service
   - Build & push FastAPI Docker image
   - Deploy FastAPI Cloud Run service

---

## 📦 Running Locally
### FastAPI
```bash
uvicorn serving.api:app --reload
```

### MLflow server (optional)
You may run MLflow locally with:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

---

## 📚 Dataset: KONIQ-10k
Download:
- https://www.kaggle.com/datasets/generalhawking/koniq-10k-dataset
- https://database.mmsp-kn.de/koniq-10k-database.html

Image size used: **512×384**

Dataset splits:
- training
- validation
- test

---

## 📬 Contact
For questions or collaboration, feel free to reach out.

