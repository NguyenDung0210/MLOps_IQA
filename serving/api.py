import io
import torch
import mlflow.pytorch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
import uvicorn


app = FastAPI()


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "koniq_iqa_model"
MODEL_ALIAS = "staging"

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.pytorch.load_model(model_uri)
model.eval()


transform = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.CenterCrop((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    x = transform(img).unsqueeze(0)  # [1, 3, 512, 384]
    with torch.no_grad():
        pred = model(x)  # IQA score
    
    pred = pred.squeeze().item()
    score = pred * 100.0

    return {"quality": score}


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)