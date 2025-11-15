import io
import torch
import mlflow.pytorch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
import uvicorn


app = FastAPI()


MODEL_NAME = "koniq_iqa_model"
MODEL_STAGE = "Staging"

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.pytorch.load_model(model_uri)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    x = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        score = model(x).item()  # IQA score

    return {"quality": score}


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)