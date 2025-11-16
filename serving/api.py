# import io
# import torch
# import mlflow.pytorch
# from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
# from torchvision import transforms
import uvicorn


app = FastAPI()


mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "koniq_iqa_model"
MODEL_ALIAS = "staging"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pytorch.load_model(model_uri)
    return model.to(device)


transform = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.CenterCrop((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@app.get("/")
def root():
    return {"message": "Hello, FastAPI!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File must be JPEG or PNG")
    
    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = transform(img).unsqueeze(0).to(device)  # [1, 3, 512, 384]
    with torch.no_grad():
        pred = model(x)  # IQA score
    
    pred = pred.squeeze().item()
    score = pred * 100.0

    return {"quality": score}


if __name__ == "__main__":
    model = load_model()
    model.eval()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)