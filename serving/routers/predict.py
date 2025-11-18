import io
import torch
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException
from torchvision import transforms
from serving.utils.load_model import load_model, device


predict_router = APIRouter()


model = load_model()
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


@predict_router.post("/predict")
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
