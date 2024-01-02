import torch
import base64
import numpy as np
import albumentations as A
from PIL import Image
from io import BytesIO
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import DeepLabV3Plus

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredRequest(BaseModel):
    img_base64: str


class PredResponse(BaseModel):
    prediction: str


async def make_prediction(img_arr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLabV3Plus(num_classes=1)
    model.load_state_dict(torch.load("src/weights/best_model.pth"))
    model.to(device)
    model.eval()

    transformations = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    img_arr /= 255.0

    augmentations = transformations(image=img_arr)
    image = augmentations["image"]
    image = image.to(device)

    output = model(image)
    prediction = output.cpu().numpy()[0, 0]
    prediction = np.expand_dims(prediction, axis=-1)

    pred_bytes = (prediction * 255).astype(np.uint8).tobytes()

    base64_pred = base64.b64encode(pred_bytes).decode("utf-8")  # .decode('utf-8') to convert to str

    return base64_pred


@app.post("/predict", response_model=PredResponse)
async def predict(payload: PredRequest):
    try:
        image_data = base64.b64decode(payload.img_base64)

        with Image.open(BytesIO(image_data)) as pil_img:
            img_arr = np.array(pil_img)

        pred_res = make_prediction(img_arr)

        return {"prediction": pred_res}
    except Exception as e:
        print(f"Error in make_prediction: {str(e)}")
        return HTTPException(status_code=500, detail="Error during model prediction.")


@app.get("/")
async def root():
    return {"message": "API for Capstone!"}
