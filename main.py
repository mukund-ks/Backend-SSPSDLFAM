import torch
import base64
import numpy as np
import albumentations as A
from PIL import Image
from io import BytesIO
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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
    mask: str


def bytesToBase64(pred_bytes: bytes):
    output_buffer = BytesIO()

    pred_img = Image.frombytes("L", (256, 256), pred_bytes)
    pred_img.save(output_buffer, format="PNG")

    pred_str = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    return pred_str


async def run_model(img_arr: np.ndarray[np.float32]):
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

    img_thres = img_arr.astype(np.float32) / 255.0

    augmentations = transformations(image=img_thres)
    image = augmentations["image"]
    image = image.to(device)
    image = image.unsqueeze(0)  # Add batch dimension

    output = model(image)
    prediction = output.cpu().detach().numpy()[0, 0]
    prediction = np.expand_dims(prediction, axis=-1)
    prediction = (prediction > 0.5).astype(np.uint8)

    pred_bytes = (prediction * 255).astype(np.uint8).tobytes()

    base64_pred = bytesToBase64(pred_bytes)

    return base64_pred


@app.post("/segment", response_model=PredResponse)
async def segment(payload: PredRequest):
    try:
        image_data = base64.b64decode(payload.img_base64)

        with Image.open(BytesIO(image_data)) as pil_img:
            img_arr = np.array(pil_img.convert("RGB"), dtype=np.float32)

        pred_res = await run_model(img_arr)

        return JSONResponse(status_code=200, content={"mask": pred_res})
    except Exception as e:
        print(f"Error in run_model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error during model run",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"message": f"{exc.detail}"})


@app.get("/")
async def root():
    return {"message": "API for Capstone!"}
