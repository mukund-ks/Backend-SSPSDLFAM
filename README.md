# Semantic Segmentation API

## Overview

This API leverages DeepLabV3+ with a ResNet50 backbone to perform binary semantic segmentation on plant-specific input images. The segmentation model implementation can be found [here](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch). Developed using FastAPI, the API accepts base64-encoded plant images, processes them through the model, and returns the resulting binary semantic segmentation mask encoded as base64.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- PIL
- FastAPI
- Uvicorn
- PyTorch

### Installation

- Clone the repository
```bash
git clone https://github.com/mukund-ks/Backend-SSPSDLFAM
```

- Change directory
```bash
cd Backend-SSPSDLFAM
```

- Create a 'weights' directory under 'src' and place your model weights there
```bash
mkdir -p src/weights
cp /path/to/your_weights.pth src/weights/
```

- Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Runnning the API
To start the FastAPI server, run the following command:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
The API will be accessible at `http://localhost:8000`. Open your web browser or use tools like cURL or Postman to interact with API endpoints.

### Segmentation Endpoint
- Method: POST
- Path: `/segment`
- Parameters: 
```json
{
    "img_base64": "base54-encoded image string"
}
```

#### Response
Upon successfull processing, the API responds with a JSON object containing the base64-encoded binary mask.
```json
{
    "mask":"base64-encoded mask string"
}
```

### Testing with Python

To test the API using Python, you can use the provided example script. The script loads an image, encodes it into base64, sends a POST request to the API's segmentation endpoint, and decodes the resulting base64-encoded segmentation mask.

```python
from PIL import Image
import requests
import base64

img_size = (256,256) # H,W of input image

with open("test_img.png", "rb") as image_file:
    encoded_img = base64.b64encode(image_file.read()).decode("utf-8")

response = requests.post(url="http://localhost:8000/segment", json={"img_base64": encoded_img})

json_res = response.json()

base64_mask = json_res["mask"]
mask_bytes = base64.b64decode(base64_mask)

mask_img = Image.frombytes("L", img_size, mask_bytes)

mask_img.save("mask.png", format="PNG")
```

## Authentication
No authentication is currently required for this API.

## Contributing
Contributions are welcomed! Feel free to submit issues, feature requests, or pull requests.

> [!NOTE]  
> This project is part of the Capstone Project for college.
