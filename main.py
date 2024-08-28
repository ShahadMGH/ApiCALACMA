from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Union
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import base64
import requests
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the model
model = load_model('new_model_trained_35class3.keras', compile=False)

# Load the calorie information
with open('calories.json') as f:
    calories_info = json.load(f)

# Define the class names
class_names = list(calories_info.keys())

class ImageData(BaseModel):
    base64: Union[str, None] = None
    url: Union[str, None] = None

def preprocess_image(img):
    img_width, img_height = 299, 299
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def decode_base64_image(base64_str):
    decoded = base64.b64decode(base64_str)
    img = Image.open(BytesIO(decoded))
    return img

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

@app.post("/predict")
async def predict(data: ImageData):
    if data.base64:
        img = decode_base64_image(data.base64)
    elif data.url:
        img = load_image_from_url(data.url)
    else:
        raise HTTPException(status_code=400, detail="No image data provided.")

    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]

    return {
        "predicted_label": predicted_label,
        "calories_info": calories_info[predicted_label]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=48862)
