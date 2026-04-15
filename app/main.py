from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from app.model import predict_image

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Image Recognition API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = predict_image(image)

    return {"top5": results}