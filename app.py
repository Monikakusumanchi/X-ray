from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    plt.imshow(img)
    plt.show()   
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    model = tf.keras.models.load_model("/workspace/X-ray/pnuemonia_sequential1.h5")
    predictions = model.predict(img)
    predictions1 = predictions * 100
    threshold = 0.5
    binary_outputs = (predictions1 > threshold).astype(int)

    print("Binary outputs:", binary_outputs.tolist())

    return binary_outputs.tolist()
