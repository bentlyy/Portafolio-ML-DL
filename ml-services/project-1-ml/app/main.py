from fastapi import FastAPI
from app.model import model
import numpy as np

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    x = np.array(data["input"]).reshape(1, -1)
    prediction = model.predict(x)

    return {"prediction": prediction.tolist()}