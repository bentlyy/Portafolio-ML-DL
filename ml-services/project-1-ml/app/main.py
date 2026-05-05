from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
import logging

from app.model import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Portfolio Service", version="1.0.0")


class PredictionInput(BaseModel):
    input: list[float]

    @field_validator("input")
    @classmethod
    def validate_input(cls, v):
        if not v:
            raise ValueError("Input array cannot be empty")
        return v


class PredictionOutput(BaseModel):
    prediction: list[float]


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    try:
        x = np.array(data.input).reshape(1, -1)
        prediction = model.predict(x)
        return PredictionOutput(prediction=prediction.tolist())
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")
