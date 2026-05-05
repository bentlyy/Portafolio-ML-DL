from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel as PydanticBaseModel, ConfigDict
from typing import Any, List
import pandas as pd
import io
import logging
import os
from app.models.registry import registry
from app.models.base import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Portfolio Service",
    description="Unified ML/DL model service with multiple algorithms, dataset upload, and training capabilities.",
    version="2.0.0",
)


class PredictRequest(PydanticBaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data: List[List[float]]
    model_id: str


class TrainRequest(PydanticBaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    target_column: str | None = None
    hyperparameters: dict[str, Any] = {}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_available": len(registry.models),
        "categories": list(registry.get_model_categories().keys()),
    }


@app.get("/models")
def list_models():
    return {"models": registry.list_models(), "categories": registry.get_model_categories()}


@app.get("/models/{model_id}")
def get_model_info(model_id: str):
    try:
        model = registry.get_model(model_id)
        return model.model_info.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/models/{model_id}/hyperparameters")
def get_hyperparameters(model_id: str):
    try:
        model = registry.get_model(model_id)
        return {"model_id": model_id, "hyperparameters": model.get_hyperparameters()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head(5).to_dict(orient="records"),
            "describe": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


@app.post("/models/{model_id}/train")
async def train_model(
    model_id: str,
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    hyperparameters: str | None = Form(None),
):
    try:
        model = registry.get_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

    df = df.dropna()

    if model.model_info.category == "clustering":
        X = df
        y = None
    else:
        if not target_column or target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found. Available: {df.columns.tolist()}",
            )

        y = df[target_column]
        X = df.drop(columns=[target_column])

    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    params = {}
    if hyperparameters:
        import json
        try:
            params = json.loads(hyperparameters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid hyperparameters JSON")

    try:
        result = model.train(X, y, **params)
        model.is_trained = True
        return result.to_dict()
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        model = registry.get_model(request.model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Model has not been trained yet. Train it first with /models/{model_id}/train",
        )

    try:
        X = pd.DataFrame(request.data)
        result = model.predict(X)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/models/{model_id}/predict-from-file")
async def predict_from_file(
    model_id: str,
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
):
    try:
        model = registry.get_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Model has not been trained yet. Train it first.",
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])

    df = pd.get_dummies(df, drop_first=True)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    try:
        result = model.predict(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
