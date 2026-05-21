import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.models.base import BaseModel, ModelInfo, TrainingResult

LINEAR_REGRESSION_INFO = ModelInfo(
    model_id="linear_regression",
    name="Regresión Lineal",
    model_type="linear",
    description="Modelo lineal básico que ajusta un hiperplano para minimizar errores cuadrados. Línea base interpretable para regresión.",
    category="regression",
    supported_tasks=["continuous"],
    hyperparameters={
        "fit_intercept": {"type": "bool", "default": True, "description": "Calcular intercepto"},
    },
)

RIDGE_REGRESSION_INFO = ModelInfo(
    model_id="ridge",
    name="Regresión Ridge",
    model_type="linear",
    description="Regresión lineal con regularización L2. Previene sobreajuste mediante contracción de coeficientes.",
    category="regression",
    supported_tasks=["continuous"],
    hyperparameters={
        "alpha": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "description": "Fuerza de regularización"},
    },
)

RANDOM_FOREST_REGRESSOR_INFO = ModelInfo(
    model_id="random_forest_regressor",
    name="Random Forest Regresor",
    model_type="ensemble",
    description="Ensamblaje de árboles de decisión que promedia predicciones. Maneja relaciones no lineales e interacciones entre características.",
    category="regression",
    supported_tasks=["continuous"],
    hyperparameters={
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Número de árboles"},
        "max_depth": {"type": "int", "default": 10, "min": 2, "max": 50, "description": "Profundidad máxima del árbol"},
        "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20, "description": "Mín. muestras para dividir"},
    },
)

GRADIENT_BOOSTING_REGRESSOR_INFO = ModelInfo(
    model_id="gradient_boosting_regressor",
    name="Gradient Boosting Regresor",
    model_type="ensemble",
    description="Ensamblaje secuencial que corrige errores de predicción. Potente para tareas de regresión complejas.",
    category="regression",
    supported_tasks=["continuous"],
    hyperparameters={
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Número de rondas de boosting"},
        "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "description": "Tasa de aprendizaje"},
        "max_depth": {"type": "int", "default": 3, "min": 1, "max": 20, "description": "Profundidad máxima del árbol"},
    },
)


def _calc_metrics(y_true, y_pred) -> dict:
    return {
        "mse": round(mean_squared_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred, squared=False), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(LINEAR_REGRESSION_INFO)
        self.model = LinearRegression()

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        self.model.fit(X, y)
        training_time = time.time() - start

        y_pred = self.model.predict(X)
        coef = self.model.coef_ if hasattr(self.model, "coef_") else [0]
        if hasattr(coef, "flatten"):
            coef = coef.flatten()
        feature_imp = dict(zip(X.columns, [round(float(c), 4) for c in coef]))

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(y, y_pred),
            training_time=training_time,
            model_params=self.model.get_params(),
            feature_importance=feature_imp,
        )

    def predict(self, X: pd.DataFrame) -> dict:
        preds = self.model.predict(X)
        return {"predictions": preds.tolist()}

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)


class RidgeRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(RIDGE_REGRESSION_INFO)
        self.model = Ridge(alpha=1.0)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        self.model.fit(X, y)
        training_time = time.time() - start

        y_pred = self.model.predict(X)
        coef = self.model.coef_ if hasattr(self.model, "coef_") else [0]
        if hasattr(coef, "flatten"):
            coef = coef.flatten()
        feature_imp = dict(zip(X.columns, [round(float(c), 4) for c in coef]))

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(y, y_pred),
            training_time=training_time,
            model_params=self.model.get_params(),
            feature_importance=feature_imp,
        )

    def predict(self, X: pd.DataFrame) -> dict:
        preds = self.model.predict(X)
        return {"predictions": preds.tolist()}

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)


class RandomForestRegressorModel(BaseModel):
    def __init__(self):
        super().__init__(RANDOM_FOREST_REGRESSOR_INFO)
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        self.model.fit(X, y)
        training_time = time.time() - start

        y_pred = self.model.predict(X)
        feature_imp = dict(zip(X.columns, self.model.feature_importances_.tolist()))

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(y, y_pred),
            training_time=training_time,
            model_params=self.model.get_params(),
            feature_importance=feature_imp,
        )

    def predict(self, X: pd.DataFrame) -> dict:
        preds = self.model.predict(X)
        return {"predictions": preds.tolist()}

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)


class GradientBoostingRegressorModel(BaseModel):
    def __init__(self):
        super().__init__(GRADIENT_BOOSTING_REGRESSOR_INFO)
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        self.model.fit(X, y)
        training_time = time.time() - start

        y_pred = self.model.predict(X)
        feature_imp = dict(zip(X.columns, self.model.feature_importances_.tolist()))

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(y, y_pred),
            training_time=training_time,
            model_params=self.model.get_params(),
            feature_importance=feature_imp,
        )

    def predict(self, X: pd.DataFrame) -> dict:
        preds = self.model.predict(X)
        return {"predictions": preds.tolist()}

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)
