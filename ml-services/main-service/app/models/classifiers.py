import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from app.models.base import BaseModel, ModelInfo, TrainingResult

RANDOM_FOREST_INFO = ModelInfo(
    model_id="random_forest",
    name="Random Forest",
    model_type="ensemble",
    description="Método de ensamblaje que usa múltiples árboles de decisión con bagging. Robusto contra sobreajuste, maneja bien relaciones no lineales.",
    category="classification",
    supported_tasks=["binary", "multiclass"],
    hyperparameters={
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Número de árboles"},
        "max_depth": {"type": "int", "default": 10, "min": 2, "max": 50, "description": "Profundidad máxima del árbol"},
        "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20, "description": "Mín. muestras para dividir"},
    },
)

GRADIENT_BOOSTING_INFO = ModelInfo(
    model_id="gradient_boosting",
    name="Gradient Boosting (XGBoost-style)",
    model_type="ensemble",
    description="Ensamblaje secuencial que corrige errores de árboles anteriores. Estado del arte para competiciones de datos tabulares.",
    category="classification",
    supported_tasks=["binary", "multiclass"],
    hyperparameters={
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Número de rondas de boosting"},
        "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "description": "Tasa de aprendizaje"},
        "max_depth": {"type": "int", "default": 3, "min": 1, "max": 20, "description": "Profundidad máxima del árbol"},
    },
)

SVM_INFO = ModelInfo(
    model_id="svm",
    name="Support Vector Machine (SVM)",
    model_type="kernel",
    description="Encuentra el hiperplano óptimo con margen máximo. Efectivo en espacios de alta dimensionalidad usando trucos de kernel.",
    category="classification",
    supported_tasks=["binary", "multiclass"],
    hyperparameters={
        "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "description": "Parámetro de regularización"},
        "kernel": {"type": "choice", "default": "rbf", "options": ["linear", "rbf", "poly"], "description": "Tipo de kernel"},
        "gamma": {"type": "choice", "default": "scale", "options": ["scale", "auto"], "description": "Coeficiente del kernel"},
    },
)

LOGISTIC_REGRESSION_INFO = ModelInfo(
    model_id="logistic_regression",
    name="Regresión Logística",
    model_type="linear",
    description="Clasificador lineal que usa función logística. Interpretable, rápido, buen modelo de referencia.",
    category="classification",
    supported_tasks=["binary", "multiclass"],
    hyperparameters={
        "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "description": "Regularización inversa"},
        "penalty": {"type": "choice", "default": "l2", "options": ["l1", "l2", "elasticnet"], "description": "Tipo de penalización"},
        "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 10000, "description": "Máx. iteraciones"},
    },
)


class RandomForestClassifierModel(BaseModel):
    def __init__(self):
        super().__init__(RANDOM_FOREST_INFO)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        self.model.fit(X, y)
        training_time = time.time() - start

        y_pred = self.model.predict(X)
        feature_imp = dict(zip(X.columns, self.model.feature_importances_.tolist()))

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=self._calc_metrics(y, y_pred),
            training_time=training_time,
            model_params=self.model.get_params(),
            feature_importance=feature_imp,
            confusion_matrix=confusion_matrix(y, y_pred).tolist(),
            classification_report=classification_report(y, y_pred, output_dict=True),
        )

    def predict(self, X: pd.DataFrame) -> dict:
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        classes = self.model.classes_.tolist()
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "classes": classes,
        }

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)

    @staticmethod
    def _calc_metrics(y_true, y_pred) -> dict:
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        }


class GradientBoostingClassifierModel(BaseModel):
    def __init__(self):
        super().__init__(GRADIENT_BOOSTING_INFO)
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        self.model.fit(X, y)
        training_time = time.time() - start

        y_pred = self.model.predict(X)
        feature_imp = dict(zip(X.columns, self.model.feature_importances_.tolist()))

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=self._calc_metrics(y, y_pred),
            training_time=training_time,
            model_params=self.model.get_params(),
            feature_importance=feature_imp,
            confusion_matrix=confusion_matrix(y, y_pred).tolist(),
            classification_report=classification_report(y, y_pred, output_dict=True),
        )

    def predict(self, X: pd.DataFrame) -> dict:
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        classes = self.model.classes_.tolist()
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "classes": classes,
        }

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)

    @staticmethod
    def _calc_metrics(y_true, y_pred) -> dict:
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        }


class SVMClassifierModel(BaseModel):
    def __init__(self):
        super().__init__(SVM_INFO)
        self.model = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        self.model.fit(X, y)
        training_time = time.time() - start

        y_pred = self.model.predict(X)

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=self._calc_metrics(y, y_pred),
            training_time=training_time,
            model_params=self.model.get_params(),
            confusion_matrix=confusion_matrix(y, y_pred).tolist(),
            classification_report=classification_report(y, y_pred, output_dict=True),
        )

    def predict(self, X: pd.DataFrame) -> dict:
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        classes = self.model.classes_.tolist()
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "classes": classes,
        }

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)

    @staticmethod
    def _calc_metrics(y_true, y_pred) -> dict:
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        }


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(LOGISTIC_REGRESSION_INFO)
        self.model = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=42, solver="lbfgs")

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> TrainingResult:
        start = time.time()
        params = dict(kwargs)
        if "penalty" in params:
            penalty = params["penalty"]
            if penalty == "l1":
                params["solver"] = "liblinear"
            elif penalty == "elasticnet":
                params["solver"] = "saga"
                params["l1_ratio"] = 0.5
        self.model.set_params(**params)
        self.model.fit(X, y)
        training_time = time.time() - start

        y_pred = self.model.predict(X)
        if self.model.coef_.ndim > 1:
            coef = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            coef = np.abs(self.model.coef_)
        feature_imp = dict(zip(X.columns, [round(float(c), 4) for c in coef]))

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=self._calc_metrics(y, y_pred),
            training_time=training_time,
            model_params=self.model.get_params(),
            feature_importance=feature_imp,
            confusion_matrix=confusion_matrix(y, y_pred).tolist(),
            classification_report=classification_report(y, y_pred, output_dict=True),
        )

    def predict(self, X: pd.DataFrame) -> dict:
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        classes = self.model.classes_.tolist()
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "classes": classes,
        }

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)

    @staticmethod
    def _calc_metrics(y_true, y_pred) -> dict:
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        }
