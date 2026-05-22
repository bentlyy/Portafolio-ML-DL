from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd
import json
from datetime import datetime


class ModelInfo:
    def __init__(
        self,
        model_id: str,
        name: str,
        model_type: str,
        description: str,
        category: str,
        supported_tasks: list[str],
        hyperparameters: dict[str, Any],
    ):
        self.model_id = model_id
        self.name = name
        self.model_type = model_type
        self.description = description
        self.category = category
        self.supported_tasks = supported_tasks
        self.hyperparameters = hyperparameters

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "model_type": self.model_type,
            "description": self.description,
            "category": self.category,
            "supported_tasks": self.supported_tasks,
            "hyperparameters": self.hyperparameters,
        }


class TrainingResult:
    def __init__(
        self,
        model_id: str,
        metrics: dict[str, float],
        training_time: float,
        model_params: dict[str, Any],
        feature_importance: dict[str, float] | None = None,
        confusion_matrix: list[list[int]] | None = None,
        classification_report: dict | None = None,
        algorithm_details: dict[str, Any] | None = None,
        predictions: list[float] | None = None,
        actual_values: list[float] | None = None,
        residuals: list[float] | None = None,
    ):
        self.model_id = model_id
        self.metrics = metrics
        self.training_time = training_time
        self.model_params = model_params
        self.feature_importance = feature_importance
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self.algorithm_details = algorithm_details
        self.predictions = predictions
        self.actual_values = actual_values
        self.residuals = residuals
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        result = {
            "model_id": self.model_id,
            "metrics": self.metrics,
            "training_time_seconds": round(self.training_time, 3),
            "timestamp": self.timestamp,
            "model_params": self.model_params,
        }
        if self.feature_importance:
            result["feature_importance"] = self.feature_importance
        if self.confusion_matrix:
            result["confusion_matrix"] = self.confusion_matrix
        if self.classification_report:
            result["classification_report"] = self.classification_report
        if self.algorithm_details:
            result["algorithm_details"] = self.algorithm_details
        if self.predictions is not None:
            result["predictions"] = self.predictions
        if self.actual_values is not None:
            result["actual_values"] = self.actual_values
        if self.residuals is not None:
            result["residuals"] = self.residuals
        return result


class BaseModel(ABC):
    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_name = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> TrainingResult:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> dict:
        pass

    @abstractmethod
    def get_hyperparameters(self) -> dict:
        pass

    @abstractmethod
    def set_hyperparameters(self, **kwargs) -> None:
        pass
