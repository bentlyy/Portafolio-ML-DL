from typing import Any
from app.models.base import BaseModel
from app.models.classifiers import (
    RandomForestClassifierModel,
    GradientBoostingClassifierModel,
    SVMClassifierModel,
    LogisticRegressionModel,
)
from app.models.regressors import (
    LinearRegressionModel,
    RidgeRegressionModel,
    RandomForestRegressorModel,
    GradientBoostingRegressorModel,
)
from app.models.clustering import (
    KMeansModel,
    DBSCANModel,
    HierarchicalClusteringModel,
)
from app.models.neural_networks import NeuralNetworkModel


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, BaseModel] = {}
        self._register_models()

    def _register_models(self):
        self.models["random_forest"] = RandomForestClassifierModel()
        self.models["gradient_boosting"] = GradientBoostingClassifierModel()
        self.models["svm"] = SVMClassifierModel()
        self.models["logistic_regression"] = LogisticRegressionModel()
        self.models["linear_regression"] = LinearRegressionModel()
        self.models["ridge"] = RidgeRegressionModel()
        self.models["random_forest_regressor"] = RandomForestRegressorModel()
        self.models["gradient_boosting_regressor"] = GradientBoostingRegressorModel()
        self.models["kmeans"] = KMeansModel()
        self.models["dbscan"] = DBSCANModel()
        self.models["hierarchical"] = HierarchicalClusteringModel()
        self.models["neural_network"] = NeuralNetworkModel()

    def get_model(self, model_id: str) -> BaseModel:
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found. Available: {list(self.models.keys())}")
        return self.models[model_id]

    def list_models(self) -> list[dict[str, Any]]:
        return [m.model_info.to_dict() for m in self.models.values()]

    def get_model_categories(self) -> dict[str, list[str]]:
        categories = {}
        for model in self.models.values():
            cat = model.model_info.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(model.model_info.model_id)
        return categories


registry = ModelRegistry()
