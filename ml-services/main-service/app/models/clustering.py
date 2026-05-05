import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from app.models.base import BaseModel, ModelInfo, TrainingResult

KMEANS_INFO = ModelInfo(
    model_id="kmeans",
    name="K-Means",
    model_type="centroid",
    description="Partitions data into K clusters by minimizing within-cluster variance. Fast and scalable for large datasets.",
    category="clustering",
    supported_tasks=["unsupervised"],
    hyperparameters={
        "n_clusters": {"type": "int", "default": 3, "min": 2, "max": 20, "description": "Number of clusters"},
        "max_iter": {"type": "int", "default": 300, "min": 50, "max": 1000, "description": "Max iterations"},
        "n_init": {"type": "int", "default": 10, "min": 1, "max": 50, "description": "Number of initializations"},
    },
)

DBSCAN_INFO = ModelInfo(
    model_id="dbscan",
    name="DBSCAN",
    model_type="density",
    description="Density-based clustering. Finds arbitrary-shaped clusters and identifies outliers. No need to specify K.",
    category="clustering",
    supported_tasks=["unsupervised"],
    hyperparameters={
        "eps": {"type": "float", "default": 0.5, "min": 0.01, "max": 10.0, "description": "Max distance between samples"},
        "min_samples": {"type": "int", "default": 5, "min": 1, "max": 50, "description": "Min samples for core point"},
    },
)

HIERARCHICAL_INFO = ModelInfo(
    model_id="hierarchical",
    name="Agglomerative Clustering",
    model_type="hierarchical",
    description="Bottom-up hierarchical clustering. Builds a tree of clusters. No assumptions about cluster shape.",
    category="clustering",
    supported_tasks=["unsupervised"],
    hyperparameters={
        "n_clusters": {"type": "int", "default": 3, "min": 2, "max": 20, "description": "Number of clusters"},
        "linkage": {"type": "choice", "default": "ward", "options": ["ward", "complete", "average", "single"], "description": "Linkage criterion"},
    },
)


def _calc_metrics(X, labels) -> dict:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    metrics = {"n_clusters": n_clusters, "n_noise": int(list(labels).count(-1))}

    if n_clusters > 1:
        metrics["silhouette_score"] = round(float(silhouette_score(X, labels)), 4)
        metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(X, labels)), 4)

    return metrics


class KMeansModel(BaseModel):
    def __init__(self):
        super().__init__(KMEANS_INFO)
        self.model = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        self.model.fit(X)
        training_time = time.time() - start

        labels = self.model.labels_

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(X.values, labels),
            training_time=training_time,
            model_params=self.model.get_params(),
        )

    def predict(self, X: pd.DataFrame) -> dict:
        labels = self.model.predict(X)
        return {
            "clusters": labels.tolist(),
            "cluster_centers": self.model.cluster_centers_.tolist(),
        }

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)


class DBSCANModel(BaseModel):
    def __init__(self):
        super().__init__(DBSCAN_INFO)
        self.model = DBSCAN(eps=0.5, min_samples=5)

    def train(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        labels = self.model.fit_predict(X)
        training_time = time.time() - start

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(X.values, labels),
            training_time=training_time,
            model_params=self.model.get_params(),
        )

    def predict(self, X: pd.DataFrame) -> dict:
        labels = self.model.fit_predict(X)
        unique_labels = list(set(labels))
        return {
            "clusters": labels.tolist(),
            "n_clusters": len([l for l in unique_labels if l != -1]),
            "n_noise": int(list(labels).count(-1)),
        }

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)


class HierarchicalClusteringModel(BaseModel):
    def __init__(self):
        super().__init__(HIERARCHICAL_INFO)
        self.model = AgglomerativeClustering(n_clusters=3, linkage="ward")

    def train(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        labels = self.model.fit_predict(X)
        training_time = time.time() - start

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(X.values, labels),
            training_time=training_time,
            model_params=self.model.get_params(),
        )

    def predict(self, X: pd.DataFrame) -> dict:
        labels = self.model.fit_predict(X)
        return {"clusters": labels.tolist()}

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)
