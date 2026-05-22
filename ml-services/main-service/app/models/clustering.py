import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage as scipy_linkage
from app.models.base import BaseModel, ModelInfo, TrainingResult

PCA_SAMPLE_LIMIT = 2000


def _compute_pca_projection(X_values: np.ndarray) -> np.ndarray:
    n_samples, n_features = X_values.shape
    n_components = min(2, n_features, n_samples)
    if n_components < 2:
        if n_features == 1:
            zeros = np.zeros((n_samples, 1))
            return np.column_stack([X_values, zeros])
        return np.tile(np.arange(n_samples).reshape(-1, 1), (1, 2))
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_values).astype(float)


def _sample_projection(points: np.ndarray, labels: np.ndarray, max_points: int = PCA_SAMPLE_LIMIT):
    if len(points) <= max_points:
        return {"points": points.tolist(), "labels": [int(l) for l in labels]}
    rng = np.random.RandomState(42)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return {"points": points[idx].tolist(), "labels": [int(labels[i]) for i in idx]}


def _build_cluster_sizes(labels: list) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for l in labels:
        key = str(l)
        sizes[key] = sizes.get(key, 0) + 1
    return sizes


def _clustering_algorithm_details(
    model_id: str,
    X_values: np.ndarray,
    labels: np.ndarray,
    extra: dict | None = None,
    projection_override: dict | None = None,
) -> dict:
    if projection_override:
        sampled = projection_override
    else:
        proj = _compute_pca_projection(X_values)
        sampled = _sample_projection(proj, labels)
    details: dict = {
        "type": model_id,
        "projection": sampled,
        "pca_explained_variance": None,
        "cluster_sizes": _build_cluster_sizes(labels.tolist()),
    }
    if X_values.shape[1] >= 2:
        try:
            pca = PCA(n_components=2, random_state=42)
            pca.fit(X_values)
            details["pca_explained_variance"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        except Exception:
            pass
    if extra:
        details.update(extra)
    return details


KMEANS_INFO = ModelInfo(
    model_id="kmeans",
    name="K-Means",
    model_type="centroid",
    description="Particiona datos en K grupos minimizando la varianza dentro de cada grupo. Rápido y escalable para grandes conjuntos de datos.",
    category="clustering",
    supported_tasks=["unsupervised"],
    hyperparameters={
        "n_clusters": {"type": "int", "default": 3, "min": 2, "max": 20, "description": "Número de grupos"},
        "max_iter": {"type": "int", "default": 300, "min": 50, "max": 1000, "description": "Máx. iteraciones"},
        "n_init": {"type": "int", "default": 10, "min": 1, "max": 50, "description": "Número de inicializaciones"},
    },
)

DBSCAN_INFO = ModelInfo(
    model_id="dbscan",
    name="DBSCAN",
    model_type="density",
    description="Agrupamiento basado en densidad. Encuentra grupos de forma arbitraria e identifica valores atípicos. No necesita especificar K.",
    category="clustering",
    supported_tasks=["unsupervised"],
    hyperparameters={
        "eps": {"type": "float", "default": 1.0, "min": 0.01, "max": 10.0, "description": "Distancia máxima entre muestras (datos auto-escalados)"},
        "min_samples": {"type": "int", "default": 5, "min": 1, "max": 50, "description": "Mín. muestras para punto central"},
    },
)

HIERARCHICAL_INFO = ModelInfo(
    model_id="hierarchical",
    name="Agrupamiento Jerárquico",
    model_type="hierarchical",
    description="Agrupamiento jerárquico ascendente. Construye un árbol de grupos. Sin suposiciones sobre la forma de los grupos.",
    category="clustering",
    supported_tasks=["unsupervised"],
    hyperparameters={
        "n_clusters": {"type": "int", "default": 3, "min": 2, "max": 20, "description": "Número de grupos"},
        "linkage": {"type": "choice", "default": "ward", "options": ["ward", "complete", "average", "single"], "description": "Criterio de enlace"},
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
        X_values = X.values

        pca_obj = PCA(n_components=2, random_state=42)
        n_components = min(2, X_values.shape[1], X_values.shape[0])
        if n_components < 2:
            proj = _compute_pca_projection(X_values)
            centroids_2d = None
        else:
            proj = pca_obj.fit_transform(X_values)
            centroids_2d = pca_obj.transform(self.model.cluster_centers_)

        sampled = _sample_projection(proj, labels)
        extra: dict = {
            "cluster_centers": self.model.cluster_centers_.tolist(),
        }
        if centroids_2d is not None:
            extra["cluster_centers_2d"] = centroids_2d.tolist()

        n_samples = min(len(X_values), 1000)
        k_max = min(11, n_samples)
        if k_max > 2:
            elbow_data = []
            for k in range(2, k_max):
                km = KMeans(n_clusters=k, max_iter=300, n_init=5, random_state=42)
                km.fit(X_values)
                elbow_data.append({"k": k, "inertia": round(float(km.inertia_), 2)})
            extra["elbow"] = elbow_data

        algorithm_details = _clustering_algorithm_details(
            self.model_info.model_id, X_values, labels, extra,
            projection_override=sampled,
        )

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(X_values, labels),
            training_time=training_time,
            model_params=self.model.get_params(),
            algorithm_details=algorithm_details,
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
        self.model = DBSCAN(eps=1.0, min_samples=5)
        self.scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        X_values = X.values
        X_scaled = self.scaler.fit_transform(X_values)
        labels = self.model.fit_predict(X_scaled)
        training_time = time.time() - start

        core_indices = set(self.model.core_sample_indices_)
        is_core = [1 if i in core_indices else 0 for i in range(len(labels))]

        pca_obj = PCA(n_components=2, random_state=42)
        n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
        if n_components < 2:
            proj = _compute_pca_projection(X_scaled)
        else:
            proj = pca_obj.fit_transform(X_scaled)

        sampled = _sample_projection(proj, labels)
        extra: dict = {
            "is_core": is_core,
            "dbscan_eps": self.model.eps,
            "dbscan_min_samples": self.model.min_samples,
        }

        algorithm_details = _clustering_algorithm_details(
            self.model_info.model_id, X_scaled, labels, extra,
            projection_override=sampled,
        )

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(X_scaled, labels),
            training_time=training_time,
            model_params=self.model.get_params(),
            algorithm_details=algorithm_details,
        )

    def predict(self, X: pd.DataFrame) -> dict:
        X_scaled = self.scaler.transform(X.values)
        labels = self.model.fit_predict(X_scaled)
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
        self.scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> TrainingResult:
        start = time.time()
        self.model.set_params(**kwargs)
        X_values = X.values
        X_scaled = self.scaler.fit_transform(X_values)
        labels = self.model.fit_predict(X_scaled)
        training_time = time.time() - start

        pca_obj = PCA(n_components=2, random_state=42)
        n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
        if n_components < 2:
            proj = _compute_pca_projection(X_scaled)
        else:
            proj = pca_obj.fit_transform(X_scaled)

        sampled = _sample_projection(proj, labels)
        extra: dict = {
            "linkage": kwargs.get("linkage", "ward"),
        }

        n_dendrogram = min(len(X_scaled), 200)
        if X_scaled.shape[0] >= 2:
            try:
                Z = scipy_linkage(X_scaled[:n_dendrogram], method=extra["linkage"])
                extra["linkage_matrix"] = Z.tolist()
            except Exception:
                pass

        algorithm_details = _clustering_algorithm_details(
            self.model_info.model_id, X_scaled, labels, extra,
            projection_override=sampled,
        )

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=_calc_metrics(X_scaled, labels),
            training_time=training_time,
            model_params=self.model.get_params(),
            algorithm_details=algorithm_details,
        )

    def predict(self, X: pd.DataFrame) -> dict:
        X_scaled = self.scaler.transform(X.values)
        labels = self.model.fit_predict(X_scaled)
        return {"clusters": labels.tolist()}

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        self.model.set_params(**kwargs)
