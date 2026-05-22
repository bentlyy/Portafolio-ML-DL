import numpy as np
import pandas as pd
import time
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.models.base import BaseModel, ModelInfo, TrainingResult
from sklearn.metrics import confusion_matrix, classification_report

NEURAL_NETWORK_INFO = ModelInfo(
    model_id="neural_network",
    name="Red Neuronal (MLP)",
    model_type="deep_learning",
    description="Perceptrón Multicapa con capas ocultas configurables, funciones de activación y regularización. Demuestra fundamentos de deep learning.",
    category="neural_network",
    supported_tasks=["binary", "multiclass", "regression"],
    hyperparameters={
        "hidden_layer_sizes": {"type": "string", "default": "128,64", "description": "Tamaños de capas ocultas separados por coma (ej. 128,64)"},
        "activation": {"type": "choice", "default": "relu", "options": ["relu", "tanh", "logistic"], "description": "Función de activación"},
        "solver": {"type": "choice", "default": "adam", "options": ["adam", "sgd", "lbfgs"], "description": "Algoritmo de optimización"},
        "learning_rate_init": {"type": "float", "default": 0.001, "min": 0.0001, "max": 0.1, "description": "Tasa de aprendizaje inicial"},
        "max_iter": {"type": "int", "default": 1000, "min": 50, "max": 2000, "description": "Máx. iteraciones"},
        "alpha": {"type": "float", "default": 0.0001, "min": 0.00001, "max": 1.0, "description": "Regularización L2"},
    },
)


class NeuralNetworkModel(BaseModel):
    def __init__(self):
        super().__init__(NEURAL_NETWORK_INFO)
        self.model = None
        self.scaler = StandardScaler()
        self.task_type = "classification"

    def train(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> TrainingResult:
        start = time.time()

        hidden_str = kwargs.get("hidden_layer_sizes", "128,64")
        if isinstance(hidden_str, str):
            hidden_layers = tuple(int(x) for x in hidden_str.split(","))
        else:
            hidden_layers = tuple(hidden_str)

        activation = kwargs.get("activation", "relu")
        solver = kwargs.get("solver", "adam")
        lr = float(kwargs.get("learning_rate_init", 0.001))
        max_iter = int(kwargs.get("max_iter", 1000))
        alpha = float(kwargs.get("alpha", 0.0001))

        X_scaled = self.scaler.fit_transform(X.values)

        is_regression = y is None or (pd.api.types.is_numeric_dtype(y) and y.nunique() > 10)

        if is_regression:
            self.task_type = "regression"
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver=solver,
                learning_rate_init=lr,
                max_iter=max_iter,
                alpha=alpha,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
            y_values = y.astype(float).values if y is not None else np.zeros(len(X))
        else:
            self.task_type = "classification"
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver=solver,
                learning_rate_init=lr,
                max_iter=max_iter,
                alpha=alpha,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
            y_values = y.values

        self.model.fit(X_scaled, y_values)
        training_time = time.time() - start

        y_pred = self.model.predict(X_scaled)

        if self.task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                "mse": round(float(mean_squared_error(y_values, y_pred)), 4),
                "rmse": round(float(mean_squared_error(y_values, y_pred, squared=False)), 4),
                "mae": round(float(mean_absolute_error(y_values, y_pred)), 4),
                "r2": round(float(r2_score(y_values, y_pred)), 4),
            }
            residuals = (y_values - y_pred).tolist()
        else:
            metrics = {
                "accuracy": round(float(accuracy_score(y_values, y_pred)), 4),
                "precision": round(float(precision_score(y_values, y_pred, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_values, y_pred, average="weighted", zero_division=0)), 4),
                "f1": round(float(f1_score(y_values, y_pred, average="weighted", zero_division=0)), 4),
            }
            residuals = None

        model_params = self.model.get_params()
        if "hidden_layer_sizes" in model_params:
            model_params["hidden_layer_sizes"] = list(model_params["hidden_layer_sizes"])

        n_layers = getattr(self.model, "n_layers_", 0)
        total_params = 0
        for w in getattr(self.model, "coefs_", []):
            total_params += w.size
        for b in getattr(self.model, "intercepts_", []):
            total_params += b.size

        architecture = (
            [X_scaled.shape[1]]
            + list(hidden_layers)
            + [self.model.n_outputs_]
        )

        best_loss_val = getattr(self.model, "best_loss_", None)
        if best_loss_val is None or (isinstance(best_loss_val, float) and np.isinf(best_loss_val)):
            best_loss_val = 0.0

        algorithm_details = {
            "loss_curve": [round(float(v), 6) for v in getattr(self.model, "loss_curve_", [])],
            "validation_scores": [round(float(v), 6) for v in getattr(self.model, "validation_scores_", [])],
            "n_iter": int(getattr(self.model, "n_iter_", 0)),
            "best_loss": round(float(best_loss_val), 6),
            "n_layers": int(n_layers),
            "architecture": architecture,
            "total_parameters": int(total_params),
            "activation": activation,
            "solver": solver,
            "converged": int(getattr(self.model, "n_iter_", 0)) < max_iter,
            "layer_weights_shapes": [list(w.shape) for w in getattr(self.model, "coefs_", [])],
        }

        result = TrainingResult(
            model_id=self.model_info.model_id,
            metrics=metrics,
            training_time=training_time,
            model_params=model_params,
            algorithm_details=algorithm_details,
        )

        if self.task_type == "regression":
            result.predictions = [round(float(v), 4) for v in y_pred]
            result.actual_values = [round(float(v), 4) for v in y_values]
            result.residuals = [round(float(v), 4) for v in residuals]
        else:
            cm = confusion_matrix(y_values, y_pred).tolist()
            result.confusion_matrix = cm
            report_dict = classification_report(
                y_values, y_pred, output_dict=True, zero_division=0
            )
            clean_report = {}
            for k, v in report_dict.items():
                if isinstance(v, dict):
                    clean_report[k] = {
                        "precision": round(float(v.get("precision", 0)), 4),
                        "recall": round(float(v.get("recall", 0)), 4),
                        "f1-score": round(float(v.get("f1-score", 0)), 4),
                        "support": int(v.get("support", 0)),
                    }
                else:
                    clean_report[k] = round(float(v), 4)
            result.classification_report = clean_report
            result.predictions = [round(float(v), 4) for v in y_pred]
            result.actual_values = [round(float(v), 4) for v in y_values]

        return result

    def predict(self, X: pd.DataFrame) -> dict:
        X_scaled = self.scaler.transform(X.values)
        predictions = self.model.predict(X_scaled)

        if self.task_type == "regression":
            return {"predictions": predictions.tolist()}
        else:
            probs = self.model.predict_proba(X_scaled)
            classes = self.model.classes_.tolist()
            return {
                "predictions": predictions.tolist(),
                "probabilities": probs.tolist(),
                "classes": classes,
            }

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        if self.model is not None:
            self.model.set_params(**kwargs)
