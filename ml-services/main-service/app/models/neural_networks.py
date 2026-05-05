import numpy as np
import pandas as pd
import time
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.models.base import BaseModel, ModelInfo, TrainingResult

NEURAL_NETWORK_INFO = ModelInfo(
    model_id="neural_network",
    name="Neural Network (MLP)",
    model_type="deep_learning",
    description="Multi-Layer Perceptron with configurable hidden layers, activation functions, and regularization. Demonstrates deep learning fundamentals.",
    category="neural_network",
    supported_tasks=["binary", "multiclass", "regression"],
    hyperparameters={
        "hidden_layer_sizes": {"type": "string", "default": "128,64", "description": "Comma-separated hidden layer sizes (e.g., 128,64)"},
        "activation": {"type": "choice", "default": "relu", "options": ["relu", "tanh", "logistic"], "description": "Activation function"},
        "solver": {"type": "choice", "default": "adam", "options": ["adam", "sgd", "lbfgs"], "description": "Optimization algorithm"},
        "learning_rate_init": {"type": "float", "default": 0.001, "min": 0.0001, "max": 0.1, "description": "Initial learning rate"},
        "max_iter": {"type": "int", "default": 300, "min": 50, "max": 2000, "description": "Max iterations"},
        "alpha": {"type": "float", "default": 0.0001, "min": 0.00001, "max": 1.0, "description": "L2 regularization"},
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
        max_iter = int(kwargs.get("max_iter", 300))
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
        else:
            metrics = {
                "accuracy": round(float(accuracy_score(y_values, y_pred)), 4),
                "precision": round(float(precision_score(y_values, y_pred, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_values, y_pred, average="weighted", zero_division=0)), 4),
                "f1": round(float(f1_score(y_values, y_pred, average="weighted", zero_division=0)), 4),
            }

        model_params = self.model.get_params()
        if "hidden_layer_sizes" in model_params:
            model_params["hidden_layer_sizes"] = list(model_params["hidden_layer_sizes"])

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=metrics,
            training_time=training_time,
            model_params=model_params,
        )

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
        pass
