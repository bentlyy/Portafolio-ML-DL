import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.models.base import BaseModel, ModelInfo, TrainingResult

NEURAL_NETWORK_INFO = ModelInfo(
    model_id="neural_network",
    name="Neural Network (MLP)",
    model_type="deep_learning",
    description="Multi-Layer Perceptron with configurable architecture. Demonstrates deep learning fundamentals with PyTorch.",
    category="neural_network",
    supported_tasks=["binary", "multiclass", "regression"],
    hyperparameters={
        "hidden_layers": {"type": "string", "default": "128,64", "description": "Comma-separated hidden layer sizes"},
        "learning_rate": {"type": "float", "default": 0.001, "min": 0.0001, "max": 0.1, "description": "Learning rate"},
        "epochs": {"type": "int", "default": 100, "min": 10, "max": 1000, "description": "Training epochs"},
        "batch_size": {"type": "int", "default": 32, "min": 8, "max": 256, "description": "Batch size"},
        "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.8, "description": "Dropout rate"},
    },
)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralNetworkModel(BaseModel):
    def __init__(self):
        super().__init__(NEURAL_NETWORK_INFO)
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_type = "classification"
        self.output_dim = 2
        self.input_dim = None

    def _build_model(self, input_dim, output_dim, hidden_layers, dropout):
        return MLP(input_dim, hidden_layers, output_dim, dropout).to(self.device)

    def train(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> TrainingResult:
        start = time.time()

        hidden_layers_str = kwargs.get("hidden_layers", "128,64")
        if isinstance(hidden_layers_str, str):
            hidden_layers = [int(x) for x in hidden_layers_str.split(",")]
        else:
            hidden_layers = hidden_layers_str

        learning_rate = float(kwargs.get("learning_rate", 0.001))
        epochs = int(kwargs.get("epochs", 100))
        batch_size = int(kwargs.get("batch_size", 32))
        dropout = float(kwargs.get("dropout", 0.2))

        self.input_dim = X.shape[1]

        X_scaled = self.scaler.fit_transform(X.values)

        is_regression = y is None or pd.api.types.is_numeric_dtype(y) and y.nunique() > 10

        if is_regression:
            self.task_type = "regression"
            self.output_dim = 1
            y_values = y.astype(float).values if y is not None else np.zeros(len(X))
        else:
            self.task_type = "classification"
            unique_classes = np.unique(y)
            self.output_dim = len(unique_classes)
            class_map = {c: i for i, c in enumerate(unique_classes)}
            y_values = np.array([class_map[c] for c in y])

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_values).to(self.device) if self.task_type == "regression" else torch.LongTensor(y_values).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = self._build_model(self.input_dim, self.output_dim, hidden_layers, dropout)

        criterion = nn.MSELoss() if self.task_type == "regression" else nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        training_time = time.time() - start

        y_pred = self.predict_internal(X_scaled)

        if self.task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                "mse": round(float(mean_squared_error(y_values, y_pred)), 4),
                "rmse": round(float(mean_squared_error(y_values, y_pred, squared=False)), 4),
                "mae": round(float(mean_absolute_error(y_values, y_pred)), 4),
                "r2": round(float(r2_score(y_values, y_pred)), 4),
            }
        else:
            y_pred_class = np.argmax(y_pred, axis=1)
            metrics = {
                "accuracy": round(float(accuracy_score(y_values, y_pred_class)), 4),
                "precision": round(float(precision_score(y_values, y_pred_class, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_values, y_pred_class, average="weighted", zero_division=0)), 4),
                "f1": round(float(f1_score(y_values, y_pred_class, average="weighted", zero_division=0)), 4),
            }

        return TrainingResult(
            model_id=self.model_info.model_id,
            metrics=metrics,
            training_time=training_time,
            model_params={
                "hidden_layers": hidden_layers,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "dropout": dropout,
                "device": str(self.device),
            },
        )

    def predict_internal(self, X_scaled):
        self.model.eval()
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            output = self.model(X_tensor)
            if self.task_type == "regression":
                return output.squeeze().cpu().numpy()
            else:
                return torch.softmax(output, dim=1).cpu().numpy()

    def predict(self, X: pd.DataFrame) -> dict:
        X_scaled = self.scaler.transform(X.values)
        predictions = self.predict_internal(X_scaled)

        if self.task_type == "regression":
            return {"predictions": predictions.tolist()}
        else:
            probs = predictions.tolist()
            pred_classes = np.argmax(predictions, axis=1).tolist()
            return {
                "predictions": pred_classes,
                "probabilities": probs,
            }

    def get_hyperparameters(self) -> dict:
        return self.model_info.to_dict()["hyperparameters"]

    def set_hyperparameters(self, **kwargs) -> None:
        pass
