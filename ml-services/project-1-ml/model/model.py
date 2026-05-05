import joblib
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if os.path.exists(MODEL_PATH):
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
else:
    logger.warning("No trained model found. Using fallback model for development.")
    model = LinearRegression()
    model.fit(np.array([[1], [2], [3], [4], [5]]), np.array([2, 4, 6, 8, 10]))
