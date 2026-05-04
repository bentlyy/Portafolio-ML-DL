from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# dataset dummy pero realista
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "../model/model.pkl")

print("Modelo entrenado y guardado")