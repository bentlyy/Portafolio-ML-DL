import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_iris():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = data.target_names[data.target]
    path = os.path.join(OUTPUT_DIR, "iris.csv")
    df.to_csv(path, index=False)
    print(f"Saved iris.csv: {len(df)} rows, {len(df.columns)} columns")


def save_wine():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["quality_class"] = data.target
    path = os.path.join(OUTPUT_DIR, "wine.csv")
    df.to_csv(path, index=False)
    print(f"Saved wine.csv: {len(df)} rows, {len(df.columns)} columns")


def save_breast_cancer():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = data.target_names[data.target]
    path = os.path.join(OUTPUT_DIR, "breast_cancer.csv")
    df.to_csv(path, index=False)
    print(f"Saved breast_cancer.csv: {len(df)} rows, {len(df.columns)} columns")


def save_titanic():
    np.random.seed(42)
    n = 891
    pclass = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(["male", "female"], n, p=[0.65, 0.35])
    age = np.random.normal(30, 14, n).clip(0.4, 80)
    sibsp = np.random.poisson(0.5, n)
    parch = np.random.poisson(0.4, n)
    fare = np.random.exponential(30, n)
    embarked = np.random.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09])

    survived = (
        (pclass == 1) * 0.6 +
        (pclass == 2) * 0.4 +
        (pclass == 3) * 0.25 +
        (sex == "female") * 0.3 -
        (age > 50) * 0.1 +
        (fare > 50) * 0.1 +
        np.random.normal(0, 0.2, n)
    )
    survived = (survived > 0.5).astype(int)

    df = pd.DataFrame({
        "pclass": pclass,
        "sex": sex,
        "age": np.round(age, 1),
        "sibsp": sibsp,
        "parch": parch,
        "fare": np.round(fare, 2),
        "embarked": embarked,
        "survived": survived,
    })
    path = os.path.join(OUTPUT_DIR, "titanic.csv")
    df.to_csv(path, index=False)
    print(f"Saved titanic.csv: {len(df)} rows, {len(df.columns)} columns")


def save_housing():
    np.random.seed(42)
    n = 500
    size = np.random.uniform(500, 5000, n)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.35, 0.2, 0.1])
    age = np.random.randint(0, 100, n)
    garage = np.random.choice([0, 1, 2, 3], n, p=[0.15, 0.4, 0.3, 0.15])

    price = (
        size * 150 +
        bedrooms * 15000 -
        age * 500 +
        garage * 25000 +
        np.random.normal(0, 30000, n)
    ).clip(50000, 1500000)

    df = pd.DataFrame({
        "size_sqft": np.round(size, 0),
        "bedrooms": bedrooms,
        "age_years": age,
        "garage_spaces": garage,
        "price": np.round(price, 0),
    })
    path = os.path.join(OUTPUT_DIR, "housing.csv")
    df.to_csv(path, index=False)
    print(f"Saved housing.csv: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    save_iris()
    save_wine()
    save_breast_cancer()
    save_titanic()
    save_housing()
    print(f"\nAll datasets saved to {OUTPUT_DIR}")
