import numpy as np
import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "csvs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)


def save_classification():
    n = 300
    edad = np.random.randint(18, 80, n)
    ingreso = np.random.randint(20000, 150000, n)
    gasto_mensual = np.random.randint(5000, 50000, n)
    antiguedad = np.random.uniform(0, 30, n)
    visitas_mes = np.random.randint(0, 20, n)
    score_credito = np.random.randint(300, 850, n)
    interacciones = np.random.randint(0, 50, n)
    tasa_uso = np.random.uniform(0, 100, n)

    riesgo_score = (
        (850 - score_credito) * 0.004 +
        gasto_mensual * 0.00002 +
        tasa_uso * 0.01 +
        (20 - visitas_mes) * 0.03 +
        np.random.normal(0, 0.6, n)
    )
    riesgo_labels = ["bajo", "medio", "alto"]
    thresholds = np.percentile(riesgo_score, [33, 67])
    riesgo_idx = np.digitize(riesgo_score, thresholds)

    df = pd.DataFrame({
        "edad": edad,
        "ingreso": ingreso,
        "gasto_mensual": gasto_mensual,
        "antiguedad": np.round(antiguedad, 1),
        "visitas_mes": visitas_mes,
        "score_credito": score_credito,
        "interacciones": interacciones,
        "tasa_uso": np.round(tasa_uso, 1),
        "riesgo": [riesgo_labels[i] for i in riesgo_idx],
    })

    path = os.path.join(OUTPUT_DIR, "classification.csv")
    df.to_csv(path, index=False)
    print(f"Created classification.csv: {len(df)} rows, target='riesgo' (bajo/medio/alto)")

    path = os.path.join(OUTPUT_DIR, "classification_test.csv")
    df_test = df.sample(50, random_state=99).reset_index(drop=True)
    df_test.to_csv(path, index=False)
    print(f"Created classification_test.csv: {len(df_test)} rows")


def save_regression():
    n = 300
    metros = np.random.randint(40, 300, n)
    habitaciones = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.2, 0.4, 0.25, 0.1])
    antiguedad = np.random.randint(0, 80, n)
    distancia = np.random.uniform(0, 30, n)
    calidad = np.random.randint(1, 11, n)
    seguridad = np.random.randint(1, 11, n)

    precio = (
        metros * 5000 +
        habitaciones * 200000 +
        (80 - antiguedad) * 5000 +
        (30 - distancia) * 30000 +
        calidad * 50000 +
        seguridad * 80000 +
        np.random.normal(0, 100000, n)
    ).clip(150000, 3000000).astype(int)

    df = pd.DataFrame({
        "metros_cuadrados": metros,
        "habitaciones": habitaciones,
        "antiguedad_anos": antiguedad,
        "distancia_centro": np.round(distancia, 1),
        "calidad_acabados": calidad,
        "seguridad_zona": seguridad,
        "precio": precio,
    })

    path = os.path.join(OUTPUT_DIR, "regression.csv")
    df.to_csv(path, index=False)
    print(f"Created regression.csv: {len(df)} rows, target='precio'")

    path = os.path.join(OUTPUT_DIR, "regression_test.csv")
    df_test = df.sample(50, random_state=99).reset_index(drop=True)
    df_test.to_csv(path, index=False)
    print(f"Created regression_test.csv: {len(df_test)} rows")


def save_clustering():
    n = 300
    edad = np.random.randint(18, 70, n)
    ingreso = np.random.randint(20000, 200000, n)
    educacion = np.random.randint(0, 21, n)
    experiencia = np.random.randint(0, 50, n)

    df = pd.DataFrame({
        "edad": edad,
        "ingreso_anual": ingreso,
        "nivel_educativo": educacion,
        "experiencia": experiencia,
    })

    path = os.path.join(OUTPUT_DIR, "clustering.csv")
    df.to_csv(path, index=False)
    print(f"Created clustering.csv: {len(df)} rows (unsupervised, no target column)")

    path = os.path.join(OUTPUT_DIR, "clustering_test.csv")
    df_test = df.sample(50, random_state=99).reset_index(drop=True)
    df_test.to_csv(path, index=False)
    print(f"Created clustering_test.csv: {len(df_test)} rows")


def save_neural_network():
    n = 500
    edad = np.random.randint(18, 50, n)
    ingreso = np.random.randint(0, 100000, n)
    horas_estudio = np.random.uniform(0, 60, n)
    cursos = np.random.randint(0, 30, n)
    proyectos = np.random.randint(0, 20, n)
    puntuacion = np.random.uniform(0, 100, n)

    puntaje = (
        horas_estudio * 0.6 +
        cursos * 1.5 +
        proyectos * 2.5 +
        puntuacion * 0.3 +
        np.random.normal(0, 5, n)
    ).clip(0, 100).astype(int)

    df = pd.DataFrame({
        "edad": edad,
        "ingreso_anual": ingreso,
        "horas_estudio_semana": np.round(horas_estudio, 1),
        "cursos_completados": cursos,
        "proyectos_practicos": proyectos,
        "puntuacion_prueba": np.round(puntuacion, 1),
        "puntaje_final": puntaje,
    })

    path = os.path.join(OUTPUT_DIR, "neural_network.csv")
    df.to_csv(path, index=False)
    print(f"Created neural_network.csv: {len(df)} rows, target='puntaje_final'")

    path = os.path.join(OUTPUT_DIR, "neural_network_test.csv")
    df_test = df.sample(50, random_state=99).reset_index(drop=True)
    df_test.to_csv(path, index=False)
    print(f"Created neural_network_test.csv: {len(df_test)} rows")


if __name__ == "__main__":
    save_classification()
    save_regression()
    save_clustering()
    save_neural_network()
    print(f"\nAll datasets saved to {OUTPUT_DIR}/")
    print("Each category has a training CSV and a test CSV for predictions.")
