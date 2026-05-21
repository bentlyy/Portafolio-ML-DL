import { type ModelInfo } from '../api/ml';

interface Props {
  model: ModelInfo | null;
}

const explanations: Record<string, { summary: string; howItWorks: string; bestFor: string; tips: string }> = {
  random_forest: {
    summary: 'Ensamblaje de múltiples árboles de decisión que votan por la clase más popular.',
    howItWorks: 'Construye cientos de árboles de decisión, cada uno entrenado con una muestra aleatoria de los datos (bootstrap). Cada árbol se extiende hasta la profundidad máxima especificada sin podar. Para predecir, cada árbol "vota" por una clase y se toma la mayoría. Este proceso de bagging reduce drásticamente el sobreajuste comparado con un solo árbol.',
    bestFor: 'Datos con relaciones no lineales, muchas características, valores faltantes. Funciona bien con datasets desbalanceados.',
    tips: 'Aumenta n_estimators para mejor precisión (con rendimiento decreciente). Reduce max_depth si hay sobreajuste. Aumenta min_samples_split para árboles más simples.',
  },
  gradient_boosting: {
    summary: 'Ensamblaje secuencial donde cada árbol corrige los errores del anterior.',
    howItWorks: 'Comienza con un árbol simple que predice la media de los datos. Luego, cada nuevo árbol se entrena para predecir los residuos (errores) del conjunto anterior. La contribución de cada árbol se escala por la tasa de aprendizaje (learning_rate), lo que permite un progreso lento y controlado hacia el objetivo. Es como un "equipo" que aprende de sus errores iterativamente.',
    bestFor: 'Datos tabulares en competiciones (Kaggle). Relaciones complejas donde se necesita alta precisión.',
    tips: 'Usa learning_rate más bajo (0.01-0.05) con más n_estimators para mejor generalización. max_depth bajo (3-5) funciona mejor. Monitorea sobreajuste con validación cruzada.',
  },
  svm: {
    summary: 'Encuentra el hiperplano óptimo que separa las clases con el mayor margen posible.',
    howItWorks: 'SVM busca la línea (o hiperplano en altas dimensiones) que maximiza la distancia entre las clases. Los puntos más cercanos al límite se llaman "vectores de soporte" y definen el margen. Con kernels (RBF, polinomial), transforma los datos a un espacio de mayor dimensión donde sean separables linealmente, sin necesidad de calcular explícitamente la transformación (kernel trick).',
    bestFor: 'Datos de alta dimensionalidad (ej. texto, genómica). Problemas donde las clases son claramente separables.',
    tips: 'Kernel RBF es buena opción por defecto. Ajusta C: valores altos = margen más estrecho (menos errores de entrenamiento). Gamma controla la influencia de cada punto: valores altos = más complejo.',
  },
  logistic_regression: {
    summary: 'Modelo lineal que estima probabilidades usando la función sigmoide.',
    howItWorks: 'Calcula una combinación lineal de las características de entrada, luego aplica la función sigmoide para convertir el resultado en una probabilidad entre 0 y 1. Si la probabilidad es > 0.5, predice clase 1; si no, clase 0. Los coeficientes indican la importancia y dirección de cada característica. Se entrena maximizando la verosimilitud de los datos observados.',
    bestFor: 'Línea base para clasificación binaria. Problemas donde la interpretabilidad es clave (medicina, finanzas).',
    tips: 'Usa penalty="l1" para selección automática de características. Aumenta C para menos regularización. Estandariza los datos para mejor convergencia.',
  },
  linear_regression: {
    summary: 'Ajusta una línea recta que minimiza los errores cuadrados entre predicciones y valores reales.',
    howItWorks: 'Encuentra los coeficientes (pendientes) que minimizan la suma de errores cuadrados (MCO). Cada coeficiente representa el cambio esperado en la variable objetivo por cada unidad de cambio en la característica correspondiente. Es un modelo paramétrico que asume una relación lineal entre características y objetivo.',
    bestFor: 'Relaciones lineales simples. Problemas donde la interpretabilidad es esencial. Línea base para regresión.',
    tips: 'Verifica que la relación sea aproximadamente lineal (gráfica de residuos). Normaliza las características si tienen escalas muy diferentes. Cuidado con outliers: afectan mucho los coeficientes.',
  },
  ridge: {
    summary: 'Regresión lineal con regularización L2 que reduce la magnitud de los coeficientes.',
    howItWorks: 'Similar a regresión lineal, pero añade un término de penalización a la función de costo: la suma de los cuadrados de los coeficientes multiplicada por α (alpha). Esto "encoge" los coeficientes hacia cero, reduciendo la varianza del modelo a costa de un pequeño sesgo. Es especialmente útil cuando hay multicolinealidad (características correlacionadas entre sí).',
    bestFor: 'Datos con muchas características correlacionadas. Cuando la regresión lineal simple sobreajusta.',
    tips: 'Alpha grande = más regularización (coeficientes más pequeños). Usa validación cruzada para encontrar el alpha óptimo. Estandariza las características primero.',
  },
  random_forest_regressor: {
    summary: 'Ensamblaje de árboles de decisión que promedia sus predicciones numéricas.',
    howItWorks: 'Construye múltiples árboles de regresión, cada uno con una muestra bootstrap de los datos. Cada árbol predice un valor numérico, y el resultado final es el promedio de todas las predicciones. Esto reduce la varianza comparado con un solo árbol, manteniendo la capacidad de capturar relaciones no lineales e interacciones entre características.',
    bestFor: 'Regresión con relaciones no lineales complejas. Datos tabulares donde las características interactúan.',
    tips: 'n_estimators más altos mejoran la precisión pero con rendimiento decreciente. max_depth profundo puede sobreajustar. min_samples_split más alto da árboles más simples.',
  },
  gradient_boosting_regressor: {
    summary: 'Ensamblaje secuencial que corrige errores de predicción iterativamente.',
    howItWorks: 'Comienza prediciendo el promedio de los datos objetivo. En cada iteración, entrena un árbol para predecir los residuos (errores) del modelo actual. Los árboles son deliberadamente débiles (max_depth bajo) y su contribución se escala por learning_rate. El proceso se repite hasta n_estimators rondas, mejorando gradualmente donde el modelo falla.',
    bestFor: 'Regresión donde se necesita alta precisión predictiva. Competiciones de datos tabulares.',
    tips: 'learning_rate bajo (0.01-0.1) con más árboles da mejor generalización. max_depth 3-5 suele ser óptimo. Monitorea el error en validación para detectar sobreajuste.',
  },
  kmeans: {
    summary: 'Divide los datos en K grupos basándose en la cercanía a centroides.',
    howItWorks: 'Inicializa K centroides aleatoriamente. Asigna cada punto al centroide más cercano (distancia euclidiana). Recalcula los centroides como el promedio de los puntos en cada grupo. Repite los pasos 2 y 3 hasta que los centroides no cambien. El objetivo es minimizar la suma de distancias al cuadrado dentro de cada grupo (inercia).',
    bestFor: 'Segmentación de clientes, compresión de imágenes, datos con grupos esféricos bien separados.',
    tips: 'Usa el método del codo (inercia vs K) para elegir K. Estandariza los datos primero. Ejecuta con diferentes semillas (n_init) para evitar óptimos locales.',
  },
  dbscan: {
    summary: 'Agrupa puntos densamente conectados e identifica automáticamente valores atípicos.',
    howItWorks: 'Define un punto como "central" si tiene al menos min_samples puntos dentro de una distancia eps. Todos los puntos dentro de eps de un punto central pertenecen al mismo grupo. Los puntos que no son centrales ni están cerca de uno se clasifican como ruido (-1). Los grupos pueden tener formas arbitrarias, no solo esféricas.',
    bestFor: 'Datos con grupos de forma arbitraria. Detección de anomalías. Cuando no sabes cuántos grupos hay.',
    tips: 'eps es el parámetro más importante: muy pequeño = muchos grupos, muy grande = un solo grupo. Usa un gráfico k-distance para estimar eps. Datos con densidades muy variables pueden ser problemáticos.',
  },
  hierarchical: {
    summary: 'Construye una jerarquía de grupos fusionando iterativamente los más cercanos.',
    howItWorks: 'Cada punto comienza como su propio grupo (bottom-up). En cada paso, fusiona los dos grupos más cercanos según el criterio de enlace (linkage): ward minimiza la varianza dentro de grupos, complete usa la distancia máxima entre puntos, average usa la distancia promedio. El proceso continúa hasta que queda un solo grupo, formando un dendrograma (árbol). Se corta en el nivel deseado (n_clusters).',
    bestFor: 'Datos con estructura jerárquica natural (taxonomías, genealogías). Exploración de datos con dendrogramas.',
    tips: 'Linkage="ward" funciona bien con grupos esféricos. Complete tiende a grupos compactos. Single puede crear cadenas. El dendrograma ayuda a visualizar la estructura.',
  },
  neural_network: {
    summary: 'Red neuronal multicapa que aprende representaciones complejas de los datos.',
    howItWorks: 'Compuesta por capas de neuronas interconectadas. Cada neurona recibe entradas, las multiplica por pesos, suma un sesgo y aplica una función de activación no lineal (ReLU, tanh). Las capas ocultas transforman progresivamente los datos en representaciones más abstractas. El entrenamiento usa retropropagación (backpropagation) y el optimizador Adam para ajustar los pesos minimizando el error. Soporta early stopping para evitar sobreajuste.',
    bestFor: 'Aprendizaje de representaciones complejas. Problemas donde las relaciones no son lineales ni capturables con métodos tradicionales.',
    tips: 'Añade más capas para mayor capacidad (riesgo de sobreajuste). Usa ReLU por defecto en capas ocultas. Early stopping ayuda a evitar sobreajuste. Estandariza los datos siempre. Reduce learning_rate si la pérdida oscila.',
  },
};

export default function AlgorithmExplanations({ model }: Props) {
  if (!model) {
    return (
      <div className="algorithm-explanations">
        <h3>¿Cómo funciona cada algoritmo?</h3>
        <p className="explanations-intro">
          Selecciona un modelo en el panel de la izquierda para ver una explicación detallada de cómo funciona.
        </p>
      </div>
    );
  }

  const info = explanations[model.model_id];
  if (!info) return null;

  return (
    <div className="algorithm-explanations">
      <h3>{model.name}</h3>
      <p className="explanations-intro">{info.summary}</p>

      <div className="explanation-category">
        <h4>¿Cómo funciona?</h4>
        <p>{info.howItWorks}</p>
      </div>

      <div className="explanation-category">
        <h4>¿Cuándo usarlo?</h4>
        <p>{info.bestFor}</p>
      </div>

      <div className="explanation-category">
        <h4>Consejos prácticos</h4>
        <p>{info.tips}</p>
      </div>
    </div>
  );
}
