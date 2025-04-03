import pandas as pd
import numpy as np

def random_undersample(X, y, random_state=None):
    """
    Realiza undersampling eliminando muestras de la clase mayoritaria de forma aleatoria
    hasta que todas las clases tengan el mismo número de muestras (la cantidad de la clase minoritaria).
    
    Parámetros:
      - X: DataFrame con las características.
      - y: Serie (o array) con la variable objetivo.
      - random_state: Semilla para la aleatorización.
    
    Retorna:
      - X_under: DataFrame de características re-balanceado.
      - y_under: Serie de la variable objetivo re-balanceada.
    """
    # Combinar X e y en un solo DataFrame
    df = X.copy()
    df['target'] = y
    
    # Determinar el número de muestras de la clase minoritaria
    min_count = df['target'].value_counts().min()
    
    # Para cada clase, se toma una muestra aleatoria de tamaño min_count
    df_under = df.groupby('target').apply(
        lambda x: x.sample(n=min_count, random_state=random_state)
    ).reset_index(drop=True)
    
    # Separar nuevamente en X e y
    y_under = df_under['target']
    X_under = df_under.drop('target', axis=1)
    
    return X_under, y_under


def random_oversample(X, y, random_state=None):
    """
    Realiza oversampling mediante duplicación de muestras de la clase minoritaria
    hasta que todas las clases tengan el mismo número de muestras que la clase mayoritaria.
    
    Parámetros:
      - X: DataFrame con las características.
      - y: Serie (o array) con la variable objetivo.
      - random_state: Semilla para la aleatorización (opcional).
    
    Retorna:
      - X_over: DataFrame con las características re-balanceado.
      - y_over: Serie con la variable objetivo re-balanceada.
    """
    # Combinar X e y en un único DataFrame
    df = X.copy()
    df['target'] = y
    
    # Determinar el número de muestras de la clase mayoritaria
    max_count = df['target'].value_counts().max()
    
    # Para cada clase, se toma una muestra aleatoria con reemplazo de tamaño max_count
    df_over = df.groupby('target', group_keys=False).apply(
        lambda x: x.sample(n=max_count, replace=True, random_state=random_state)
    ).reset_index(drop=True)
    
    # Separar nuevamente en X e y
    y_over = df_over['target']
    X_over = df_over.drop('target', axis=1)
    
    return X_over, y_over

def smote_oversample(X, y, k=5, random_state=None):
    """
    Realiza oversampling mediante SMOTE para balancear un dataset binario,
    generando muestras sintéticas de la clase minoritaria hasta que tenga el mismo número de muestras
    que la clase mayoritaria.

    Parámetros:
      - X: DataFrame con las características.
      - y: Serie (o array) con la variable objetivo.
      - k: número de vecinos a considerar para la generación de muestras sintéticas.
      - random_state: semilla para la aleatorización (opcional).

    Retorna:
      - X_smote: DataFrame con las características, incluyendo las muestras sintéticas.
      - y_smote: Serie con la variable objetivo, incluyendo las muestras sintéticas.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convertir y a array para facilitar el manejo
    y = np.array(y)
    X = X.copy()
    
    # Identificar la clase mayoritaria y la minoritaria (se asume problema binario)
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    
    # Extraer las muestras correspondientes a la clase minoritaria
    X_min = X[y == minority_class].values  # Convertimos a array para facilitar cálculos
    n_min = X_min.shape[0]
    n_maj = class_counts[majority_class]
    
    # Número de muestras sintéticas a generar
    n_synthetic = n_maj - n_min
    
    synthetic_samples = []
    
    # Para cada muestra sintética a generar:
    for _ in range(n_synthetic):
        # Seleccionar aleatoriamente una muestra de la clase minoritaria
        idx = np.random.randint(0, n_min)
        sample = X_min[idx]
        
        # Calcular distancias Euclidianas desde esta muestra a todas las muestras minoritarias
        distances = np.linalg.norm(X_min - sample, axis=1)
        # Excluir la propia muestra
        distances[idx] = np.inf
        
        # Seleccionar los k vecinos más cercanos
        nn_indices = np.argsort(distances)[:k]
        # Seleccionar aleatoriamente uno de los vecinos
        nn_idx = np.random.choice(nn_indices)
        neighbor = X_min[nn_idx]
        
        # Generar una nueva muestra sintética mediante interpolación:
        # new_sample = sample + gap * (neighbor - sample), donde gap es un valor aleatorio entre 0 y 1.
        gap = np.random.rand()
        synthetic_sample = sample + gap * (neighbor - sample)
        synthetic_samples.append(synthetic_sample)
    
    # Convertir las muestras sintéticas a un DataFrame con las mismas columnas que X
    synthetic_samples = np.array(synthetic_samples)
    X_synthetic = pd.DataFrame(synthetic_samples, columns=X.columns)
    y_synthetic = np.array([minority_class] * n_synthetic)
    
    # Combinar el dataset original con las muestras sintéticas
    X_smote = pd.concat([X, X_synthetic], axis=0).reset_index(drop=True)
    y_smote = np.concatenate([y, y_synthetic])
    y_smote = pd.Series(y_smote, name='target')
    
    return X_smote, y_smote

