import pandas as pd
import numpy as np

def zscore_standardize2(X, return_params=False, exclude_cols=None):
    """
    Estandariza el dataset X utilizando z-score para todas las columnas
    excepto aquellas listadas en exclude_cols.
    
    Parámetros:
      - X: pandas DataFrame.
      - return_params (bool): si es True, retorna además la media y la desviación
        estándar de cada feature normalizada.
      - exclude_cols (list): lista de nombres de columnas a excluir de la normalización.
    
    Retorna:
      - X_std: DataFrame estandarizado (con las columnas excluidas sin cambios).
      - (opcional) mean, std: Series con la media y la desviación estándar de cada feature normalizada.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Esta función solo está implementada para pandas DataFrames.")
    
    # Crear una copia para no modificar el DataFrame original
    X_std = X.copy()
    
    # Si no se especifica, no se excluye ninguna columna
    if exclude_cols is None:
        exclude_cols = []
    
    # Determinar las columnas que se normalizarán (todas excepto las de exclude_cols)
    columns_to_normalize = [col for col in X.columns if col not in exclude_cols]
    
    # Calcular la media y la desviación estándar solo para las columnas a normalizar
    mean = X_std[columns_to_normalize].mean()
    std = X_std[columns_to_normalize].std(ddof=0)  # ddof=0 para la desviación estándar poblacional
    
    # Aplicar la estandarización
    X_std[columns_to_normalize] = (X_std[columns_to_normalize] - mean) / std
    
    if return_params:
        return X_std, mean, std
    else:
        return X_std


