import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import math

class knnimputation:
    """
    Clase para imputar valores faltantes en un DataFrame usando KNN.
    
    Soporta:
      - Variables numéricas: utiliza sklearn.impute.KNNImputer.
      - Variables categóricas: codifica con LabelEncoder, aplica KNNImputer, redondea y decodifica.
    
    Parámetros:
      k (int): número de vecinos a considerar (default=5).
    """
    
    def __init__(self, k=5):
        self.k = k
        # Diccionario para almacenar los label encoder de cada variable categórica
        self.label_encoders = {}
    
    def fit_transform(self, df):
        """
        Imputa los valores faltantes del DataFrame y retorna una copia sin NaN.
        
        Parámetros:
          df (pd.DataFrame): DataFrame de entrada con posibles valores NaN.
        
        Retorna:
          pd.DataFrame: DataFrame con valores imputados.
        """
        # Crear una copia para no modificar el original
        df_imputed = df.copy()
        
        # Separar columnas numéricas y categóricas
        numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df_imputed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Imputación para columnas numéricas
        if numeric_cols:
            imputer_num = KNNImputer(n_neighbors=self.k)
            df_imputed[numeric_cols] = imputer_num.fit_transform(df_imputed[numeric_cols])
        
        # 2. Imputación para columnas categóricas
        # Se usa LabelEncoder para transformar las categorías a números.
        for col in cat_cols:
            le = LabelEncoder()
            # Rellenar NaN con un marcador temporal (por ejemplo, 'missing')
            df_imputed[col] = df_imputed[col].fillna('missing')
            df_imputed[col] = le.fit_transform(df_imputed[col])
            self.label_encoders[col] = le  # Guardar el encoder para luego decodificar
        
        if cat_cols:
            imputer_cat = KNNImputer(n_neighbors=self.k)
            df_imputed[cat_cols] = imputer_cat.fit_transform(df_imputed[cat_cols])
            # Los valores imputados en variables categóricas son float, se redondean y convierten a int
            df_imputed[cat_cols] = df_imputed[cat_cols].round(0).astype(int)
            # Convertir de vuelta a las categorías originales
            for col in cat_cols:
                le = self.label_encoders[col]
                df_imputed[col] = le.inverse_transform(df_imputed[col])
        
        return df_imputed

