import pandas as pd

class OneHotEncoderCustom:
    """
    Clase para aplicar One Hot Encoding a las variables categóricas 'CellType' y 'GeneticMutation'.

    Las categorías definidas son:
      - CellType: ['Epithelial', 'Mesenchymal', 'Unknown']
      - GeneticMutation: ['Present', 'Absent', 'Unknown']
    """
    
    def __init__(self):
        self.cell_categories = ['Epthlial', 'Mesnchymal', '???']
        self.mutation_categories = ['Presnt', 'Absnt', '???']
        self.fitted = False

    def fit(self, df):
        """
        Verifica que las columnas 'CellType' y 'GeneticMutation' existan en el DataFrame.
        
        Parámetros:
          df (pd.DataFrame): DataFrame de entrada.
        
        Retorna:
          self
        """
        missing_columns = []
        if 'CellType' not in df.columns:
            missing_columns.append('CellType')
        if 'GeneticMutation' not in df.columns:
            missing_columns.append('GeneticMutation')
        if missing_columns:
            raise ValueError(f"Las siguientes columnas no se encuentran en el DataFrame: {missing_columns}")
        
        self.fitted = True
        return self

    def transform(self, df):
        """
        Transforma el DataFrame aplicando One Hot Encoding a 'CellType' y 'GeneticMutation'
        asegurando que las columnas resultantes sean numéricas (0 y 1) y que se respeten las categorías
        definidas.
        
        Parámetros:
          df (pd.DataFrame): DataFrame de entrada.
        
        Retorna:
          pd.DataFrame: DataFrame transformado, sin las columnas originales.
        """
        if not self.fitted:
            raise Exception("El encoder no ha sido ajustado. Ejecuta el método fit() antes de transformar.")
        
        df_encoded = df.copy()
        
        # Procesar la columna CellType
        if 'CellType' in df_encoded.columns:
            dummies_cell = pd.get_dummies(df_encoded['CellType'], prefix='CellType')
            # Aseguramos que existan las columnas para todas las categorías definidas
            for cat in self.cell_categories:
                col_name = f'CellType_{cat}'
                if col_name not in dummies_cell.columns:
                    dummies_cell[col_name] = 0
            # Reordenamos las columnas según el orden deseado
            dummies_cell = dummies_cell[[f'CellType_{cat}' for cat in self.cell_categories]]
            dummies_cell = dummies_cell.astype(int)
            df_encoded = pd.concat([df_encoded, dummies_cell], axis=1)
            df_encoded.drop('CellType', axis=1, inplace=True)
        
        # Procesar la columna GeneticMutation
        if 'GeneticMutation' in df_encoded.columns:
            dummies_mut = pd.get_dummies(df_encoded['GeneticMutation'], prefix='GeneticMutation')
            for cat in self.mutation_categories:
                col_name = f'GeneticMutation_{cat}'
                if col_name not in dummies_mut.columns:
                    dummies_mut[col_name] = 0
            dummies_mut = dummies_mut[[f'GeneticMutation_{cat}' for cat in self.mutation_categories]]
            dummies_mut = dummies_mut.astype(int)
            df_encoded = pd.concat([df_encoded, dummies_mut], axis=1)
            df_encoded.drop('GeneticMutation', axis=1, inplace=True)
        
        return df_encoded

    def fit_transform(self, df):
        """
        Ajusta el encoder y transforma el DataFrame en una sola operación.
        
        Parámetros:
          df (pd.DataFrame): DataFrame de entrada.
        
        Retorna:
          pd.DataFrame: DataFrame transformado.
        """
        self.fit(df)
        return self.transform(df)

class TrainTestSplit:
    """
    Clase para dividir un DataFrame en conjuntos de entrenamiento y validación.
    
    Parámetros:
      - test_size (float): Fracción de datos a utilizar para el conjunto de validación (entre 0 y 1).
      - random_state (int): Semilla para la generación de números aleatorios (opcional).
      - shuffle (bool): Indica si se debe barajar el DataFrame antes de dividir (default True).
    """
    
    def __init__(self, test_size=0.2, random_state=None, shuffle=True):
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        
    def split(self, df):
        """
        Divide el DataFrame en conjuntos de entrenamiento y validación.
        
        Parámetros:
          - df (pd.DataFrame): DataFrame de entrada.
        
        Retorna:
          - df_train (pd.DataFrame): Conjunto de entrenamiento.
          - df_test (pd.DataFrame): Conjunto de validación.
        """
        if self.shuffle:
            df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        n_samples = len(df)
        n_test = int(n_samples * self.test_size)
        
        df_test = df.iloc[:n_test]
        df_train = df.iloc[n_test:]
        
        return df_train, df_test




