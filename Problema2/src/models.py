import numpy as np
import pandas as pd


class LogisticRegressionMulticlass:
    """
    Implementa una regresión logística multiclase con regularización L2.
    
    Parámetros:
      - learning_rate: tasa de aprendizaje para el descenso de gradiente.
      - n_iters: número de iteraciones para el ajuste.
      - reg_lambda: parámetro de regularización L2.
      - verbose: si es True, imprime el costo cada 100 iteraciones.
    """
    
    def __init__(self, learning_rate=0.01, n_iters=1000, reg_lambda=0.1, verbose=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.theta = None  # Coeficientes (incluye sesgo en la primera fila)
        self.classes_ = None

    def _softmax(self, z):
        """
        Calcula la función softmax de la matriz z de forma numéricamente estable.
        z: matriz de forma (m, k) donde m es el número de muestras y k el número de clases.
        Retorna: matriz de probabilidades de forma (m, k)
        """
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot(self, y):
        """
        Convierte un vector de etiquetas (con valores 0, 1, ..., k-1) en una matriz one-hot.
        """
        m = y.shape[0]
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        Y = np.zeros((m, k))
        for idx, cls in enumerate(self.classes_):
            Y[y == cls, idx] = 1
        return Y

    def _cost_function(self, X, Y):
        """
        Calcula el costo (función de pérdida) con regularización L2.
        X: matriz de características con sesgo (forma: m x (n+1))
        Y: matriz one-hot de etiquetas (forma: m x k)
        """
        m = X.shape[0]
        z = X.dot(self.theta)
        h = self._softmax(z)
        # Evitar log(0) sumando una constante pequeña
        cost = -np.sum(Y * np.log(h + 1e-15)) / m
        # Regularización: no se aplica sobre el sesgo (primera fila de theta)
        reg_term = (self.reg_lambda / (2 * m)) * np.sum(self.theta[1:,:] ** 2)
        return cost + reg_term

    def fit(self, X, y):
        """
        Ajusta el modelo a los datos.
        
        Parámetros:
          - X: matriz de características de forma (m, n)
          - y: vector de etiquetas de forma (m,)
        """
        m, n = X.shape
        # Convertir y a formato one-hot
        Y = self._one_hot(y)
        k = Y.shape[1]
        # Agregar columna de 1's para el sesgo
        X_bias = np.hstack([np.ones((m, 1)), X])
        # Inicializar theta (n+1 x k) en ceros
        self.theta = np.zeros((n + 1, k))
        
        # Descenso de gradiente
        for i in range(self.n_iters):
            z = X_bias.dot(self.theta)
            h = self._softmax(z)
            error = h - Y  # (m, k)
            grad = (X_bias.T.dot(error)) / m
            # Regularización: no se regulariza la primera fila (sesgo)
            reg = (self.reg_lambda / m) * np.vstack([np.zeros((1, k)), self.theta[1:,:]])
            grad += reg
            self.theta -= self.learning_rate * grad
            
            if self.verbose and i % 100 == 0:
                cost = self._cost_function(X_bias, Y)
                print(f"Iteración {i}, costo: {cost:.6f}")
        return self

    def predict_proba(self, X):
        """
        Retorna las probabilidades predichas para cada clase.
        
        Parámetros:
          - X: matriz de características de forma (m, n)
        
        Retorna:
          matriz de probabilidades de forma (m, k)
        """
        m = X.shape[0]
        X_bias = np.hstack([np.ones((m, 1)), X])
        z = X_bias.dot(self.theta)
        return self._softmax(z)
    
    def predict(self, X):
        """
        Retorna la clase predicha para cada muestra.
        
        Parámetros:
          - X: matriz de características de forma (m, n)
        
        Retorna:
          vector de etiquetas predichas de forma (m,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class LDA:
    def __init__(self):
        self.classes = None   # Clases presentes en los datos
        self.means = {}       # Medias de cada clase
        self.priors = {}      # Priori de cada clase
        self.cov_inv = None   # Inversa de la matriz de covarianza común
        self.W = {}           # Parámetro W para cada clase (Σ^{-1} * μ_c)
        self.b = {}           # Bias para cada clase

    def fit(self, X, y):
        """
        Ajusta el modelo LDA a los datos de entrenamiento.
        
        Parámetros:
          - X: matriz de características (n_samples x n_features)
          - y: vector de etiquetas (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Calcular medias y priors para cada clase
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / n_samples
        
        # Calcular la matriz de covarianza común (ponderada por n_c - 1)
        pooled_cov = np.zeros((n_features, n_features))
        for c in self.classes:
            X_c = X[y == c]
            # np.cov con rowvar=False calcula la matriz de covarianza de forma correcta
            cov_c = np.cov(X_c, rowvar=False, bias=False)
            pooled_cov += (X_c.shape[0] - 1) * cov_c
        pooled_cov /= (n_samples - len(self.classes))
        
        # Calcular la inversa de la matriz de covarianza
        self.cov_inv = np.linalg.pinv(pooled_cov)

        
        # Calcular los parámetros de la función discriminante para cada clase
        for c in self.classes:
            mean_c = self.means[c]
            W_c = self.cov_inv.dot(mean_c)
            b_c = -0.5 * mean_c.dot(self.cov_inv).dot(mean_c) + np.log(self.priors[c])
            self.W[c] = W_c
            self.b[c] = b_c

    def predict(self, X):
        """
        Predice la clase para cada muestra en X.
        
        Parámetros:
          - X: matriz de características (n_samples x n_features)
          
        Retorna:
          - Array con las predicciones para cada muestra.
        """
        X = np.array(X)
        preds = []
        for x in X:
            # Calcular la función discriminante para cada clase
            scores = {c: x.dot(self.W[c]) + self.b[c] for c in self.classes}
            # Asignar la clase con mayor puntaje
            pred_class = max(scores, key=scores.get)
            preds.append(pred_class)
        return np.array(preds)

    def predict_proba(self, X):
        """
        Calcula las probabilidades predichas para cada clase utilizando la función softmax.
        
        Parámetros:
          - X: matriz de características (n_samples x n_features)
          
        Retorna:
          - Array (n_samples x n_classes) con las probabilidades para cada clase.
        """
        X = np.array(X)
        proba = []
        for x in X:
            scores = [x.dot(self.W[c]) + self.b[c] for c in self.classes]
            # Aplicar softmax
            exp_scores = np.exp(scores)
            prob = exp_scores / np.sum(exp_scores)
            proba.append(prob)
        return np.array(proba)

import numpy as np
from collections import Counter
import math
import random

def entropy(y):
    """Calcula la entropía de un vector de etiquetas."""
    counts = np.bincount(y)
    probabilities = counts[np.nonzero(counts)] / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, y_left, y_right):
    """Calcula la ganancia de información obtenida al dividir el vector y en y_left y y_right."""
    n = len(y)
    gain = entropy(y)
    n_left = len(y_left)
    n_right = len(y_right)
    if n_left > 0:
        gain -= (n_left/n) * entropy(y_left)
    if n_right > 0:
        gain -= (n_right/n) * entropy(y_right)
    return gain


import numpy as np
import pandas as pd
from collections import Counter

# -----------------------------
# Árbol de Decisión (DecisionTree)
# -----------------------------
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
    
    def fit(self, X, y):
        # Convertir X a numpy array de tipo float para asegurar cálculos numéricos
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        
        # Caso base: si se alcanza la profundidad máxima, todas las etiquetas son iguales o hay pocas muestras
        if (depth >= self.max_depth) or (num_labels == 1) or (num_samples < self.min_samples_split):
            return {'type': 'leaf', 'value': self._most_common_label(y)}
        
        # Seleccionar un subconjunto aleatorio de features
        feature_indices = np.random.choice(num_features, self.max_features, replace=False)
        best_gain = -1
        best_split = None
        
        for feature in feature_indices:
            col = X[:, feature]
            thresholds = np.unique(col)
            for thresh in thresholds:
                left_indices = col <= thresh
                right_indices = col > thresh
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                y_left = y[left_indices]
                y_right = y[right_indices]
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature,
                        'threshold': float(thresh),
                        'X_left': X[left_indices],
                        'y_left': y[left_indices],
                        'X_right': X[right_indices],
                        'y_right': y[right_indices]
                    }
        if best_split is None:
            return {'type': 'leaf', 'value': self._most_common_label(y)}
        
        left_tree = self._build_tree(best_split['X_left'], best_split['y_left'], depth+1)
        right_tree = self._build_tree(best_split['X_right'], best_split['y_right'], depth+1)
        return {
            'type': 'node',
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }
    
    def _information_gain(self, y, y_left, y_right):
        gain = self._entropy(y)
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        if n_left > 0:
            gain -= (n_left/n) * self._entropy(y_left)
        if n_right > 0:
            gain -= (n_right/n) * self._entropy(y_right)
        return gain
    
    def _entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _traverse_tree(self, x, node):
        if node['type'] == 'leaf':
            return node['value']
        # Forzamos la conversión de la feature a float (por seguridad)
        feature_val = float(x[node['feature']])
        if feature_val <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])
    
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = [self._traverse_tree(x, self.tree) for x in X]
        return np.array(predictions)

# -----------------------------
# Random Forest para Clasificación Multiclase
# -----------------------------
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features=None):
        """
        Parámetros:
          - n_estimators: cantidad de árboles en el bosque.
          - max_depth: profundidad máxima de cada árbol.
          - min_samples_split: número mínimo de muestras para dividir un nodo.
          - max_features: número de características a evaluar en cada división.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Muestreo bootstrap: seleccionar muestras con reemplazo
            indices = np.random.choice(n_samples, n_samples, replace=True)
            # Usar .iloc si X es DataFrame; si no, indexar directamente
            if isinstance(X, pd.DataFrame):
                X_sample = X.iloc[indices]
            else:
                X_sample = X[indices]
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_sample = y.iloc[indices]
            else:
                y_sample = y[indices]
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        """
        Realiza la predicción para cada muestra combinando las predicciones de cada árbol mediante votación mayoritaria.
        """
        # Obtener las predicciones de cada árbol (cada árbol devuelve un vector de predicciones)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # all_preds tendrá forma (n_estimators, n_samples); la transponemos para iterar por muestra.
        predictions = []
        for sample_preds in all_preds.T:
            vote = Counter(sample_preds).most_common(1)[0][0]
            predictions.append(vote)
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Retorna la probabilidad predicha para cada clase, calculada como la frecuencia de predicción en el bosque.
        Retorna:
          - proba: array de forma (n_samples, n_classes)
          - classes: arreglo con las clases únicas.
        """
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        n_samples = all_preds.shape[1]
        classes = np.unique(all_preds)
        proba = np.zeros((n_samples, len(classes)))
        for i in range(n_samples):
            counter = Counter(all_preds[:, i])
            for j, cls in enumerate(classes):
                proba[i, j] = counter[cls] / self.n_estimators
        return proba, classes
