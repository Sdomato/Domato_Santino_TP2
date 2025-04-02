import numpy as np

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


