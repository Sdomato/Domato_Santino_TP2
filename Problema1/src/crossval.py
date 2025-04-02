import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Problema1.src.metrics import f1_score_macro

def cross_validate_lambda_f1(model_class, X, y, lambdas, k=20, learning_rate=0.1, n_iters=1000, verbose=False):
    """
    Realiza k-fold cross validation para evaluar distintos valores de lambda
    usando el macro F1 Score como métrica.
    
    Parámetros:
      - model_class: la clase del modelo a entrenar (ej. LogisticRegressionMulticlass)
      - X: matriz de características (numpy array)
      - y: vector de etiquetas (numpy array)
      - lambdas: lista de valores de lambda a evaluar.
      - k: número de folds para la validación cruzada.
      - learning_rate: tasa de aprendizaje para el modelo.
      - n_iters: número de iteraciones para el entrenamiento.
      - verbose: si True, imprime información durante el entrenamiento.
    
    Retorna:
      - best_lambda: el valor de lambda con mayor macro F1 promedio.
      - results: diccionario con la macro F1 promedio para cada lambda.
    """
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    fold_size = m // k
    
    # Matriz para almacenar el F1 Score de cada fold para cada lambda
    scores_matrix = np.zeros((k, len(lambdas)))
    results = {}
    
    for l_idx, lam in enumerate(lambdas):
        fold_scores = []
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else m
            val_indices = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))
            
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_val = X.iloc[val_indices]
            y_val = y.iloc[val_indices]
            
            model = model_class(learning_rate=learning_rate, n_iters=n_iters, reg_lambda=lam, verbose=verbose)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score_macro(y_val, y_pred)
            fold_scores.append(f1)
            scores_matrix[i, l_idx] = f1
        
        avg_f1 = np.mean(fold_scores)
        results[lam] = avg_f1
        print(f"Lambda: {lam} - Macro F1 Promedio: {avg_f1:.4f}")
    
    best_lambda = max(results, key=results.get)
    print(f"\nEl mejor lambda es: {best_lambda} con Macro F1 {results[best_lambda]:.4f}")
    
    # Plotear la matriz de folds vs lambdas
    plt.figure(figsize=(10, 7))
    im = plt.imshow(scores_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title("Macro F1 Score por Fold y Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Fold")
    plt.xticks(np.arange(len(lambdas)), lambdas)
    plt.yticks(np.arange(k), np.arange(1, k+1))
    plt.colorbar(im)
    
    # Anotar cada celda con su valor de F1 Score
    thresh = scores_matrix.max() / 2.0
    for i in range(scores_matrix.shape[0]):
        for j in range(scores_matrix.shape[1]):
            plt.text(j, i, f"{scores_matrix[i, j]:.3f}",
                     ha="center", va="center",
                     color="white" if scores_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()
    
    return best_lambda, results