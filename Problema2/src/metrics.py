import numpy as np

import numpy as np
import pandas as pd
from collections import Counter

# --------------------------
# 1. Matriz de Confusión
# --------------------------
def compute_confusion_matrix(y_true, y_pred, classes):
    """
    Calcula la matriz de confusión para clasificación multiclase.
    
    Parámetros:
      - y_true: vector de etiquetas verdaderas.
      - y_pred: vector de etiquetas predichas.
      - classes: lista de clases (por ejemplo, [1, 2, 3]).
    
    Retorna:
      - matriz: array de dimensión (n_clases, n_clases) donde la fila i corresponde a las verdaderas etiquetas de la clase i
                y la columna j a las predichas como la clase j.
    """
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = classes.index(t)
        j = classes.index(p)
        matrix[i, j] += 1
    return matrix

# --------------------------
# 2. Accuracy
# --------------------------
def compute_accuracy(y_true, y_pred):
    """
    Calcula el accuracy (precisión global).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

# --------------------------
# 3. Precision, Recall y F-Score (por clase y macro)
# --------------------------
def compute_precision_recall_fscore(y_true, y_pred, classes):
    """
    Calcula precision, recall y f-score por clase, así como los promedios macro.
    
    Retorna:
      - precisions: diccionario {clase: precision}
      - recalls: diccionario {clase: recall}
      - f_scores: diccionario {clase: f-score}
      - macro_precision, macro_recall, macro_fscore: promedios (aritméticos) sobre las clases.
    """
    cm = compute_confusion_matrix(y_true, y_pred, classes)
    precisions = {}
    recalls = {}
    f_scores = {}
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precisions[cls] = precision
        recalls[cls] = recall
        f_scores[cls] = f_score
    macro_precision = np.mean(list(precisions.values()))
    macro_recall = np.mean(list(recalls.values()))
    macro_fscore = np.mean(list(f_scores.values()))
    return precisions, recalls, f_scores, macro_precision, macro_recall, macro_fscore

# --------------------------
# 4. Curva Precision-Recall (para un problema binario)
# --------------------------
def compute_precision_recall_curve_binary(y_true, y_score):
    """
    Calcula la curva Precision-Recall para un problema binario.
    
    Parámetros:
      - y_true: vector binario (0/1) de etiquetas verdaderas.
      - y_score: scores continuos para la clase positiva.
    
    Retorna:
      - precisions: array de precisiones.
      - recalls: array de recalls.
      - thresholds: array de thresholds evaluados.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    thresholds = np.sort(np.unique(y_score))[::-1]
    precisions = []
    recalls = []
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls), thresholds

# --------------------------
# 5. Curva ROC (para un problema binario)
# --------------------------
def compute_roc_curve_binary(y_true, y_score):
    """
    Calcula la curva ROC para un problema binario.
    
    Retorna:
      - fpr: tasa de falsos positivos.
      - tpr: tasa de verdaderos positivos.
      - thresholds: array de thresholds.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    thresholds = np.sort(np.unique(y_score))[::-1]
    fpr = []
    tpr = []
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        tpr_val = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    return np.array(fpr), np.array(tpr), thresholds

# --------------------------
# 6. AUC (utilizando la regla del trapecio)
# --------------------------
def compute_auc(x, y):
    """
    Calcula el AUC utilizando la regla del trapecio.
    Se asume que x está ordenado de forma ascendente.
    """
    return np.trapz(y, x)

# Funciones específicas para AUC-ROC y AUC-PR, basadas en las curvas calculadas.
def compute_auc_roc(fpr, tpr):
    return compute_auc(fpr, tpr)

def compute_auc_pr(precisions, recalls):
    # Nota: En la curva PR, típicamente la integral se calcula respecto a recall.
    # Es importante ordenar por recall (ascendente)
    order = np.argsort(recalls)
    sorted_recalls = recalls[order]
    sorted_precisions = precisions[order]
    return compute_auc(sorted_recalls, sorted_precisions)

# --------------------------
# 7. Funciones para problemas Multiclase (One-vs-Rest)
# --------------------------
def compute_multiclass_roc_pr(y_true, y_scores, classes):
    """
    Para cada clase, calcula la curva ROC y la curva Precision-Recall utilizando el enfoque one-vs-rest.
    
    Parámetros:
      - y_true: vector de etiquetas verdaderas (forma (n_samples,)).
      - y_scores: array de scores o probabilidades.
                  Se espera que tenga forma (n_samples, n_clases). Si tiene forma (n_clases, n_samples), se transpone.
      - classes: lista ordenada de clases, e.g. [1, 2, 3].
      
    Retorna:
      - results: diccionario donde cada clave es una clase y el valor es otro diccionario con:
            'fpr', 'tpr', 'roc_thresholds', 'auc_roc',
            'precisions', 'recalls', 'pr_thresholds', 'auc_pr'
    """
    import numpy as np
    y_true = np.array(y_true)
    # Forzar la conversión a arreglo de NumPy
    y_scores = np.array(y_scores)
    
    # Comprobar la forma de y_scores
    # Se espera que el número de filas sea igual a la cantidad de muestras.
    if y_scores.shape[0] != len(y_true):
        # Si no coincide, asumimos que está transpuesta
        y_scores = y_scores.T
        
    # Comprobación final
    if y_scores.shape[0] != len(y_true):
        raise ValueError(f"Después de la transposición, se esperaba que y_scores tuviera {len(y_true)} filas, pero tiene {y_scores.shape[0]}")
    
    results = {}
    for i, cls in enumerate(classes):
        # Convertir el problema a binario: 1 si la muestra pertenece a la clase cls, 0 en caso contrario.
        y_true_bin = (y_true == cls).astype(int)
        # Extraer la columna correspondiente: se espera que y_scores[:, i] sea un vector de tamaño (n_samples,)
        y_score_cls = y_scores[:, i]
        if y_score_cls.shape[0] != len(y_true):
            raise ValueError(f"Para la clase {cls}, se esperaba un vector de tamaño ({len(y_true)},) pero se obtuvo {y_score_cls.shape}")
        
        # Calcular la curva ROC para la clase (usando la función para problemas binarios)
        fpr, tpr, roc_thresholds = compute_roc_curve_binary(y_true_bin, y_score_cls)
        auc_roc = compute_auc_roc(fpr, tpr)
        
        # Calcular la curva Precision-Recall para la clase
        precisions, recalls, pr_thresholds = compute_precision_recall_curve_binary(y_true_bin, y_score_cls)
        auc_pr = compute_auc_pr(precisions, recalls)
        
        results[cls] = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'auc_roc': auc_roc,
            'precisions': precisions,
            'recalls': recalls,
            'pr_thresholds': pr_thresholds,
            'auc_pr': auc_pr
        }
    return results
