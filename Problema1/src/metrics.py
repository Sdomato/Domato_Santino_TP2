import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f1_score_macro(y_true, y_pred):
    """
    Calcula el F1 Score macro (promedio de F1 por cada clase) para clasificación multiclase.
    """
    classes = np.unique(np.concatenate((y_true, y_pred)))
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)

# ---------------------------
# Funciones para calcular métricas

def compute_confusion_elements(y_true, y_pred):
    """
    Calcula los elementos de la matriz de confusión para un problema binario.
    Retorna: TP, FP, TN, FN
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, FP, TN, FN

def compute_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def compute_precision(y_true, y_pred):
    TP, FP, TN, FN = compute_confusion_elements(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def compute_recall(y_true, y_pred):
    TP, FP, TN, FN = compute_confusion_elements(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def compute_f1(y_true, y_pred):
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

# Función para calcular la curva ROC manualmente
def compute_roc_curve(y_true, y_scores):
    """
    Calcula manualmente la curva ROC.
    Devuelve: array de FPR, array de TPR y los thresholds evaluados.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Obtener todos los thresholds únicos, ordenados de mayor a menor
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr_list = []
    fpr_list = []
    
    for thresh in thresholds:
        # Predicción: 1 si score >= threshold, 0 en caso contrario
        y_pred = (y_scores >= thresh).astype(int)
        TP, FP, TN, FN = compute_confusion_elements(y_true, y_pred)
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensibilidad
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return np.array(fpr_list), np.array(tpr_list), thresholds

# Función para calcular el área bajo la curva (AUC) usando la regla del trapecio
def compute_auc(x, y):
    # Aseguramos que x esté ordenado de menor a mayor
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    return np.trapz(y_sorted, x_sorted)

# Función para calcular la curva Precision-Recall manualmente
def compute_precision_recall_curve(y_true, y_scores):
    """
    Calcula manualmente la curva Precision-Recall.
    Devuelve: array de precisiones, array de recalls y los thresholds evaluados.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    thresholds = np.sort(np.unique(y_scores))[::-1]
    precision_list = []
    recall_list = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        TP, FP, TN, FN = compute_confusion_elements(y_true, y_pred)
        prec = TP / (TP + FP) if (TP + FP) > 0 else 1  # Si no se predice nada, definimos precisión = 1
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision_list.append(prec)
        recall_list.append(rec)
        
    return np.array(precision_list), np.array(recall_list), thresholds