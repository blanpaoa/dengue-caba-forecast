"""
src/utils/metrics.py
Métricas de evaluación del modelo por comuna y horizonte temporal.

Incluye métricas de:
- Regresión: MAE, RMSE, MAPE, R²
- Clasificación de brotes: Precisión, Recall, AUC-ROC
- Métricas de negocio: tasa de detección temprana, tiempo de anticipación
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE robusto a valores cero."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> Dict[str, float]:
    """
    Calcula métricas de regresión estándar.
    
    Returns
    -------
    dict con MAE, RMSE, MAPE, R²
    """
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    
    if label:
        logger.info("Métricas [%s]: MAE=%.2f | RMSE=%.2f | MAPE=%.1f%% | R²=%.3f",
                    label, metrics["MAE"], metrics["RMSE"], metrics["MAPE"], metrics["R2"])
    return metrics


def compute_outbreak_metrics(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    label: str = "",
) -> Dict[str, float]:
    """
    Calcula métricas de clasificación para detección de brotes.
    
    Parameters
    ----------
    y_true_binary : 1 si es brote, 0 si no
    y_pred_binary : predicción binaria
    y_pred_proba : probabilidad predicha (para AUC-ROC)
    
    Returns
    -------
    dict con Precisión, Recall, F1, AUC-ROC
    """
    metrics = {
        "Precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "Recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "F1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
    }
    
    if y_pred_proba is not None and len(np.unique(y_true_binary)) > 1:
        metrics["AUC_ROC"] = roc_auc_score(y_true_binary, y_pred_proba)
    
    if label:
        logger.info(
            "Clasificación [%s]: Precisión=%.2f | Recall=%.2f | F1=%.2f",
            label, metrics["Precision"], metrics["Recall"], metrics["F1"]
        )
    return metrics


def evaluate_by_comuna(
    df_results: pd.DataFrame,
    true_col: str = "y_true",
    pred_col: str = "y_pred",
    comuna_col: str = "comuna_id",
    outbreak_threshold: int = 5,
) -> pd.DataFrame:
    """
    Evalúa el modelo desagregado por cada una de las 15 comunas de CABA.
    
    Criterio del plan de proyecto (HU7): calcular métricas por comuna
    para identificar donde el modelo tiene peor performance.
    
    Parameters
    ----------
    df_results : DataFrame con predicciones y valores reales
    outbreak_threshold : umbral de casos/semana para clasificar como brote
    
    Returns
    -------
    DataFrame con métricas por comuna (15 filas)
    """
    results = []
    
    for comuna in sorted(df_results[comuna_col].unique()):
        df_c = df_results[df_results[comuna_col] == comuna]
        y_true = df_c[true_col].values
        y_pred = df_c[pred_col].values
        
        # Métricas de regresión
        reg_metrics = compute_regression_metrics(y_true, y_pred)
        
        # Métricas de clasificación de brotes
        y_true_bin = (y_true >= outbreak_threshold).astype(int)
        y_pred_bin = (y_pred >= outbreak_threshold).astype(int)
        clf_metrics = compute_outbreak_metrics(y_true_bin, y_pred_bin)
        
        row = {"comuna_id": comuna, "n_observations": len(df_c)}
        row.update(reg_metrics)
        row.update(clf_metrics)
        results.append(row)
    
    df_metrics = pd.DataFrame(results)
    
    logger.info(
        "Evaluación por comuna completada. MAE promedio: %.2f (min: %.2f, max: %.2f)",
        df_metrics["MAE"].mean(),
        df_metrics["MAE"].min(),
        df_metrics["MAE"].max(),
    )
    return df_metrics


def evaluate_by_horizon(
    df_results: pd.DataFrame,
    true_col: str = "y_true",
    pred_col_template: str = "y_pred_h{h}",
    horizons: List[int] = [1, 2, 3, 4],
) -> pd.DataFrame:
    """
    Evalúa el modelo para cada horizonte de predicción (1 a 4 semanas).
    
    Returns
    -------
    DataFrame con métricas por horizonte (4 filas)
    """
    results = []
    
    for h in horizons:
        pred_col = pred_col_template.format(h=h)
        if pred_col not in df_results.columns:
            logger.warning("Columna no encontrada: %s", pred_col)
            continue
        
        y_true = df_results[true_col].values
        y_pred = df_results[pred_col].values
        
        metrics = compute_regression_metrics(y_true, y_pred, label=f"Horizonte {h}w")
        metrics["horizon_weeks"] = h
        results.append(metrics)
    
    return pd.DataFrame(results).set_index("horizon_weeks")


def detection_rate(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
) -> float:
    """
    Tasa de detección temprana de brotes.
    Métrica de negocio del plan de proyecto: objetivo > 80%.
    
    Equivalente al Recall para la clase 'brote'.
    """
    return recall_score(y_true_binary, y_pred_binary, zero_division=0)


def compute_full_evaluation(
    df_results: pd.DataFrame,
    config: Optional[dict] = None,
) -> Dict:
    """
    Evaluación completa del modelo: por comuna, por horizonte y global.
    
    Returns
    -------
    dict con métricas globales, por comuna y por horizonte.
    """
    threshold = 5
    if config:
        threshold = config.get("prediction", {}).get("classification_threshold", 5)
    
    # Global
    global_metrics = compute_regression_metrics(
        df_results["y_true"].values,
        df_results["y_pred"].values,
        label="Global"
    )
    
    # Por brote (global)
    y_true_bin = (df_results["y_true"].values >= threshold).astype(int)
    y_pred_bin = (df_results["y_pred"].values >= threshold).astype(int)
    outbreak_metrics = compute_outbreak_metrics(y_true_bin, y_pred_bin, label="Global brotes")
    
    # Métricas de negocio
    det_rate = detection_rate(y_true_bin, y_pred_bin)
    logger.info(
        "Tasa de detección de brotes: %.1f%% (objetivo: >80%%)",
        det_rate * 100
    )
    
    # Por comuna
    by_comuna = evaluate_by_comuna(df_results, outbreak_threshold=threshold)
    
    return {
        "global_regression": global_metrics,
        "global_outbreak": outbreak_metrics,
        "detection_rate": det_rate,
        "by_comuna": by_comuna,
    }
