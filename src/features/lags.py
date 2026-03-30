"""
src/features/lags.py
Construcción de variables rezagadas temporales y promedios climáticos.

Basado en la metodología de:
- Sebastianelli et al. (2024): features temporales con lags 1-4 semanas
- HU5 del plan de proyecto: creación de features de rezago temporal
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


def build_epidemiological_lags(
    df: pd.DataFrame,
    target_col: str = "confirmed_cases",
    group_col: str = "comuna_id",
    lag_weeks: List[int] = [1, 2, 3, 4],
) -> pd.DataFrame:
    """
    Crea variables rezagadas de casos de dengue.
    
    Por cada lag k, genera la variable:
        cases_lag_k = casos de dengue k semanas antes
    
    Importante: los lags se calculan dentro de cada comuna
    para no mezclar información entre unidades espaciales.
    
    Parameters
    ----------
    df : DataFrame con columnas [group_col, 'epi_week', target_col]
    target_col : columna de casos a rezagar
    group_col : columna de agrupación espacial (comuna)
    lag_weeks : lista de semanas de rezago
    
    Returns
    -------
    DataFrame original con columnas adicionales: cases_lag_1, ..., cases_lag_4
    """
    df = df.copy().sort_values([group_col, "year", "epi_week"])
    
    for lag in lag_weeks:
        col_name = f"cases_lag_{lag}w"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        logger.debug("Creado feature: %s", col_name)
    
    logger.info(
        "Features de rezago epidemiológico creados: %s",
        [f"cases_lag_{k}w" for k in lag_weeks]
    )
    return df


def build_climate_rolling_averages(
    df: pd.DataFrame,
    climate_cols: List[str],
    group_col: str = "comuna_id",
    windows_days: List[int] = [7, 14, 21],
) -> pd.DataFrame:
    """
    Calcula promedios móviles de variables climáticas.
    
    Captura el efecto retardado del clima sobre el ciclo del mosquito:
    - 7 días: condiciones de la semana inmediata anterior
    - 14 días: promedio de 2 semanas (ciclo larval del Aedes)
    - 21 días: promedio de 3 semanas (ciclo completo mosquito)
    
    Parameters
    ----------
    df : DataFrame con columnas climáticas diarias ya agregadas a semanal
    climate_cols : lista de columnas climáticas a promediar
    windows_days : ventanas en días para los promedios móviles
    
    Returns
    -------
    DataFrame con columnas adicionales: {col}_{n}d_avg
    """
    df = df.copy().sort_values([group_col, "year", "epi_week"])
    
    # Convertir ventanas de días a semanas (approx)
    windows_weeks = {d: max(1, d // 7) for d in windows_days}
    
    for col in climate_cols:
        if col not in df.columns:
            logger.warning("Columna climática no encontrada: %s", col)
            continue
        
        for days, weeks in windows_weeks.items():
            col_name = f"{col}_{days}d_avg"
            df[col_name] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.shift(1).rolling(window=weeks, min_periods=1).mean())
            )
            logger.debug("Creado feature: %s", col_name)
    
    logger.info("Features de promedios climáticos creados para: %s", climate_cols)
    return df


def build_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables de estacionalidad y temporada epidémica.
    
    Variables generadas:
    - week_sin, week_cos: codificación cíclica de la semana del año
    - month, season: mes y estación del año
    - is_epidemic_season: indicador temporada alta (dic-abr)
    - weeks_to_peak: semanas hasta el pico histórico (semana 8 aprox.)
    
    Returns
    -------
    DataFrame con columnas de estacionalidad añadidas.
    """
    df = df.copy()
    
    # Codificación cíclica de la semana (preserva periodicidad)
    df["week_sin"] = np.sin(2 * np.pi * df["epi_week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["epi_week"] / 52)
    
    # Mes aproximado desde semana epidemiológica
    df["month"] = ((df["epi_week"] - 1) // 4 + 1).clip(1, 12)
    
    # Estación del año (hemisferio sur)
    df["season"] = df["month"].map({
        12: "summer", 1: "summer", 2: "summer",
        3: "autumn", 4: "autumn", 5: "autumn",
        6: "winter", 7: "winter", 8: "winter",
        9: "spring", 10: "spring", 11: "spring",
    })
    df["season"] = pd.Categorical(
        df["season"], categories=["summer", "autumn", "winter", "spring"]
    )
    
    # Temporada epidémica de dengue (diciembre-abril = semanas 1-17 y 48-52)
    epidemic_weeks = list(range(1, 18)) + list(range(48, 53))
    df["is_epidemic_season"] = df["epi_week"].isin(epidemic_weeks).astype(int)
    
    # Años epidémicos previos (feature para capturar ciclos inter-epidémicos)
    # TODO: calcular desde datos históricos reales
    
    logger.info("Features de estacionalidad creados")
    return df


def build_climate_anomalies(
    df: pd.DataFrame,
    climate_cols: List[str],
    group_col: str = "comuna_id",
) -> pd.DataFrame:
    """
    Calcula anomalías climáticas respecto a normales históricas.
    
    Anomalía = valor_observado - media_histórica_misma_semana
    
    Captura condiciones atípicas que pueden potenciar brotes.
    
    Returns
    -------
    DataFrame con columnas adicionales: {col}_anomaly
    """
    df = df.copy()
    
    # Calcular medias históricas por semana del año
    historical_means = (
        df.groupby(["epi_week", group_col])[climate_cols]
        .mean()
        .reset_index()
        .rename(columns={col: f"{col}_hist_mean" for col in climate_cols})
    )
    
    df = df.merge(historical_means, on=["epi_week", group_col], how="left")
    
    for col in climate_cols:
        if col in df.columns:
            df[f"{col}_anomaly"] = df[col] - df[f"{col}_hist_mean"]
            df = df.drop(columns=[f"{col}_hist_mean"])
    
    logger.info("Features de anomalías climáticas creados")
    return df


def run_feature_engineering(
    df_epidemio: pd.DataFrame,
    df_climate: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    
    Combina features temporales, climáticos y de estacionalidad
    siguiendo la metodología de HU5 del plan de proyecto.
    
    Returns
    -------
    DataFrame listo para modelado con todas las features.
    """
    logger.info("=== Iniciando Feature Engineering ===")
    
    # Defaults si no se pasa config
    lag_weeks = [1, 2, 3, 4]
    climate_cols = ["temp_min", "temp_mean", "temp_max", "humidity", "precipitation"]
    
    if config:
        lag_weeks = config.get("features", {}).get("lags", {}).get("epidemiological", [1, 2, 3, 4])
        climate_cols_config = config.get("climate_variables", [])
        climate_cols = [v["name"] for v in climate_cols_config] if climate_cols_config else climate_cols
    
    # 1. Lags epidemiológicos
    df = build_epidemiological_lags(df_epidemio, lag_weeks=lag_weeks)
    
    # 2. Merge con datos climáticos
    if not df_climate.empty:
        df = df.merge(df_climate, on=["year", "epi_week", "comuna_id"], how="left")
    
    # 3. Promedios climáticos móviles
    available_climate = [c for c in climate_cols if c in df.columns]
    if available_climate:
        df = build_climate_rolling_averages(df, available_climate)
        df = build_climate_anomalies(df, available_climate)
    
    # 4. Estacionalidad
    df = build_seasonality_features(df)
    
    # Reporte de features generados
    feature_cols = [c for c in df.columns if any(
        tag in c for tag in ["lag", "avg", "anomaly", "sin", "cos", "season", "epidemic"]
    )]
    logger.info("Total de features generados: %d", len(feature_cols))
    logger.info("Features: %s", feature_cols)
    
    return df
