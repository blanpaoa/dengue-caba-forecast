"""
src/data/climate_ingestion.py
==============================
Descarga y procesamiento de datos climáticos para CABA
usando la API de Open-Meteo (reanálisis ERA5).

API utilizada:
    Open-Meteo Historical Weather API
    https://open-meteo.com/en/docs/historical-weather-api
    - Gratuita, sin registro, sin límite de uso académico
    - Datos de reanálisis ERA5 (calidad científica)
    - Cobertura: 1940 hasta ayer
    - Sin gaps (modelo matemático, no estación física)

Estación de referencia para CABA:
    Observatorio Central Buenos Aires
    Latitud: -34.58° | Longitud: -58.48°

Variables descargadas (diarias):
    - temperature_2m_max    : temperatura máxima (°C)
    - temperature_2m_min    : temperatura mínima (°C)
    - temperature_2m_mean   : temperatura media (°C)
    - precipitation_sum     : precipitación acumulada (mm)
    - relative_humidity_2m_max : humedad relativa máxima (%)
    - relative_humidity_2m_min : humedad relativa mínima (%)

Pipeline:
    Paso 1 → Descargar datos diarios de la API
    Paso 2 → Limpiar y validar rangos
    Paso 3 → Calcular humedad media y variables derivadas
    Paso 4 → Agregar de diario a semanal (semana epidemiológica)
    Paso 5 → Calcular índice de calor
    Paso 6 → Calcular anomalías respecto a normales históricas
    Paso 7 → Guardar en data/processed/
"""

import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Coordenadas del Observatorio Central Buenos Aires
# Es la estación meteorológica de referencia histórica para CABA
CABA_LAT = -34.58
CABA_LON = -58.48

# Período que cubre nuestros datos epidemiológicos
START_DATE = "2022-01-01"  # un año antes para tener lags disponibles desde 2023
END_DATE   = "2025-12-31"

# URL base de la API de Open-Meteo (reanálisis ERA5)
OPEN_METEO_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Variables a descargar
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
]

# Rangos válidos para validación (config.yaml)

    #la temperatura mínima absoluta registrada en Buenos Aires fue de -5.4°C en 1918 
    # y El límite superior de 45 es porque la máxima histórica registrada fue de 43.3°C en enero 2022 (se dejo un pequeño margen)
    # En la precipitación el límite de 500mm semanales viene de que el récord histórico de lluvia en una semana en CABA está alrededor de 300-350mm
    #500mm es un valor muy extremo si hay datos con ese valor deben ser erroneos.

VALID_RANGES = { 

    "temp_max":  (-5, 45),                           
    "temp_min":  (-5, 45), 
    "temp_mean": (-5, 45),
    "precipitation": (0, 500),
    "humidity_max": (0, 100),
    "humidity_min": (0, 100),
}


# =============================================================================
# PASO 1: DESCARGA DE DATOS CLIMÁTICOS
# La API de Open-Meteo devuelve un JSON con arrays de valores diarios.
# Convertimos ese JSON a un DataFrame de pandas.
# =============================================================================

def descargar_clima(
    lat: float = CABA_LAT,
    lon: float = CABA_LON,
    start: str = START_DATE,
    end: str = END_DATE,
    dest: Path = RAW_DIR / "clima_caba_diario_raw.csv",
) -> pd.DataFrame:
    """
    Descarga datos climáticos diarios de Open-Meteo para CABA.

    Si el archivo ya existe lo carga desde disco para no
    volver a descargarlo en cada ejecución.

    Returns
    -------
    DataFrame con una fila por día y columnas de variables climáticas.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Archivo climático ya existe — cargando desde disco: %s", dest)
        return pd.read_csv(dest, parse_dates=["date"])

    logger.info("--- PASO 1: Descargando datos climáticos de Open-Meteo ---")
    logger.info("  Ubicación: CABA (%.2f°, %.2f°)", lat, lon)
    logger.info("  Período: %s → %s", start, end)

    # Parámetros de la solicitud a la API
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(DAILY_VARIABLES),
        "timezone":   "America/Argentina/Buenos_Aires",
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    response.raise_for_status()

    # El JSON tiene una clave "daily" con arrays de valores
    # y una clave "daily_units" con las unidades de cada variable
    data = response.json()
    daily = data["daily"]

    # Convertimos el JSON a DataFrame
    # Cada clave del dict "daily" se convierte en una columna
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])

    # Renombramos a nombres más cortos y descriptivos
    df = df.rename(columns={
        "temperature_2m_max":      "temp_max",
        "temperature_2m_min":      "temp_min",
        "temperature_2m_mean":     "temp_mean",
        "precipitation_sum":       "precipitation",
        "relative_humidity_2m_max": "humidity_max",
        "relative_humidity_2m_min": "humidity_min",
    })

    logger.info(
        "  Descargados: %d días (%s → %s)",
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
    )

    # Guardamos el raw para no volver a descargar
    df.to_csv(dest, index=False)
    logger.info("  Guardado en: %s", dest)

    return df


# =============================================================================
# PASO 2: LIMPIEZA Y VALIDACIÓN DE RANGOS
# Verificamos que los valores estén dentro de rangos físicamente razonables.
# ERA5 raramente tiene outliers, pero es buena práctica verificar.
# =============================================================================

def limpiar_clima(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida rangos físicos de las variables climáticas.
    Registra outliers pero no los elimina — los marca como NaN
    para que el paso de agregación los maneje correctamente.
    """
    logger.info("--- PASO 2: Validando rangos climáticos ---")
    df = df.copy()

    total_outliers = 0
    for col, (vmin, vmax) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        fuera_rango = ~df[col].between(vmin, vmax) & df[col].notna()
        n = fuera_rango.sum()
        if n > 0:
            logger.warning(
                "  %s: %d valores fuera de rango [%g, %g] → marcados como NaN",
                col, n, vmin, vmax
            )
            df.loc[fuera_rango, col] = np.nan
            total_outliers += n

    if total_outliers == 0:
        logger.info("  Todos los valores están dentro de rangos válidos.")

    # Verificar gaps (no debería haber con ERA5)
    gaps = df["date"].diff().dt.days
    gaps_grandes = (gaps > 1).sum()
    if gaps_grandes > 0:
        logger.warning("  Gaps detectados en la serie: %d", gaps_grandes)
    else:
        logger.info("  Serie continua sin gaps.")

    return df


# =============================================================================
# PASO 3: VARIABLES DERIVADAS
# Calculamos la humedad media y el índice de calor.
# También preparamos la columna de año y semana epidemiológica
# para poder agregar a escala semanal en el Paso 4.
# =============================================================================

def calcular_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables derivadas al DataFrame diario:
    - humidity_mean: promedio de humedad máxima y mínima
    - year: año del registro
    - epi_week: semana epidemiológica (isocalendar)
    - heat_index: índice de calor (combinación de temp y humedad)
    """
    logger.info("--- PASO 3: Calculando variables derivadas ---")
    df = df.copy()

    # Humedad media diaria
    df["humidity_mean"] = (df["humidity_max"] + df["humidity_min"]) / 2

    # Año y semana epidemiológica
    # Usamos isocalendar() que sigue el estándar ISO 8601
    # La semana epidemiológica en Argentina empieza el domingo,
    # pero para este proyecto usamos ISO (empieza el lunes) como aproximación
    df["year"]     = df["date"].dt.isocalendar().year.astype(int)
    df["epi_week"] = df["date"].dt.isocalendar().week.astype(int)

    # Índice de calor (Heat Index)
    # Fórmula de Rothfusz (usada por NOAA) — válida cuando temp > 27°C y humedad > 40%
    # Para valores menores simplifica a la temperatura media
    T = df["temp_mean"]
    H = df["humidity_mean"]

    # Fórmula completa de Rothfusz
    hi = (
        -8.78469475556
        + 1.61139411 * T
        + 2.33854883889 * H
        - 0.14611605 * T * H
        - 0.012308094 * T**2
        - 0.0164248277778 * H**2
        + 0.002211732 * T**2 * H
        + 0.00072546 * T * H**2
        - 0.000003582 * T**2 * H**2
    )

    # Aplicamos la fórmula solo cuando tiene sentido físico
    # Para condiciones moderadas usamos directamente la temperatura media
    df["heat_index"] = np.where(
        (T >= 27) & (H >= 40),
        hi,
        T  # fuera del rango de validez, el índice de calor ≈ temperatura
    )

    logger.info(
        "  Variables derivadas creadas: humidity_mean, heat_index, year, epi_week"
    )
    return df


# =============================================================================
# PASO 4: AGREGACIÓN DE DIARIO A SEMANAL
# El modelo trabaja a escala de semana epidemiológica.
# Agregamos los datos diarios calculando:
#   - temperatura: promedio semanal de max, min y media
#   - precipitación: suma semanal acumulada
#   - humedad: promedio semanal
#   - índice de calor: promedio semanal
# =============================================================================

def agregar_semanal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega datos diarios a semana epidemiológica.
    Una fila por (year, epi_week).
    """
    logger.info("--- PASO 4: Agregando de diario a semanal ---")

    df_weekly = (
        df.groupby(["year", "epi_week"], as_index=False)
        .agg(
            temp_max_mean    = ("temp_max",    "mean"),   # promedio de máximas
            temp_min_mean    = ("temp_min",    "mean"),   # promedio de mínimas
            temp_mean        = ("temp_mean",   "mean"),   # temperatura media
            precipitation    = ("precipitation", "sum"),  # lluvia acumulada
            humidity_mean    = ("humidity_mean", "mean"), # humedad media
            heat_index_mean  = ("heat_index",  "mean"),   # índice de calor
            n_dias           = ("date",        "count"),  # días en la semana
            fecha_inicio     = ("date",        "min"),    # primera fecha de la semana
        )
        .sort_values(["year", "epi_week"])
        .reset_index(drop=True)
    )

    logger.info(
        "  Semanas agregadas: %d | Período: %d SE%02d → %d SE%02d",
        len(df_weekly),
        df_weekly["year"].min(), df_weekly["epi_week"].min(),
        df_weekly["year"].max(), df_weekly["epi_week"].max(),
    )
    return df_weekly


# =============================================================================
# PASO 5: ANOMALÍAS CLIMÁTICAS
# Calculamos cuánto se desvía cada semana respecto a su promedio histórico.
# Una anomalía positiva de temperatura significa que esa semana fue más
# calurosa de lo normal — lo cual puede potenciar los brotes de dengue.
#
# Fórmula: anomalía = valor_observado - media_histórica_misma_semana
# =============================================================================

def calcular_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula anomalías climáticas respecto a la media histórica
    de cada semana del año (usando todos los años disponibles).

    Variables con anomalía: temp_mean, precipitation, humidity_mean
    """
    logger.info("--- PASO 5: Calculando anomalías climáticas ---")
    df = df.copy()

    variables_anomalia = ["temp_mean", "precipitation", "humidity_mean"]

    # Calculamos la media histórica por semana del año
    # usando todos los años disponibles en el dataset
    media_historica = (
        df.groupby("epi_week")[variables_anomalia]
        .mean()
        .reset_index()
        .rename(columns={v: f"{v}_hist_mean" for v in variables_anomalia})
    )

    # Unimos la media histórica al dataset
    df = df.merge(media_historica, on="epi_week", how="left")

    # Calculamos la anomalía como diferencia al promedio histórico
    for var in variables_anomalia:
        df[f"{var}_anomaly"] = df[var] - df[f"{var}_hist_mean"]
        df = df.drop(columns=[f"{var}_hist_mean"])

    logger.info(
        "  Anomalías calculadas para: %s",
        [f"{v}_anomaly" for v in variables_anomalia]
    )
    return df


# =============================================================================
# REPORTE DE CALIDAD
# =============================================================================

def reporte_calidad(df: pd.DataFrame):
    """Imprime resumen del dataset climático procesado."""
    print("\n" + "=" * 55)
    print("  REPORTE DE CALIDAD — clima_caba_semanal")
    print("=" * 55)
    print(f"  Semanas totales:        {len(df)}")
    print(f"  Años cubiertos:         {sorted(df['year'].unique().tolist())}")
    print(f"  Temperatura media:      {df['temp_mean'].mean():.1f}°C")
    print(f"  Temp. máx. registrada:  {df['temp_max_mean'].max():.1f}°C")
    print(f"  Temp. mín. registrada:  {df['temp_min_mean'].min():.1f}°C")
    print(f"  Precipitación media:    {df['precipitation'].mean():.1f} mm/semana")
    print(f"  Precipitación máxima:   {df['precipitation'].max():.1f} mm/semana")
    print(f"  Humedad media:          {df['humidity_mean'].mean():.1f}%")
    print(f"  Índice calor máximo:    {df['heat_index_mean'].max():.1f}°C")
    print(f"  Valores NaN en temp:    {df['temp_mean'].isna().sum()}")
    print(f"  Valores NaN en lluvia:  {df['precipitation'].isna().sum()}")
    print("=" * 55 + "\n")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_climate_ingestion(save: bool = True) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de ingesta climática.

    Paso 1 → Descarga datos diarios de Open-Meteo
    Paso 2 → Limpia y valida rangos
    Paso 3 → Calcula variables derivadas (humedad media, índice de calor)
    Paso 4 → Agrega de diario a semanal (semana epidemiológica)
    Paso 5 → Calcula anomalías respecto a normales históricas
    Paso 6 → Guarda en data/processed/
    """
    print("\n" + "=" * 55)
    print("  SPRINT 1 — Ingesta climática CABA (Open-Meteo ERA5)")
    print("=" * 55 + "\n")

    # Paso 1: descarga
    df_diario = descargar_clima()

    # Paso 2: limpieza
    df_diario = limpiar_clima(df_diario)

    # Paso 3: variables derivadas
    df_diario = calcular_variables_derivadas(df_diario)

    # Paso 4: agregación semanal
    df_semanal = agregar_semanal(df_diario)

    # Paso 5: anomalías
    df_semanal = calcular_anomalias(df_semanal)

    # Reporte
    reporte_calidad(df_semanal)

    # Paso 6: guardar
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        parquet_path = PROCESSED_DIR / "clima_caba_semanal.parquet"
        df_semanal.to_parquet(parquet_path, index=False)
        logger.info("Guardado en: %s", parquet_path)

        csv_path = PROCESSED_DIR / "clima_caba_semanal.csv"
        df_semanal.to_csv(csv_path, index=False)
        logger.info("CSV de inspección: %s", csv_path)

    return df_semanal


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    df = run_climate_ingestion(save=True)

    print("Muestra de semanas de verano (alta temporada dengue):")
    verano = df[df["epi_week"].between(1, 17)]
    print(
        verano[["year", "epi_week", "temp_mean", "precipitation",
                "humidity_mean", "heat_index_mean", "temp_mean_anomaly"]]
        .head(20)
        .to_string(index=False)
    )