"""
src/data/ingestion.py
Descarga y carga de datos de fuentes públicas argentinas.

Fuentes:
- BEN: Boletín Epidemiológico Nacional (casos dengue por semana)
- BES-CABA: Boletín Epidemiológico Semanal CABA (por comuna)
- SMN: Servicio Meteorológico Nacional (variables climáticas)
- datos.gob.ar: API de vigilancia dengue
"""

import logging
import yaml
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Carga configuración del proyecto."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Datos epidemiológicos
# ---------------------------------------------------------------------------

def load_dengue_datos_gob(raw_dir: Path) -> pd.DataFrame:
    """
    Carga datos de dengue desde datos.gob.ar.
    
    El dataset de Vigilancia Dengue y Zika contiene:
    - semana_epidemiologica
    - anio
    - provincia / departamento
    - casos_confirmados / casos_sospechosos
    
    Returns
    -------
    pd.DataFrame con columnas estandarizadas.
    """
    # TODO Sprint 1 - Tarea 2:
    # Descargar CSV desde:
    # https://datos.gob.ar/dataset/salud-dengue-zika-vigilancia
    # y guardarlo en data/raw/dengue_datos_gob.csv
    
    filepath = raw_dir / "dengue_datos_gob.csv"
    if not filepath.exists():
        logger.warning(
            "Archivo no encontrado: %s\n"
            "Descargar manualmente desde datos.gob.ar y guardar en data/raw/",
            filepath
        )
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info("Cargados %d registros desde datos.gob.ar", len(df))
    return df


def load_ben_weekly(raw_dir: Path) -> pd.DataFrame:
    """
    Carga datos del Boletín Epidemiológico Nacional (BEN).
    Los BEN se publican en PDF; requiere extracción previa.
    
    Returns
    -------
    pd.DataFrame con columnas:
        - year, epi_week, province, confirmed_cases
    """
    # TODO Sprint 1 - Tarea 1:
    # Los BEN están en:
    # https://www.argentina.gob.ar/salud/epidemiologia/boletines
    # Extraer tabla de dengue de cada PDF y consolidar en CSV.
    
    filepath = raw_dir / "ben_dengue_consolidated.csv"
    if not filepath.exists():
        logger.warning("BEN no encontrado. Ver instrucciones en docs/data_sources.md")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    logger.info("Cargados %d registros del BEN", len(df))
    return df


def load_bes_caba(raw_dir: Path) -> pd.DataFrame:
    """
    Carga datos del Boletín Epidemiológico Semanal CABA (BES).
    Granularidad: por comuna y semana epidemiológica.
    
    Returns
    -------
    pd.DataFrame con columnas:
        - year, epi_week, comuna_id (1-15), confirmed_cases
    """
    # TODO Sprint 1 - Tarea 1:
    # Los BES están en:
    # https://www.buenosaires.gob.ar/salud/epidemiologia
    # Extraer tabla de dengue por comuna de cada boletín.
    
    filepath = raw_dir / "bes_caba_dengue.csv"
    if not filepath.exists():
        logger.warning("BES-CABA no encontrado. Ver instrucciones en docs/data_sources.md")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    logger.info("Cargados %d registros del BES-CABA", len(df))
    return df


# ---------------------------------------------------------------------------
# Datos climáticos
# ---------------------------------------------------------------------------

def load_smn_climate(raw_dir: Path) -> pd.DataFrame:
    """
    Carga datos climáticos del Servicio Meteorológico Nacional (SMN).
    
    Variables diarias por estación meteorológica:
    - temp_min, temp_mean, temp_max (°C)
    - humidity (%)
    - precipitation (mm)
    
    Returns
    -------
    pd.DataFrame con columnas estandarizadas.
    """
    # TODO Sprint 1 - Tarea HU2:
    # Descargar datos desde:
    # https://www.smn.gob.ar/descarga-de-datos
    # Estaciones relevantes: AERO BUENOS AIRES, OBSERVATORIO CENTRAL BA
    
    filepath = raw_dir / "smn_climate_daily.csv"
    if not filepath.exists():
        logger.warning("Datos SMN no encontrados. Ver instrucciones en docs/data_sources.md")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, parse_dates=["date"])
    logger.info(
        "Cargados %d registros climáticos SMN (%s a %s)",
        len(df),
        df["date"].min().date() if len(df) else "N/A",
        df["date"].max().date() if len(df) else "N/A"
    )
    return df


def load_ba_datos_climate(raw_dir: Path) -> pd.DataFrame:
    """
    Carga datos climáticos de BA en Datos (GCBA).
    
    Returns
    -------
    pd.DataFrame con variables climáticas históricas de CABA.
    """
    # TODO Sprint 1 - Tarea HU2:
    # Descargar desde:
    # https://data.buenosaires.gob.ar (sección Cambio Climático)
    
    filepath = raw_dir / "ba_datos_climate.csv"
    if not filepath.exists():
        logger.warning("Datos BA en Datos no encontrados.")
        return pd.DataFrame()
    
    return pd.read_csv(filepath, parse_dates=["date"])


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def run_ingestion(config_path: str = "config.yaml"):
    """
    Pipeline principal de ingesta.
    Carga todas las fuentes y guarda reporte de disponibilidad.
    """
    config = load_config(config_path)
    raw_dir = Path(config["data"]["paths"]["raw"])
    
    logger.info("=== Iniciando ingesta de datos ===")
    
    # Epidemiológicos
    df_datos_gob = load_dengue_datos_gob(raw_dir)
    df_ben = load_ben_weekly(raw_dir)
    df_bes = load_bes_caba(raw_dir)
    
    # Climáticos
    df_smn = load_smn_climate(raw_dir)
    df_ba = load_ba_datos_climate(raw_dir)
    
    # Reporte de disponibilidad
    report = {
        "datos_gob_ar": len(df_datos_gob),
        "BEN": len(df_ben),
        "BES_CABA": len(df_bes),
        "SMN": len(df_smn),
        "BA_datos": len(df_ba),
    }
    
    logger.info("=== Reporte de ingesta ===")
    for source, n_records in report.items():
        status = "OK" if n_records > 0 else "PENDIENTE"
        logger.info("  %-20s: %6d registros [%s]", source, n_records, status)
    
    return report


if __name__ == "__main__":
    run_ingestion()
