"""
src/data/cleaning.py
=====================
Unificación y limpieza del dataset maestro.

Une los dos datasets procesados en Sprint 1:
    - dengue_weekly_comuna.parquet  → casos por semana y comuna
    - clima_caba_semanal.parquet    → variables climáticas semanales

Y genera un único dataset maestro limpio listo para el EDA y modelado.

Pipeline:
    Paso 1 → Cargar ambos datasets procesados
    Paso 2 → Verificar alineación temporal (semanas en común)
    Paso 3 → Unificar por year + epi_week
    Paso 4 → Identificar valores faltantes
    Paso 5 → Identificar duplicados
    Paso 6 → Verificar rangos y outliers
    Paso 7 → Estandarizar formatos
    Paso 8 → Generar reporte de calidad
    Paso 9 → Guardar dataset maestro
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES
# =============================================================================

PROCESSED_DIR = Path("data/processed")

DENGUE_FILE  = PROCESSED_DIR / "dengue_weekly_comuna.parquet"
CLIMA_FILE   = PROCESSED_DIR / "clima_caba_semanal.parquet"
MAESTRO_FILE = PROCESSED_DIR / "dataset_maestro.parquet"
MAESTRO_CSV  = PROCESSED_DIR / "dataset_maestro.csv"

# Rangos válidos para validación final del dataset unificado
RANGOS_VALIDOS = {
    "confirmed_cases": (0, 5000),
    "temp_max_mean":   (-5, 45),
    "temp_min_mean":   (-5, 45),
    "temp_mean":       (-5, 45),
    "precipitation":   (0, 500),
    "humidity_mean":   (0, 100),
    "heat_index_mean": (-5, 60),
}


# =============================================================================
# PASO 1: CARGA DE DATASETS
# =============================================================================

def cargar_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los dos datasets procesados del Sprint 1.
    Verifica que ambos existan antes de continuar.
    """
    logger.info("--- PASO 1: Cargando datasets procesados ---")

    for filepath in [DENGUE_FILE, CLIMA_FILE]:
        if not filepath.exists():
            raise FileNotFoundError(
                f"\nArchivo no encontrado: {filepath}\n"
                "Ejecutá primero ingestion.py y climate_ingestion.py"
            )

    df_dengue = pd.read_parquet(DENGUE_FILE)
    df_clima  = pd.read_parquet(CLIMA_FILE)

    logger.info(
        "  Dengue:  %d filas | columnas: %s",
        len(df_dengue), list(df_dengue.columns)
    )
    logger.info(
        "  Clima:   %d filas | columnas: %s",
        len(df_clima), list(df_clima.columns)
    )

    return df_dengue, df_clima


# =============================================================================
# PASO 2: VERIFICAR ALINEACIÓN TEMPORAL
# Antes de unir verificamos que los períodos se superpongan correctamente.
# Los datos de dengue cubren 2023-2025.
# Los datos de clima cubren 2022-2026 (más amplio, para tener lags).
# La intersección debería ser 2023-2025.
# =============================================================================

def verificar_alineacion(
    df_dengue: pd.DataFrame,
    df_clima: pd.DataFrame,
) -> None:
    """
    Verifica que los datasets tienen semanas en común suficientes.
    Advierte si hay semanas en dengue sin datos climáticos o viceversa.
    """
    logger.info("--- PASO 2: Verificando alineación temporal ---")

    # Años y semanas únicos en cada dataset
    semanas_dengue = set(
        zip(df_dengue["year"].astype(int), df_dengue["epi_week"].astype(int))
    )
    semanas_clima = set(
        zip(df_clima["year"].astype(int), df_clima["epi_week"].astype(int))
    )

    # Semanas que están en dengue pero no en clima
    solo_dengue = semanas_dengue - semanas_clima
    # Semanas que están en clima pero no en dengue
    solo_clima = semanas_clima - semanas_dengue
    # Intersección
    comunes = semanas_dengue & semanas_clima

    logger.info(
        "  Semanas en dengue:          %d (años: %s)",
        len(semanas_dengue),
        sorted(df_dengue["year"].unique().tolist())
    )
    logger.info(
        "  Semanas en clima:           %d (años: %s)",
        len(semanas_clima),
        sorted(df_clima["year"].unique().tolist())
    )
    logger.info("  Semanas en común:           %d", len(comunes))

    if solo_dengue:
        logger.warning(
            "  Semanas en dengue SIN datos climáticos: %d — quedarán con NaN",
            len(solo_dengue)
        )
    if solo_clima:
        logger.info(
            "  Semanas en clima sin datos de dengue: %d — fuera del período",
            len(solo_clima)
        )


# =============================================================================
# PASO 3: UNIFICACIÓN
# Hacemos un LEFT JOIN desde dengue hacia clima.
# Esto significa que conservamos todas las filas del dataset de dengue
# (todas las comunas × semanas) y agregamos las columnas climáticas.
# Como el clima no varía por comuna (es el mismo para toda CABA),
# cada semana tiene los mismos valores climáticos para las 15 comunas.
# =============================================================================

def unificar_datasets(
    df_dengue: pd.DataFrame,
    df_clima: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une dengue y clima por year + epi_week usando LEFT JOIN.

    El resultado tiene una fila por (year, epi_week, comuna_id) con
    todas las variables epidemiológicas y climáticas juntas.
    """
    logger.info("--- PASO 3: Unificando datasets ---")

    # Aseguramos que los tipos sean consistentes para el join
    df_dengue = df_dengue.copy()
    df_clima  = df_clima.copy()

    for df in [df_dengue, df_clima]:
        df["year"]     = df["year"].astype(int)
        df["epi_week"] = df["epi_week"].astype(int)

    # Columnas climáticas a incluir (descartamos fecha_inicio del clima
    # para no tener columnas duplicadas con la de dengue)
    cols_clima = [c for c in df_clima.columns
                  if c not in ["year", "epi_week", "fecha_inicio", "n_dias"]]

    df_maestro = df_dengue.merge(
        df_clima[["year", "epi_week"] + cols_clima],
        on=["year", "epi_week"],
        how="left",
    )

    logger.info(
        "  Dataset maestro: %d filas × %d columnas",
        len(df_maestro), len(df_maestro.columns)
    )
    logger.info("  Columnas: %s", list(df_maestro.columns))

    return df_maestro


# =============================================================================
# PASO 4: IDENTIFICAR VALORES FALTANTES
# Después del join pueden quedar NaN en las columnas climáticas
# si hay semanas en dengue sin cobertura climática.
# =============================================================================

def identificar_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica y reporta valores faltantes por columna.
    Para los NaN en variables climáticas aplica interpolación lineal.
    """
    logger.info("--- PASO 4: Identificando valores faltantes ---")
    df = df.copy()

    # Conteo de NaN por columna
    nans = df.isnull().sum()
    nans_pct = (nans / len(df) * 100).round(2)

    cols_con_nans = nans[nans > 0]
    if len(cols_con_nans) == 0:
        logger.info("  No hay valores faltantes en el dataset.")
        return df

    logger.warning("  Columnas con valores faltantes:")
    for col in cols_con_nans.index:
        logger.warning("    %-30s: %4d NaN (%.1f%%)", col, nans[col], nans_pct[col])

    # Imputación para columnas climáticas
    # Estrategia: interpolación lineal ordenada por year + epi_week
    # Es la estrategia más adecuada para series temporales continuas
    cols_climaticas = [
        "temp_max_mean", "temp_min_mean", "temp_mean",
        "precipitation", "humidity_mean", "heat_index_mean",
        "temp_mean_anomaly", "precipitation_anomaly", "humidity_mean_anomaly"
    ]

    cols_a_imputar = [c for c in cols_climaticas if c in df.columns and df[c].isnull().any()]

    if cols_a_imputar:
        df = df.sort_values(["year", "epi_week", "comuna_id"])
        for col in cols_a_imputar:
            n_antes = df[col].isnull().sum()
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            n_despues = df[col].isnull().sum()
            logger.info(
                "  Imputación lineal en %-30s: %d → %d NaN",
                col, n_antes, n_despues
            )

    return df


# =============================================================================
# PASO 5: IDENTIFICAR DUPLICADOS
# Un duplicado sería una fila con el mismo (year, epi_week, comuna_id).
# No debería haber ninguno dado por cómo se construyerón los datasets,
# pero lo verificamos por completitud.
# =============================================================================

def identificar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica y elimina filas duplicadas por (year, epi_week, comuna_id).
    """
    logger.info("--- PASO 5: Identificando duplicados ---")
    df = df.copy()

    clave = ["year", "epi_week", "comuna_id"]
    n_duplicados = df.duplicated(subset=clave).sum()

    if n_duplicados == 0:
        logger.info("  No hay duplicados en el dataset.")
    else:
        logger.warning(
            "  Duplicados encontrados: %d — eliminando, conservando primera ocurrencia",
            n_duplicados
        )
        df = df.drop_duplicates(subset=clave, keep="first")

    return df


# =============================================================================
# PASO 6: VERIFICAR RANGOS Y OUTLIERS
# Verificamos que todos los valores estén dentro de rangos físicamente
# razonables definidos en RANGOS_VALIDOS.
# Los outliers extremos se marcan como NaN y se imputan.
# =============================================================================

def verificar_rangos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica rangos válidos para cada variable.
    Marca como NaN los valores fuera de rango y los imputa.
    """
    logger.info("--- PASO 6: Verificando rangos y outliers ---")
    df = df.copy()

    total_outliers = 0
    for col, (vmin, vmax) in RANGOS_VALIDOS.items():
        if col not in df.columns:
            continue

        fuera_rango = ~df[col].between(vmin, vmax) & df[col].notna()
        n = fuera_rango.sum()

        if n > 0:
            logger.warning(
                "  %-30s: %d valores fuera de [%g, %g] → imputados",
                col, n, vmin, vmax
            )
            df.loc[fuera_rango, col] = np.nan
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            total_outliers += n

    if total_outliers == 0:
        logger.info("  Todos los valores dentro de rangos válidos.")

    return df


# =============================================================================
# PASO 7: ESTANDARIZAR FORMATOS
# Aseguramos tipos de datos consistentes y orden de columnas lógico.
# =============================================================================

def estandarizar_formatos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza tipos de datos y orden de columnas del dataset maestro.
    """
    logger.info("--- PASO 7: Estandarizando formatos ---")
    df = df.copy()

    # Tipos enteros para claves
    df["year"]           = df["year"].astype(int)
    df["epi_week"]       = df["epi_week"].astype(int)
    df["comuna_id"]      = df["comuna_id"].astype(int)
    df["confirmed_cases"] = df["confirmed_cases"].astype(int)

    # Tipos float para variables continuas
    cols_float = [
        "temp_max_mean", "temp_min_mean", "temp_mean",
        "precipitation", "humidity_mean", "heat_index_mean",
        "temp_mean_anomaly", "precipitation_anomaly", "humidity_mean_anomaly"
    ]
    for col in cols_float:
        if col in df.columns:
            df[col] = df[col].astype(float).round(3)

    # Orden lógico de columnas: claves → epidemiológicas → climáticas → derivadas
    orden = (
        ["year", "epi_week", "comuna_id", "confirmed_cases"] +
        [c for c in df.columns if "temp" in c and "anomaly" not in c] +
        [c for c in df.columns if "precipitation" in c and "anomaly" not in c] +
        [c for c in df.columns if "humidity" in c and "anomaly" not in c] +
        [c for c in df.columns if "heat_index" in c] +
        [c for c in df.columns if "anomaly" in c] +
        [c for c in df.columns if c not in [
            "year", "epi_week", "comuna_id", "confirmed_cases"
        ] and "temp" not in c and "precipitation" not in c
        and "humidity" not in c and "heat_index" not in c
        and "anomaly" not in c]
    )

    # Eliminamos duplicados del orden
    orden_unico = list(dict.fromkeys(orden))
    df = df[[c for c in orden_unico if c in df.columns]]

    logger.info(
        "  Dataset estandarizado: %d filas × %d columnas",
        len(df), len(df.columns)
    )

    return df


# =============================================================================
# PASO 8: REPORTE DE CALIDAD
# Genera un resumen completo del dataset maestro para documentar
# el cumplimiento de los criterios de aceptación de HU3.
# =============================================================================

def reporte_calidad(df: pd.DataFrame):
    """
    Genera el reporte de calidad del dataset maestro.
    Cumple con el criterio de aceptación HU3 del plan de proyecto.
    """
    print("\n" + "=" * 60)
    print("  REPORTE DE CALIDAD — dataset_maestro")
    print("  Sprint 2 / HU3 — Criterios de aceptación")
    print("=" * 60)

    # Completitud general
    print(f"\n  COMPLETITUD")
    print(f"  Filas totales:            {len(df):,}")
    print(f"  Columnas:                 {len(df.columns)}")
    print(f"  Comunas cubiertas:        {df['comuna_id'].nunique()} / 15")
    print(f"  Años cubiertos:           {sorted(df['year'].unique().tolist())}")
    print(f"  Semanas epidemiológicas:  {df['epi_week'].nunique()} / 52")

    # Valores faltantes
    print(f"\n  VALORES FALTANTES")
    nans = df.isnull().sum()
    if nans.sum() == 0:
        print("  No hay valores faltantes.")
    else:
        for col, n in nans[nans > 0].items():
            print(f"    {col}: {n} NaN ({n/len(df)*100:.1f}%)")

    # Duplicados
    print(f"\n  DUPLICADOS")
    n_dup = df.duplicated(subset=["year", "epi_week", "comuna_id"]).sum()
    print(f"  Duplicados (year/semana/comuna): {n_dup}")

    # Estadísticas descriptivas clave
    print(f"\n  ESTADÍSTICAS DESCRIPTIVAS")
    print(f"  Casos dengue:")
    print(f"    Total período:          {df['confirmed_cases'].sum():,}")
    print(f"    Media/semana/comuna:    {df['confirmed_cases'].mean():.2f}")
    print(f"    Máximo:                 {df['confirmed_cases'].max()}")
    print(f"    Semanas con casos > 0:  {(df['confirmed_cases'] > 0).sum():,} "
          f"({(df['confirmed_cases'] > 0).mean()*100:.1f}%)")

    print(f"\n  Variables climáticas:")
    print(f"    Temperatura media:      {df['temp_mean'].mean():.1f}°C")
    print(f"    Precipitación media:    {df['precipitation'].mean():.1f} mm/semana")
    print(f"    Humedad media:          {df['humidity_mean'].mean():.1f}%")

    # Casos por año
    print(f"\n  CASOS POR AÑO")
    for year in sorted(df["year"].unique()):
        total = df[df["year"] == year]["confirmed_cases"].sum()
        print(f"    {year}: {total:,} casos")

    # Top 5 semanas con más casos (CABA total)
    print(f"\n  TOP 5 SEMANAS CON MÁS CASOS (CABA total)")
    top = (
        df.groupby(["year", "epi_week"])["confirmed_cases"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    for _, row in top.iterrows():
        print(f"    {int(row['year'])} SE{int(row['epi_week']):02d}: "
              f"{int(row['confirmed_cases']):,} casos")

    print("\n" + "=" * 60)
    print("  Criterios de aceptación HU3:")
    print(f"  ✓ Valores faltantes identificados y documentados")
    print(f"  ✓ Estrategia de imputación implementada (interpolación lineal)")
    print(f"  ✓ Rangos validados para todas las variables")
    print(f"  ✓ Formatos de fecha y códigos de comuna estandarizados")
    print(f"  ✓ Estadísticas descriptivas por fuente generadas")
    print(f"  ✓ Porcentajes de completitud documentados")
    print("=" * 60 + "\n")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_cleaning(save: bool = True) -> pd.DataFrame:
    """
    Pipeline completo de unificación y limpieza.

    Paso 1 → Cargar datasets procesados
    Paso 2 → Verificar alineación temporal
    Paso 3 → Unificar por year + epi_week
    Paso 4 → Identificar y tratar valores faltantes
    Paso 5 → Identificar y eliminar duplicados
    Paso 6 → Verificar rangos y outliers
    Paso 7 → Estandarizar formatos
    Paso 8 → Reporte de calidad
    Paso 9 → Guardar dataset maestro
    """
    print("\n" + "=" * 60)
    print("  SPRINT 2 — Unificación y limpieza de datos")
    print("  HU2 (sincronización) + HU3 (limpieza y validación)")
    print("=" * 60 + "\n")

    # Pasos 1-2: carga y verificación
    df_dengue, df_clima = cargar_datasets()
    verificar_alineacion(df_dengue, df_clima)

    # Paso 3: unificación
    df = unificar_datasets(df_dengue, df_clima)

    # Pasos 4-7: limpieza
    df = identificar_faltantes(df)
    df = identificar_duplicados(df)
    df = verificar_rangos(df)
    df = estandarizar_formatos(df)

    # Paso 8: reporte
    reporte_calidad(df)

    # Paso 9: guardar
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        df.to_parquet(MAESTRO_FILE, index=False)
        logger.info("Dataset maestro guardado en: %s", MAESTRO_FILE)

        df.to_csv(MAESTRO_CSV, index=False)
        logger.info("CSV de inspección guardado en: %s", MAESTRO_CSV)

    return df


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    df = run_cleaning(save=True)

    print("Primeras filas del dataset maestro:")
    print(df.head(5).to_string(index=False))

    print(f"\nColumnas del dataset maestro ({len(df.columns)} total):")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")