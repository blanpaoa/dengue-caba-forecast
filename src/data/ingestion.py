"""
src/data/ingestion.py
=====================
Carga y procesamiento de datos de dengue CABA desde múltiples fuentes.

Fuentes:
    1. datos.gob.ar 2023: dengue_nacional_2023.csv
       Columnas: id_depto_indec_residencia, departamento_residencia,
                 id_prov_indec_residencia, provincia_residencia,
                 anio_min, evento, id_grupo_etario, grupo_etario,
                 sepi_min, cantidad

    2. datos.gob.ar 2024: dengue_nacional_2024.csv
       Mismas columnas que 2023

    3. BA Data 2025: dengue_ba_data_raw.csv
       Columnas: ano, semana_epidemiologica,
                 fecha_inicio_semana_epidemiologica,
                 grupo_etario, departamento_residencia,
                 n_confirmados, fecha_semana_epidemiologica

Pipeline:
    Paso 1 → Cargar cada fuente por separado
    Paso 2 → Estandarizar columnas al mismo formato
    Paso 3 → Filtrar solo CABA y solo Dengue
    Paso 4 → Limpiar comunas (mapear a ID 1-15, descartar SIN DATO)
    Paso 5 → Limpiar columnas temporales
    Paso 6 → Limpiar casos confirmados
    Paso 7 → Combinar los tres datasets
    Paso 8 → Agregar por (year, epi_week, comuna_id)
    Paso 9 → Completar semanas sin casos con 0
    Paso 10 → Guardar en data/processed/
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Archivos de entrada
ARCHIVO_2023      = RAW_DIR / "dengue_nacional_2023.csv"
ARCHIVO_2024      = RAW_DIR / "dengue_nacional_2024.csv"
ARCHIVO_BA_2025   = RAW_DIR / "dengue_ba_data_raw.csv"

# ID de CABA en el dataset nacional (provincia_id = 2)
CABA_PROVINCIA_ID = 2

# Mapeo nombre de comuna → ID numérico (aplica a ambas fuentes)
COMUNA_MAP = {
    "COMUNA 1":  1,  "COMUNA 2":  2,  "COMUNA 3":  3,
    "COMUNA 4":  4,  "COMUNA 5":  5,  "COMUNA 6":  6,
    "COMUNA 7":  7,  "COMUNA 8":  8,  "COMUNA 9":  9,
    "COMUNA 10": 10, "COMUNA 11": 11, "COMUNA 12": 12,
    "COMUNA 13": 13, "COMUNA 14": 14, "COMUNA 15": 15,
}

# Columnas estándar que tendrá el dataset combinado
# Todas las fuentes se convierten a este formato antes de unirse
COLUMNAS_ESTANDAR = [
    "year",         # año (int)
    "epi_week",     # semana epidemiológica (int)
    "comuna_id",    # ID de la comuna 1-15 (int)
    "n_casos",      # casos confirmados (int)
    "fuente",       # origen del dato: "nacional_2023", "nacional_2024", "ba_data_2025"
]


# =============================================================================
# PASO 1 y 2: CARGA Y ESTANDARIZACIÓN DE CADA FUENTE
# Cada función carga una fuente y devuelve un DataFrame con
# exactamente las columnas de COLUMNAS_ESTANDAR.
# Esto facilita combinarlas en el Paso 7.
# =============================================================================

def cargar_nacional(filepath: Path, fuente: str) -> pd.DataFrame:
    """
    Carga un archivo del dataset nacional (2023 o 2024) y lo
    estandariza al formato común.

    Pasos internos:
        - Lee el CSV (separador puede ser , o ;)
        - Filtra solo filas de CABA (id_prov_indec_residencia == 2)
        - Filtra solo filas de Dengue (evento == 'Dengue')
        - Descarta registros con departamento 'desconocido' o id=999
        - Renombra columnas al formato estándar
    """
    logger.info("Cargando %s...", filepath.name)

    # Intentamos primero con coma, luego con punto y coma
    try:
        df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
        # Si todas las columnas quedaron en una sola, es porque el separador es ;
        if len(df.columns) == 1:
            df = pd.read_csv(filepath, sep=";", encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        # Algunos archivos del gobierno usan latin-1
        df = pd.read_csv(filepath, sep=";", encoding="latin-1", low_memory=False)

    logger.info("  Filas cargadas: %d | Columnas: %s", len(df), list(df.columns))

    # Filtrar solo CABA
    # id_prov_indec_residencia = 2 corresponde a CABA
    antes = len(df)
    df = df[df["id_prov_indec_residencia"] == CABA_PROVINCIA_ID].copy()
    logger.info("  Filas CABA: %d (descartadas otras provincias: %d)", len(df), antes - len(df))

    # Filtrar solo Dengue (el dataset incluye también Zika y otras)
    df = df[df["evento"].str.upper() == "DENGUE"].copy()
    logger.info("  Filas Dengue CABA: %d", len(df))

    # Descartar registros con departamento desconocido (id=999)
    # Estos no tienen comuna asignada y no sirven para el modelo espacial
    n_desconocidos = (df["id_depto_indec_residencia"] == 999).sum()
    if n_desconocidos > 0:
        logger.warning("  Registros con departamento desconocido (id=999): %d — descartados", n_desconocidos)
        df = df[df["id_depto_indec_residencia"] != 999].copy()

    # Estandarizar nombres de columnas
    df = df.rename(columns={
        "anio_min":               "year",
        "sepi_min":               "epi_week",
        "departamento_residencia": "departamento_residencia",
        "cantidad":               "n_casos",
    })

    # Agregar columna que identifica la fuente
    df["fuente"] = fuente

    # Nos quedamos solo con las columnas que necesitamos
    df = df[["year", "epi_week", "departamento_residencia", "n_casos", "fuente"]].copy()

    return df


def cargar_ba_data(filepath: Path) -> pd.DataFrame:
    """
    Carga el archivo de BA Data (2025) y lo estandariza al
    formato común.

    Pasos internos:
        - Lee el CSV con separador punto y coma
        - Renombra columnas al formato estándar
        - Descarta filas con SIN DATO en departamento (n_confirmados = 0)
    """
    logger.info("Cargando %s...", filepath.name)

    df = pd.read_csv(filepath, sep=";", encoding="utf-8", low_memory=False)
    logger.info("  Filas cargadas: %d", len(df))

    # Descartar SIN DATO — ya confirmamos que tienen n_confirmados = 0
    sin_dato = df["departamento_residencia"].astype(str).str.upper().isin(
        ["SIN DATO", "NAN", ""]
    )
    n_sin_dato = sin_dato.sum()
    logger.info("  Registros SIN DATO: %d (todos con 0 casos) — descartados", n_sin_dato)
    df = df[~sin_dato].copy()

    # Estandarizar nombres de columnas
    df = df.rename(columns={
        "ano":                    "year",
        "semana_epidemiologica":  "epi_week",
        "n_confirmados":          "n_casos",
    })

    df["fuente"] = "ba_data_2025"

    df = df[["year", "epi_week", "departamento_residencia", "n_casos", "fuente"]].copy()

    return df


# =============================================================================
# PASO 3: LIMPIAR COMUNAS
# Aplica a cualquier DataFrame que tenga 'departamento_residencia'.
# Mapea "COMUNA X" → ID numérico X.
# =============================================================================

def limpiar_comunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza departamento_residencia y genera comuna_id (1-15).
    Descarta filas que no correspondan a una comuna válida.
    """
    df = df.copy()

    # Normalizamos texto: mayúsculas y sin espacios extra
    df["departamento_residencia"] = (
        df["departamento_residencia"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Mapeamos al ID numérico
    df["comuna_id"] = df["departamento_residencia"].map(COMUNA_MAP)

    # Registros que no se pudieron mapear (no son ninguna COMUNA X)
    no_mapeados = df["comuna_id"].isna()
    if no_mapeados.any():
        valores = df.loc[no_mapeados, "departamento_residencia"].unique()
        logger.warning("  Valores no mapeados a comuna: %s — descartados", valores)
        df = df[~no_mapeados].copy()

    df["comuna_id"] = df["comuna_id"].astype(int)
    return df


# =============================================================================
# PASO 4: LIMPIAR COLUMNAS TEMPORALES
# Convierte year y epi_week a enteros y filtra rangos válidos.
# =============================================================================

def limpiar_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte year y epi_week a enteros y filtra rangos válidos.
    """
    df = df.copy()

    df["year"]     = pd.to_numeric(df["year"],     errors="coerce").astype("Int64")
    df["epi_week"] = pd.to_numeric(df["epi_week"], errors="coerce").astype("Int64")

    antes = len(df)
    df = df[df["year"].between(2018, 2026)]
    df = df[df["epi_week"].between(1, 53)]

    descartados = antes - len(df)
    if descartados > 0:
        logger.warning("  Filas con rango temporal inválido: %d — descartadas", descartados)

    return df


# =============================================================================
# PASO 5: LIMPIAR CASOS
# Convierte n_casos a entero y descarta valores negativos.
# =============================================================================

def limpiar_casos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte n_casos a entero, descarta negativos, rellena vacíos con 0.
    """
    df = df.copy()

    df["n_casos"] = pd.to_numeric(df["n_casos"], errors="coerce")

    negativos = (df["n_casos"] < 0).sum()
    if negativos > 0:
        logger.warning("  Valores negativos: %d — eliminados", negativos)
        df = df[df["n_casos"] >= 0]

    df["n_casos"] = df["n_casos"].fillna(0).astype(int)
    return df


# =============================================================================
# PASO 6: COMBINAR LOS TRES DATASETS
# Une verticalmente los DataFrames de cada fuente.
# =============================================================================

def combinar_fuentes(dfs: list) -> pd.DataFrame:
    """
    Combina una lista de DataFrames en uno solo.
    Todos deben tener las mismas columnas estándar.
    """
    logger.info("--- PASO 6: Combinando fuentes ---")

    df_combinado = pd.concat(dfs, ignore_index=True)

    # Resumen por fuente
    for fuente, grupo in df_combinado.groupby("fuente"):
        logger.info(
            "  %-20s → %6d filas | %6d casos | años: %s",
            fuente,
            len(grupo),
            grupo["n_casos"].sum(),
            sorted(grupo["year"].unique().tolist()),
        )

    logger.info("  Total combinado: %d filas", len(df_combinado))
    return df_combinado


# =============================================================================
# PASO 7: AGREGAR POR COMUNA Y SEMANA
# Suma todos los grupos etarios para cada (year, epi_week, comuna_id).
# =============================================================================

def agregar_por_comuna_semana(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega casos sumando grupos etarios.
    Resultado: una fila por (year, epi_week, comuna_id).
    """
    logger.info("--- PASO 7: Agregando por año, semana y comuna ---")

    df_agg = (
        df.groupby(["year", "epi_week", "comuna_id"], as_index=False)
        .agg(confirmed_cases=("n_casos", "sum"))
        .sort_values(["year", "epi_week", "comuna_id"])
        .reset_index(drop=True)
    )

    logger.info("  Filas después de agregar: %d", len(df_agg))
    return df_agg


# =============================================================================
# PASO 8: COMPLETAR SEMANAS SIN CASOS CON 0
# Si una comuna no tuvo casos en una semana, no aparece en el dataset.
# Completamos esas combinaciones con confirmed_cases = 0.
# =============================================================================

def completar_semanas_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena con 0 las combinaciones (year, epi_week, comuna_id)
    que no aparecen en el dataset porque no hubo casos.
    """
    logger.info("--- PASO 8: Completando semanas sin casos con 0 ---")

    years = sorted(df["year"].dropna().unique())

    todas_las_combinaciones = pd.MultiIndex.from_product(
        [years, range(1, 53), range(1, 16)],
        names=["year", "epi_week", "comuna_id"],
    )

    df_completo = (
        df.set_index(["year", "epi_week", "comuna_id"])
        .reindex(todas_las_combinaciones, fill_value=0)
        .reset_index()
    )

    logger.info(
        "  Filas totales: %d (%d años × 52 semanas × 15 comunas)",
        len(df_completo),
        len(years),
    )
    return df_completo


# =============================================================================
# REPORTE DE CALIDAD
# =============================================================================

def reporte_calidad(df: pd.DataFrame):
    """Imprime un resumen del dataset procesado final."""
    print("\n" + "=" * 55)
    print("  REPORTE DE CALIDAD — dengue_weekly_comuna")
    print("=" * 55)
    print(f"  Filas totales:          {len(df):,}")
    print(f"  Comunas cubiertas:      {df['comuna_id'].nunique()} / 15")
    print(f"  Años cubiertos:         {sorted(df['year'].unique().tolist())}")
    print(f"  Semanas con casos > 0:  {(df['confirmed_cases'] > 0).sum():,} "
          f"({(df['confirmed_cases'] > 0).mean()*100:.1f}%)")
    print(f"  Total casos período:    {df['confirmed_cases'].sum():,}")
    print(f"  Media casos/sem/comuna: {df['confirmed_cases'].mean():.2f}")
    print(f"  Máximo casos/sem/comuna:{df['confirmed_cases'].max()}")

    print("\n  Casos por año:")
    for year in sorted(df["year"].unique()):
        total = df[df["year"] == year]["confirmed_cases"].sum()
        print(f"    {year}: {total:,} casos")

    comunas_faltantes = set(range(1, 16)) - set(df["comuna_id"].unique())
    if comunas_faltantes:
        print(f"\n  ADVERTENCIA — Comunas sin datos: {sorted(comunas_faltantes)}")
    else:
        print("\n  Todas las 15 comunas tienen datos.")
    print("=" * 55 + "\n")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_ingestion(save: bool = True) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de ingesta combinando las tres fuentes.

    Verifica que los archivos existan antes de comenzar y lanza
    un mensaje claro si falta alguno.
    """
    print("\n" + "=" * 55)
    print("  SPRINT 1 — Ingesta multi-fuente Dengue CABA")
    print("  Fuentes: 2023 + 2024 (nacional) + 2025 (BA Data)")
    print("=" * 55 + "\n")

    # Verificar que todos los archivos estén presentes
    archivos_requeridos = {
        "2023 (nacional)": ARCHIVO_2023,
        "2024 (nacional)": ARCHIVO_2024,
        "2025 (BA Data)":  ARCHIVO_BA_2025,
    }
    faltantes = [
        nombre for nombre, path in archivos_requeridos.items()
        if not path.exists()
    ]
    if faltantes:
        raise FileNotFoundError(
            f"\nFaltan los siguientes archivos en data/raw/:\n" +
            "\n".join(f"  - {nombre}: {archivos_requeridos[nombre].name}"
                      for nombre in faltantes) +
            "\n\nDescargalos desde:\n"
            "  2023/2024: https://datos.gob.ar/dataset/salud-vigilancia-enfermedades-por-virus-dengue-zika\n"
            "  2025: https://data.buenosaires.gob.ar/dataset/reporte-epidemiologico-de-dengue"
        )

    # --- Paso 1-2: Cargar y estandarizar cada fuente ---
    logger.info("=== PASO 1-2: Carga y estandarización ===")
    df_2023 = cargar_nacional(ARCHIVO_2023, fuente="nacional_2023")
    df_2024 = cargar_nacional(ARCHIVO_2024, fuente="nacional_2024")
    df_2025 = cargar_ba_data(ARCHIVO_BA_2025)

    # --- Paso 3: Limpiar comunas en cada fuente ---
    logger.info("=== PASO 3: Limpieza de comunas ===")
    df_2023 = limpiar_comunas(df_2023)
    df_2024 = limpiar_comunas(df_2024)
    df_2025 = limpiar_comunas(df_2025)

    # --- Paso 4: Limpiar temporales ---
    logger.info("=== PASO 4: Limpieza temporal ===")
    df_2023 = limpiar_temporal(df_2023)
    df_2024 = limpiar_temporal(df_2024)
    df_2025 = limpiar_temporal(df_2025)

    # --- Paso 5: Limpiar casos ---
    logger.info("=== PASO 5: Limpieza de casos ===")
    df_2023 = limpiar_casos(df_2023)
    df_2024 = limpiar_casos(df_2024)
    df_2025 = limpiar_casos(df_2025)

    # --- Paso 6: Combinar ---
    df_todo = combinar_fuentes([df_2023, df_2024, df_2025])

    # --- Paso 7: Agregar por comuna y semana ---
    df_agg = agregar_por_comuna_semana(df_todo)

    # --- Paso 8: Completar semanas sin casos ---
    df_final = completar_semanas_faltantes(df_agg)

    # Reporte de calidad
    reporte_calidad(df_final)

    # --- Paso 9: Guardar ---
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Parquet para el modelo
        df_final["year"]     = df_final["year"].astype(int)
        df_final["epi_week"] = df_final["epi_week"].astype(int)
        parquet_path = PROCESSED_DIR / "dengue_weekly_comuna.parquet"
        df_final.to_parquet(parquet_path, index=False)
        logger.info("Guardado en: %s", parquet_path)

        # CSV para inspección en Excel
        csv_path = PROCESSED_DIR / "dengue_weekly_comuna.csv"
        df_final.to_csv(csv_path, index=False)
        logger.info("CSV de inspección: %s", csv_path)

    return df_final


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    df = run_ingestion(save=True)

    print("Muestra de semanas CON casos (top 10 por cantidad):")
    print(
        df[df["confirmed_cases"] > 0]
        .sort_values("confirmed_cases", ascending=False)
        .head(10)
        .to_string(index=False)
    )