# Fuentes de datos — dengue-caba-forecast

Guía para la obtención y descarga de datos de cada fuente.  
**Sprint 1 / HU1 y HU2**

---

## Registro de cambios

| Revisión | Cambio | Fecha |
|----------|--------|-------|
| 0 | Creación del documento — fuentes planificadas originalmente | Sprint 1 |
| 1 | Actualización fuente climática: SMN reemplazado por Open-Meteo ERA5 | Sprint 1 (ejecución) |

---

## Datos epidemiológicos

### 1. datos.gob.ar — Vigilancia Dengue

**URL:** https://datos.gob.ar/dataset/salud-dengue-zika-vigilancia  
**Archivo destino:** `data/raw/dengue_datos_gob.csv`  
**Frecuencia:** Semanal  
**Cobertura:** Nacional, por departamento  

**Pasos:**
1. Ir a la URL
2. Descargar el CSV del dataset "Dengue y Zika - Casos y Brotes"
3. Guardar en `data/raw/dengue_datos_gob.csv`

**Columnas esperadas:**
- `año`, `semanas_epidemiologicas`, `provincia`, `departamento`
- `casos_confirmados`, `casos_sospechosos`, `tasa_x_100000`

---

### 2. BEN — Boletín Epidemiológico Nacional

**URL:** https://www.argentina.gob.ar/salud/epidemiologia/boletines  
**Archivo destino:** `data/raw/ben_dengue_consolidated.csv`  
**Frecuencia:** Semanal (publicación del Ministerio de Salud de la Nación)  

**Pasos:**
1. Descargar boletines semanales en PDF desde la URL
2. Buscar la tabla "Dengue por semana epidemiológica y provincia"
3. Extraer datos y consolidar en CSV (puede usarse tabula-py o extracción manual)
4. Columnas requeridas: `year`, `epi_week`, `province`, `confirmed_cases`

**Nota:** Los datos de CABA están desagregados a nivel departamento en algunos boletines.

---

### 3. BES-CABA — Boletín Epidemiológico Semanal CABA

**URL:** https://www.buenosaires.gob.ar/salud/epidemiologia  
**Archivo destino:** `data/raw/bes_caba_dengue.csv`  
**Frecuencia:** Semanal (publicación del Ministerio de Salud GCBA)  
**Granularidad:** Por **comuna** — la fuente más valiosa para el proyecto  

**Pasos:**
1. Descargar los boletines semanales de la sección Epidemiología
2. Buscar tabla "Dengue por comuna"
3. Extraer y consolidar en CSV
4. Columnas requeridas: `year`, `epi_week`, `comuna_id` (1-15), `confirmed_cases`

---

## Datos climáticos

### 4. ~~SMN — Servicio Meteorológico Nacional~~ *(DESCARTADO — ver nota)*

> ⚠️ **Esta fuente fue descartada durante la ejecución del Sprint 1.**
>
> **Razón:** Los datasets del SMN disponibles en datos.gob.ar no se actualizan
> desde noviembre de 2021 y no cubren el período de interés del proyecto
> (2022-2025). La descarga manual desde el portal del SMN tampoco resultó
> viable por inconsistencias en el formato y gaps en los datos.
>
> **Decisión técnica:** DT-04 del registro de decisiones técnicas Sprint 1.
>
> **Reemplazado por:** Open-Meteo ERA5 Historical Forecast API (ver fuente 4b).

~~**URL:** https://www.smn.gob.ar/descarga-de-datos~~  
~~**Archivo destino:** `data/raw/smn_climate_daily.csv`~~  

---

### 4b. Open-Meteo — Historical Forecast API (ERA5) *(FUENTE ACTIVA)*

**URL de la API:**
```
https://historical-forecast-api.open-meteo.com/v1/forecast
```

**Archivo destino:** `data/raw/clima_caba_diario_raw.csv`  
**Archivo procesado:** `data/processed/clima_caba_semanal.parquet`  
**Frecuencia de descarga:** Una sola vez (datos históricos estáticos)  
**Modelo meteorológico:** ERA5 — modelo de reanálisis del ECMWF  
**Resolución espacial:** 9-25 km  
**Resolución temporal:** Diaria (agregada a semanal en el pipeline)  

**Coordenadas utilizadas:**
- Observatorio Central Buenos Aires: lat -34.58°, lon -58.48°
- Decisión: DT-05 del registro de decisiones técnicas Sprint 1

**Período descargado:** 2022-01-01 a 2025-12-31

**Variables descargadas:**

| Variable API | Variable en dataset | Unidad |
|-------------|-------------------|--------|
| `temperature_2m_max` | `temp_max_mean` | °C |
| `temperature_2m_min` | `temp_min_mean` | °C |
| `temperature_2m_mean` | `temp_mean` | °C |
| `precipitation_sum` | `precipitation` | mm/semana |
| `relative_humidity_2m_max` | `humidity_max` | % |
| `relative_humidity_2m_min` | `humidity_min` | % |

**Variables derivadas calculadas en el pipeline:**

| Variable | Fórmula | Descripción |
|----------|---------|-------------|
| `heat_index_mean` | Rothfusz (NOAA, 1990) | Temperatura percibida combinando calor y humedad |
| `temp_mean_anomaly` | temp_mean - media histórica semana | Desviación respecto al promedio histórico |
| `precipitation_anomaly` | precipitation - media histórica semana | Desviación respecto al promedio histórico |
| `humidity_mean_anomaly` | humidity_mean - media histórica semana | Desviación respecto al promedio histórico |

**Ejemplo de llamada a la API:**
```
https://historical-forecast-api.open-meteo.com/v1/forecast
  ?latitude=-34.58
  &longitude=-58.48
  &start_date=2022-01-01
  &end_date=2025-12-31
  &daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,
         precipitation_sum,relative_humidity_2m_max,relative_humidity_2m_min
  &timezone=America/Argentina/Buenos_Aires
```

**Script de descarga:** `src/data/climate_ingestion.py`

**¿Por qué Open-Meteo en lugar del SMN?**

| Criterio | SMN | Open-Meteo ERA5 |
|----------|-----|-----------------|
| Cobertura temporal | Hasta nov 2021 | 2022-presente |
| Gaps en datos | Posibles | Ninguno (modelo matemático) |
| Descarga automática | No (manual) | Sí (API REST) |
| Actualización | Irregular | Diaria con delay de 5-7 días |
| Costo | Gratuito | Gratuito |
| Calidad | Mediciones reales | Reanálisis científico |

La principal desventaja de ERA5 es que son datos de reanálisis (modelo matemático) y no mediciones directas de estaciones. Sin embargo ERA5 tiene alta precisión para variables de superficie en zonas urbanas y es el estándar internacional en investigación climática.

**Historial de URLs probadas durante el Sprint 1:**

| URL | Estado | Motivo |
|-----|--------|--------|
| `archive.meteo.open-meteo.com/v1/era5` | ❌ Discontinuada | Dominio no resuelve DNS |
| `api.open-meteo.com/v1/archive` | ❌ Not Found | Endpoint incorrecto |
| `api.open-meteo.com/v1/forecast?past_days=365` | ❌ Inválido | Solo permite hasta 93 días |
| `historical-forecast-api.open-meteo.com/v1/forecast` | ✅ **Activa** | URL correcta y funcional |

---

### 5. BA en Datos — Cambio Climático GCBA

**URL:** https://data.buenosaires.gob.ar  
**Sección:** Medio Ambiente > Cambio Climático  
**Archivo destino:** `data/raw/ba_datos_climate.csv`  
**Estado:** Fuente complementaria — no utilizada en Sprint 1  

**Pasos:**
1. Buscar datasets de variables climáticas en el portal
2. Descargar en formato CSV
3. Guardar en `data/raw/ba_datos_climate.csv`

---

## Datos geográficos

### 6. GeoJSON comunas CABA

**Archivo destino:** `data/external/comunas_caba.geojson`  
**Estado:** Pendiente — requerido para Sprint 4 (features espaciales)

**Fuentes alternativas:**
- https://data.buenosaires.gob.ar/dataset/comunas
- https://github.com/mgaitan/baires (GeoJSON pre-procesado)

**Columnas necesarias:**
- `COMUNAS` (id numérico 1-15)
- `geometry` (polígono)

---

### 7. Matriz de vecindad entre comunas

**Archivo destino:** `data/external/comunas_adjacency.csv`  
**Estado:** Pendiente — requerido para Sprint 4 (features espaciales)

Se puede generar automáticamente desde el GeoJSON usando GeoPandas:

```python
import geopandas as gpd
from shapely.geometry import mapping

gdf = gpd.read_file('data/external/comunas_caba.geojson')
adjacency = []
for i, row_i in gdf.iterrows():
    for j, row_j in gdf.iterrows():
        if i != j and row_i.geometry.touches(row_j.geometry):
            adjacency.append({'comuna_a': row_i['COMUNAS'], 'comuna_b': row_j['COMUNAS']})

pd.DataFrame(adjacency).to_csv('data/external/comunas_adjacency.csv', index=False)
```

---

### 8. Población por comuna — Censo 2022 INDEC *(NUEVA — Sprint 4)*

**Fuente:** Censo Nacional de Población, Hogares y Viviendas 2022  
**URL:** https://www.indec.gob.ar/indec/web/Nivel4-Tema-2-41-165  
**Archivo destino:** `data/external/poblacion_comunas_caba_2022.csv`  
**Estado:** Pendiente — requerido para Sprint 4  

**Justificación:** El EDA del Sprint 3 identificó que los casos absolutos por comuna no son comparables sin normalizar por población. La tasa de incidencia per cápita (casos por 10,000 habitantes) es la métrica epidemiológicamente correcta.

**Datos a incorporar:**

| Comuna | Población estimada (Censo 2022) |
|--------|--------------------------------|
| 1 | A completar desde INDEC |
| 2 | A completar desde INDEC |
| ... | ... |
| 15 | A completar desde INDEC |

**Decisión técnica:** DT-13 del registro de decisiones técnicas Sprint 3.

---

## Validación de rangos

Usar los rangos del `config.yaml` para validar datos al momento de la ingesta:

| Variable | Mínimo | Máximo | Justificación |
|----------|--------|--------|---------------|
| Temperatura | -5°C | 45°C | Rango histórico CABA (mín absoluta: -5.4°C en 1918, máx: 43.3°C en 2022) |
| Humedad | 0% | 100% | Límite físico de la humedad relativa |
| Precipitación | 0 mm | 500 mm | Récord histórico CABA ~350 mm/semana |
| Casos dengue | 0 | 5,000 | Sin límite superior práctico |

---

## Estado actual de fuentes (post Sprint 3)

| Fuente | Estado | Archivo generado |
|--------|--------|-----------------|
| datos.gob.ar 2023 | ✅ Descargado | `data/raw/dengue_nacional_2023.csv` |
| datos.gob.ar 2024 | ✅ Descargado | `data/raw/dengue_nacional_2024.csv` |
| BA Data 2025 | ✅ Descargado | `data/raw/dengue_ba_data_raw.csv` |
| Open-Meteo ERA5 | ✅ Descargado | `data/raw/clima_caba_diario_raw.csv` |
| GeoJSON comunas | ⏳ Pendiente Sprint 4 | `data/external/comunas_caba.geojson` |
| Matriz vecindad | ⏳ Pendiente Sprint 4 | `data/external/comunas_adjacency.csv` |
| Población Censo 2022 | ⏳ Pendiente Sprint 4 | `data/external/poblacion_comunas_caba_2022.csv` |

---

## Checklist actualizado

### Sprint 1 — completado ✅
- [x] datos.gob.ar 2023 descargado y guardado
- [x] datos.gob.ar 2024 descargado y guardado
- [x] BA Data 2025 descargado y guardado
- [x] Open-Meteo ERA5 descargado (reemplazó al SMN)
- [x] Reporte de completitud por fuente documentado

### Sprint 4 — pendiente ⏳
- [ ] GeoJSON comunas CABA disponible en `data/external/`
- [ ] Matriz de vecindad generada
- [ ] Población por comuna (Censo 2022 INDEC) incorporada
