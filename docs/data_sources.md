# Fuentes de datos — dengue-caba-forecast

Guía para la obtención y descarga de datos de cada fuente.  
**Sprint 1 / HU1 y HU2**

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

### 4. SMN — Servicio Meteorológico Nacional

**URL:** https://www.smn.gob.ar/descarga-de-datos  
**Archivo destino:** `data/raw/smn_climate_daily.csv`  
**Frecuencia:** Diaria  

**Estaciones relevantes para CABA:**
- Observatorio Central Buenos Aires (lat: -34.58, lon: -58.48)
- Aeroparque Jorge Newbery (lat: -34.56, lon: -58.42)

**Variables a descargar:**
- Temperatura mínima, media, máxima (°C)
- Humedad relativa (%)
- Precipitación acumulada (mm)

**Pasos:**
1. Ir a la URL de descarga de datos SMN
2. Seleccionar estaciones de CABA
3. Seleccionar período histórico disponible
4. Descargar en formato CSV o TXT
5. Estandarizar columnas y guardar en `data/raw/smn_climate_daily.csv`

**Columnas esperadas:**
```
date, station_id, temp_min, temp_mean, temp_max, humidity, precipitation
```

---

### 5. BA en Datos — Cambio Climático GCBA

**URL:** https://data.buenosaires.gob.ar  
**Sección:** Medio Ambiente > Cambio Climático  
**Archivo destino:** `data/raw/ba_datos_climate.csv`  

**Pasos:**
1. Buscar datasets de variables climáticas en el portal
2. Descargar en formato CSV
3. Guardar en `data/raw/ba_datos_climate.csv`

---

## Datos geográficos

### 6. GeoJSON comunas CABA

**Archivo destino:** `data/external/comunas_caba.geojson`  

**Fuentes alternativas:**
- https://data.buenosaires.gob.ar/dataset/comunas
- https://github.com/mgaitan/baires (GeoJSON pre-procesado)

**Columnas necesarias:**
- `COMUNAS` (id numérico 1-15)
- `geometry` (polígono)

---

### 7. Matriz de vecindad entre comunas

**Archivo destino:** `data/external/comunas_adjacency.csv`  

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

## Validación de rangos

Usar los rangos del `config.yaml` para validar datos al momento de la ingesta:

| Variable | Mínimo | Máximo |
|----------|--------|--------|
| Temperatura | -5°C | 45°C |
| Humedad | 0% | 100% |
| Precipitación | 0 mm | 500 mm |
| Casos dengue | 0 | (sin límite superior) |

---

## Checklist Sprint 1

- [ ] datos.gob.ar descargado y guardado
- [ ] BEN consolidado (mínimo 3 años epidemiológicos)
- [ ] BES-CABA consolidado con columna `comuna_id`
- [ ] SMN datos diarios descargados
- [ ] BA en Datos descargado
- [ ] GeoJSON comunas CABA disponible
- [ ] Matriz de vecindad generada
- [ ] Reporte de completitud por fuente documentado
