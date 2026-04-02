# Registro de Decisiones Técnicas — Sprint 1

**Proyecto:** Predicción espacio-temporal de brotes de dengue en CABA  
**Sprint:** 1 — Recolección y preparación de datos   
**Responsable:** Ing. Paola Andrea Blanco Blanco  

---

## DT-01: Fuentes de datos epidemiológicos — estrategia multi-fuente

**Contexto**  
Se evaluaron las fuentes públicas disponibles para obtener datos históricos de casos de dengue en CABA con granularidad de comuna y semana epidemiológica.

**Opciones evaluadas**

| Fuente | Granularidad espacial | Período | Formato |
|--------|----------------------|---------|---------|
| BA Data — Reporte Epidemiológico de Dengue | Por comuna (1-15) | 2025 | CSV |
| datos.gob.ar — Vigilancia Dengue y Zika 2023-2024 | Por comuna (1-15) | 2023-2024 | CSV |
| datos.gob.ar — Vigilancia Dengue y Zika 2022 | Solo total CABA | 2022 | CSV |

**Decisión**  
Se utilizaron las fuentes de 2023 y 2024 de datos.gob.ar junto con BA Data 2025, descartando el dataset de 2022 porque no tiene desagregación por comuna. Esto resulta en 3 años de datos (2023-2025) con granularidad de comuna.

**Justificación**  
El modelo requiere datos a nivel de comuna para la componente espacial. Un dataset sin desagregación por comuna no aporta valor para el objetivo del proyecto. Con 3 años se obtienen aproximadamente 2,340 observaciones (15 comunas × 52 semanas × 3 años), suficiente para entrenar modelos de ML según el paper de referencia (Sebastianelli et al., 2024).

**Consecuencias y riesgos**  
- Riesgo: 3 años puede ser insuficiente para capturar ciclos inter-epidémicos. Mitigación: se incluirá el año 2022 a nivel CABA como feature adicional si fuera necesario.
- 2025 tiene datos parciales (solo hasta semana 45) con 175 casos — muy pocos comparados con 2023 (13,132) y 2024 (23,771). Se documentará esta limitación en la memoria.

---

## DT-02: Tratamiento de registros SIN DATO en departamento_residencia

**Contexto**  
El dataset de BA Data contiene registros donde `departamento_residencia = "SIN DATO"`. Se debía decidir si conservarlos o descartarlos.

**Análisis realizado**  
Se verificó que el 100% de los registros con SIN DATO tienen `n_confirmados = 0`. No aportan información de casos reales.

**Decisión**  
Descartar los registros con SIN DATO ya que tienen cero casos confirmados y no aportan información útil al modelo.

**Justificación**  
No hay pérdida de información real — los casos confirmados en esas filas son todos cero. Mantenerlos solo agregaría ruido al dataset.

---

## DT-03: Separador del CSV de BA Data

**Contexto**  
Al intentar cargar el CSV de BA Data con `pd.read_csv()` por defecto (separador coma), todas las columnas quedaron en una sola columna gigante.

**Causa identificada**  
El CSV de BA Data usa punto y coma (`;`) como separador en lugar de coma (`,`), lo cual es común en datasets publicados en Argentina donde la coma se usa como separador decimal.

**Decisión**  
Usar `sep=";"` en la función `pd.read_csv()` para este dataset específicamente. Para los datasets de datos.gob.ar se detecta el separador automáticamente.

**Aprendizaje**  
Siempre verificar el separador antes de asumir que es coma. Se documentó en `docs/data_sources.md` para futuras incorporaciones de datos.

---

## DT-04: Fuente de datos climáticos — Open-Meteo vs SMN

**Contexto**  
Se evaluaron las fuentes disponibles para obtener datos históricos de temperatura, humedad y precipitación para CABA en el período 2022-2025.

**Opciones evaluadas**

| Fuente | Cobertura temporal | Gaps | Descarga | Calidad |
|--------|-------------------|------|----------|---------|
| SMN — datos.gob.ar | Hasta nov 2021 | Posibles | Manual | Mediciones reales |
| SMN — portal directo | Reciente | Posibles | Manual | Mediciones reales |
| Open-Meteo ERA5 (historical-forecast-api) | 2022-presente | Ninguno | Automática vía API | Reanálisis científico |

**Decisión**  
Se utilizó la API de Open-Meteo con datos del modelo de reanálisis ERA5 accesible a través del endpoint `historical-forecast-api.open-meteo.com`.

**Justificación**  
- Los datasets del SMN en datos.gob.ar no se actualizan desde noviembre 2021 — no cubren el período de interés
- Open-Meteo es gratuito, no requiere registro y permite descarga automática vía Python
- ERA5 es el modelo de reanálisis más usado en investigación climática mundial (citado en Sebastianelli et al., 2024)
- Sin gaps garantizados — el modelo matemático produce un valor para cada día y coordenada
- Resolución espacial de 9-25 km — suficiente para una ciudad del tamaño de CABA

**URL final utilizada**  
`https://historical-forecast-api.open-meteo.com/v1/forecast`

**Nota técnica**  
Durante el desarrollo se identificó que el endpoint original documentado (`archive.meteo.open-meteo.com/v1/era5`) está discontinuado. La URL correcta y funcional es la indicada arriba.

**Consecuencias**  
- Los datos son de reanálisis, no mediciones directas de estaciones en CABA. Esto es una limitación menor — ERA5 tiene alta precisión para variables de superficie en zonas urbanas.
- Se documenta como limitación en la sección de gobernanza del proyecto.

---

## DT-05: Estación meteorológica de referencia para CABA

**Contexto**  
CABA es una ciudad pequeña (202 km²) con múltiples estaciones meteorológicas. Se debía elegir una coordenada de referencia para la descarga de datos climáticos.

**Opciones evaluadas**
- Observatorio Central Buenos Aires: lat -34.58°, lon -58.48° (centro geográfico)
- Aeroparque Jorge Newbery: lat -34.56°, lon -58.42° (norte)
- Ezeiza: lat -34.82°, lon -58.53° (fuera de CABA)

**Decisión**  
Se utilizó el Observatorio Central Buenos Aires (-34.58°, -58.48°) como punto de referencia único para todo CABA.

**Justificación**  
- Es la estación histórica de referencia para Buenos Aires, con registros desde 1906
- Está ubicada en el centro geográfico de CABA, minimizando la distancia promedio a todas las comunas
- ERA5 tiene resolución de 25km — la diferencia entre estaciones dentro de CABA es menor a la resolución del modelo, por lo que usar una sola coordenada es suficiente

**Limitación reconocida**  
Las 15 comunas no tienen exactamente el mismo microclima. Las comunas del sur (8, 9) cerca del Riachuelo pueden tener más humedad que las del norte. Esta variabilidad intra-ciudad no es capturada por el modelo actual. Se documenta como trabajo futuro.

---

## DT-06: Período de descarga de datos climáticos

**Contexto**  
Los datos epidemiológicos cubren 2023-2025. Se debía decidir desde qué año descargar los datos climáticos.

**Decisión**  
Descargar datos climáticos desde el 1 de enero de 2022, un año antes del inicio de los datos epidemiológicos.

**Justificación**  
El modelo va a usar variables rezagadas (lags) de hasta 4 semanas para las variables climáticas. Para tener lags disponibles desde la primera semana epidemiológica de 2023, necesitamos datos climáticos de las últimas semanas de 2022. Descargar desde enero 2022 garantiza que no falten lags al inicio de la serie.

---

## DT-07: Resolución temporal — diario vs semanal

**Contexto**  
Los datos climáticos de Open-Meteo se obtienen con resolución diaria. Los datos epidemiológicos tienen resolución semanal (semana epidemiológica). Se debía decidir cómo manejar esta diferencia.

**Decisión**  
Descargar datos climáticos a resolución diaria y agregarlos a resolución semanal mediante:
- Temperatura: promedio semanal de máximas, mínimas y medias
- Precipitación: suma acumulada semanal
- Humedad: promedio semanal

**Justificación**  
- La variable objetivo (casos de dengue) está a resolución semanal — todas las features deben estar en la misma resolución para el modelado
- Agregar de diario a semanal preserva más información que descargar datos semanales directamente
- La suma de precipitación semanal es más informativa que el promedio — lo que importa es cuánta lluvia cayó en la semana, no el promedio diario

---

## DT-08: Variables climáticas derivadas incluidas

**Contexto**  
Además de las variables crudas (temperatura, humedad, precipitación), se evaluó qué variables derivadas calcular.

**Decisión**  
Se calcularon dos variables derivadas:
1. **Índice de calor** (Heat Index) — usando la fórmula de Rothfusz (NOAA, 1990)
2. **Anomalías climáticas** — desviación respecto a la media histórica de cada semana

**Justificación**  
- El índice de calor captura la interacción temperatura × humedad en una sola variable, más relevante para el ciclo del *Aedes aegypti* que ambas variables por separado
- Las anomalías climáticas son mejores predictores de brotes que los valores absolutos según el paper de referencia (Sebastianelli et al., 2024) — una semana cálida en invierno es más relevante epidemiológicamente que una semana cálida en verano
- Ambas variables están documentadas en el `config.yaml` del proyecto

---

## Resumen de datasets generados

| Archivo | Filas | Período | Descripción |
|---------|-------|---------|-------------|
| `data/processed/dengue_weekly_comuna.parquet` | 2,340 | 2023-2025 | Casos por semana y comuna |
| `data/processed/clima_caba_semanal.parquet` | 210 | 2022-2026 | Variables climáticas semanales |

## Próximos pasos (Sprint 2)

- Unir ambos datasets en un único DataFrame maestro
- Verificar alineación de semanas epidemiológicas entre fuentes
- Análisis de completitud final antes del EDA
