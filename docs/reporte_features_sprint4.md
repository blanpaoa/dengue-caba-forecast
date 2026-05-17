# Reporte de Feature Engineering — Sprint 4

**Proyecto:** Predicción espacio-temporal de brotes de dengue en CABA  
**Sprint:** 4 — HU5: Creación de features para el modelado  
**Script:** `src/features/lags.py`  ---

## 1. Resumen ejecutivo

A partir del dataset maestro (`dataset_maestro.parquet`, 2,340 filas × 13 columnas) se generaron **67 features** organizadas en 5 categorías. El dataset resultante (`dataset_features.parquet`) está listo para el modelado del Sprint 5.

| Categoría | Cantidad de features |
|-----------|---------------------|
| Epidemiológicas (casos + lags) | 10 |
| Climáticas base | 7 |
| Climáticas con lags | 28 |
| Estacionalidad | 4 |
| Espaciales | 18 |
| **TOTAL** | **67** |

De estas 67, `confirmed_cases` e `incidencia_x10000` son las **variables objetivo** — lo que el modelo predice. Las restantes **65 son features predictoras**.

---

## 2. Variables objetivo

Son las variables que el modelo va a predecir. No se usan como input sino como output esperado durante el entrenamiento.

| Variable | Tipo | Descripción | Unidad |
|----------|------|-------------|--------|
| `confirmed_cases` | Entero | Casos confirmados de dengue por semana y comuna | Casos absolutos |
| `incidencia_x10000` | Float | Casos por cada 10,000 habitantes de la comuna | Casos / 10,000 hab |

**¿Por qué dos variables objetivo?**

`confirmed_cases` permite predecir la magnitud absoluta del brote — cuántas personas se van a enfermar. Es la métrica más directa pero favorece a las comunas más pobladas.

`incidencia_x10000` normaliza por la población de cada comuna, permitiendo comparar el riesgo entre comunas de diferente tamaño de forma justa. Una comuna de 200,000 habitantes con 100 casos tiene más riesgo per cápita que una de 500,000 habitantes con los mismos 100 casos.

---

## 3. Features epidemiológicas — 10 features

### 3.1 Casos base

| Feature | Descripción | Decisión técnica |
|---------|-------------|-----------------|
| `confirmed_cases` | Casos confirmados de dengue en la semana actual | Variable objetivo (no usar como predictor) |
| `incidencia_x10000` | Tasa de incidencia por 10,000 habitantes | DT-13 — Censo 2022 INDEC |

### 3.2 Lags de casos por comuna

Los lags capturan la **inercia epidémica** — si esta semana hay muchos casos, la semana siguiente probablemente también habrá muchos. Se calculan **dentro de cada comuna** para respetar la dimensión espacial.

| Feature | Descripción | Justificación epidemiológica |
|---------|-------------|------------------------------|
| `cases_lag1` | Casos confirmados hace 1 semana (misma comuna) | Inercia epidémica de corto plazo |
| `cases_lag2` | Casos confirmados hace 2 semanas (misma comuna) | Período de incubación del dengue (4-10 días) |
| `cases_lag3` | Casos confirmados hace 3 semanas (misma comuna) | Dinámica de propagación comunitaria |
| `cases_lag4` | Casos confirmados hace 4 semanas (misma comuna) | Ciclo completo de transmisión |
| `incidencia_lag1` | Tasa de incidencia hace 1 semana (misma comuna) | Versión normalizada de cases_lag1 |
| `incidencia_lag2` | Tasa de incidencia hace 2 semanas (misma comuna) | Versión normalizada de cases_lag2 |
| `incidencia_lag3` | Tasa de incidencia hace 3 semanas (misma comuna) | Versión normalizada de cases_lag3 |
| `incidencia_lag4` | Tasa de incidencia hace 4 semanas (misma comuna) | Versión normalizada de cases_lag4 |

**Nota importante:** Los primeros 1 a 4 registros de cada comuna tienen NaN en estas columnas porque no existe semana anterior. En el entrenamiento estas filas se eliminan automáticamente o se imputan con cero según la estrategia del modelo.

---

## 4. Features climáticas base — 7 features

Variables climáticas de la semana actual. Son las mismas para las 15 comunas en cada semana (fuente única: Observatorio Central CABA, Open-Meteo ERA5).

| Feature | Descripción | Unidad | Fuente |
|---------|-------------|--------|--------|
| `temp_mean` | Temperatura media semanal | °C | Open-Meteo ERA5 |
| `precipitation` | Precipitación acumulada semanal | mm | Open-Meteo ERA5 |
| `humidity_mean` | Humedad relativa media semanal | % | Open-Meteo ERA5 |
| `heat_index_mean` | Índice de calor medio semanal | °C | Calculado — fórmula Rothfusz (NOAA, 1990) |
| `temp_mean_anomaly` | Desviación de temperatura respecto a la media histórica de esa semana | °C | Calculado |
| `precipitation_anomaly` | Desviación de precipitación respecto a la media histórica de esa semana | mm | Calculado |
| `humidity_mean_anomaly` | Desviación de humedad respecto a la media histórica de esa semana | % | Calculado |

**¿Qué es el índice de calor?**
Combina temperatura y humedad para representar cómo percibe el calor el cuerpo humano. Es más relevante para la reproducción del Aedes aegypti que la temperatura sola porque el mosquito responde al calor real percibido, no al que marca el termómetro.

**¿Qué son las anomalías?**
Miden si el clima fue inusualmente cálido, lluvioso o húmedo para esa época del año — independientemente de la estación. Una semana con 28°C en julio es mucho más anómala que en enero, aunque el valor absoluto sea el mismo. Las anomalías separan el efecto climático puro de la estacionalidad. Decisión técnica DT-11.

---

## 5. Features climáticas con lags — 28 features

El EDA mostró que el clima actual no predice bien los casos actuales (correlación débil con lag 0). El clima de semanas anteriores es más predictivo porque el mosquito tarda tiempo en reproducirse y transmitir el virus.

Del EDA con Spearman: temperatura media lag 4 semanas tiene r = +0.45, precipitación lag 1 semana tiene r = +0.25.

Se generan lags de 1 a 4 semanas para las 7 variables climáticas = **28 features**.

| Feature | Descripción | Lag óptimo según EDA |
|---------|-------------|---------------------|
| `temp_mean_lag1` a `temp_mean_lag4` | Temperatura media de 1 a 4 semanas atrás | Lag 4 (r=+0.45) |
| `precipitation_lag1` a `precipitation_lag4` | Precipitación de 1 a 4 semanas atrás | Lag 1 (r=+0.25) |
| `humidity_mean_lag1` a `humidity_mean_lag4` | Humedad de 1 a 4 semanas atrás | Lag 4 |
| `heat_index_mean_lag1` a `heat_index_mean_lag4` | Índice de calor de 1 a 4 semanas atrás | Lag 4 |
| `temp_mean_anomaly_lag1` a `temp_mean_anomaly_lag4` | Anomalía de temperatura de 1 a 4 semanas atrás | Todos relevantes |
| `precipitation_anomaly_lag1` a `precipitation_anomaly_lag4` | Anomalía de precipitación de 1 a 4 semanas atrás | Lag 1 |
| `humidity_mean_anomaly_lag1` a `humidity_mean_anomaly_lag4` | Anomalía de humedad de 1 a 4 semanas atrás | Todos relevantes |

**Nota:** Se incluyen todos los lags (1 a 4) para cada variable y se deja que el modelo determine cuáles son más importantes mediante la importancia de features. No se preselecciona un único lag por variable para evitar descartar información útil. Decisión técnica DT-12.

---

## 6. Features de estacionalidad — 4 features

La semana del año es el predictor más importante según el EDA — la diferencia entre el pico (SE13, 1,797 casos promedio) y el valle (SE25, 1 caso promedio) es de 1,797 veces.

| Feature | Descripción | Valores | Decisión técnica |
|---------|-------------|---------|-----------------|
| `semana_sin` | Seno de la semana epidemiológica | -1 a +1 | DT-09 |
| `semana_cos` | Coseno de la semana epidemiológica | -1 a +1 | DT-09 |
| `is_epidemic_season` | 1 si estamos en temporada alta (SE 1-17 o SE 48-52), 0 si no | 0 o 1 | DT-10 |
| `mes_aprox` | Mes aproximado del año (1-12) | 1 a 12 | Interpretabilidad |

**¿Por qué seno y coseno en lugar del número de semana directamente?**

Si usamos el número de semana (1 a 52) el modelo percibe que la SE52 y la SE1 están "lejos" (diferencia de 51 unidades). Pero en la realidad son semanas consecutivas — última semana de diciembre y primera de enero, ambas en pleno verano austral y período de riesgo.

Con seno y coseno cada semana queda representada como un punto en un círculo. La SE52 y la SE1 quedan casi en el mismo punto del círculo — el modelo las percibe como semanas cercanas, que es correcto epidemiológicamente.

**¿Por qué is_epidemic_season además del seno y coseno?**

La codificación cíclica captura la posición gradual en el año. `is_epidemic_season` da una señal binaria fuerte y directa — "estamos o no en temporada de riesgo". Los árboles de decisión pueden usar esta variable como condición primaria de forma muy eficiente. Ambas variables se complementan.

---

## 7. Features espaciales — 18 features

| Feature | Descripción | Uso recomendado |
|---------|-------------|----------------|
| `comuna_id` | ID numérico de la comuna (1 a 15) | XGBoost y Random Forest |
| `poblacion` | Población de la comuna según Censo 2022 INDEC | Contexto y normalización |
| `es_comuna_1` | 1 si es la Comuna 1, 0 si no | Todos los modelos |
| `comuna_1` | 1 si es la Comuna 1, 0 si no (One-Hot) | Modelos lineales y LSTM |
| `comuna_2` | 1 si es la Comuna 2, 0 si no (One-Hot) | Modelos lineales y LSTM |
| `comuna_3` a `comuna_15` | Ídem para cada comuna (One-Hot) | Modelos lineales y LSTM |

**¿Por qué dos representaciones de la misma variable?**

`comuna_id` como entero (1-15): los modelos de árboles (XGBoost, Random Forest) hacen cortes binarios del tipo "¿es la comuna 1? sí/no". No asumen que el número 15 vale más que el número 1. Para estos modelos el entero es suficiente y eficiente.

One-Hot Encoding (`comuna_1` a `comuna_15`): los modelos lineales sí interpretan los números como magnitudes con orden implícito. Sin One-Hot el modelo lineal asumiría que la Comuna 15 tiene mayor "peso" que la Comuna 1 por tener un número más alto, lo cual no tiene ningún sentido geográfico ni epidemiológico.

**¿Por qué es_comuna_1 además de las dummies?**

El EDA comparativo demostró que la Comuna 1 tiene un comportamiento estructuralmente diferente al resto:
- Concentra el 39.9% de todos los casos
- Tiene solo 57.1% de semanas con cero casos vs 69.8% del resto
- Su incidencia relativa fue 7.15 veces el promedio de CABA en 2024

`es_comuna_1` le da al modelo una señal explícita sobre este comportamiento atípico, sin que tenga que aprenderlo solo desde los datos.

---

## 8. Normalización

Se aplicó **StandardScaler** (media=0, desviación estándar=1) a todas las variables numéricas continuas. Las variables normalizadas tienen el sufijo `_norm`.

**Regla crítica:** el scaler se ajustó **únicamente con el conjunto de train** y se aplicó a los tres splits. Esto evita data leakage — el modelo no ve las estadísticas de validación ni test durante el entrenamiento.

El scaler guardado en `data/processed/scaler.pkl` permite normalizar datos nuevos en producción con los mismos parámetros.

**¿Qué modelos necesitan normalización?**
- Regresión lineal (baseline): sí
- LSTM: sí
- XGBoost: no (pero no perjudica tenerla)
- Random Forest: no (pero no perjudica tenerla)

---

## 9. Split temporal

| Partición | Período | Filas | Descripción |
|-----------|---------|-------|-------------|
| **Train** | 2023 completo | 780 | Año completo con brote moderado |
| **Validation** | 2024 SE 1-26 | 390 | Primer semestre incluyendo el brote histórico |
| **Test** | 2024 SE 27-52 + 2025 | 1,170 | Segundo semestre y año siguiente |

**¿Por qué este corte y no otro?**

El split respeta la regla fundamental de las series temporales: siempre entrenamos con el pasado y evaluamos con el futuro. Nunca el contrario.

El período de validación incluye la SE1 a SE26 de 2024 — que contiene el brote histórico más grande del período (pico en SE12 con 3,029 casos). Esto permite evaluar si el modelo puede anticipar un brote de gran magnitud habiendo entrenado solo con 2023.

El test incluye el segundo semestre de 2024 y todo 2025 — datos genuinamente futuros al momento de entrenamiento.

---

## 10. Archivos generados

| Archivo | Ubicación | Descripción |
|---------|-----------|-------------|
| `dataset_features.parquet` | `data/processed/` | Dataset completo con 67 features |
| `dataset_features.csv` | `data/processed/` | Copia CSV para inspección manual |
| `train.parquet` | `data/processed/` | Conjunto de entrenamiento (2023) |
| `validation.parquet` | `data/processed/` | Conjunto de validación (2024 S1) |
| `test.parquet` | `data/processed/` | Conjunto de test (2024 S2 + 2025) |
| `scaler.pkl` | `data/processed/` | StandardScaler ajustado con train |

---

## 11. Criterios de aceptación HU5 — verificación

| Criterio | Estado |
|----------|--------|
| Features de rezago temporal (casos t-1 a t-4 y clima 1-4 semanas) | ✅ |
| Promedios de variables climáticas en múltiples ventanas | ✅ |
| Variables de estacionalidad (semana_sin, semana_cos, is_epidemic_season) | ✅ |
| Normalización de features | ✅ |
| Split temporal train/validation/test | ✅ |
| Tasa de incidencia per cápita (DT-13) | ✅ |
| One-Hot Encoding para modelos lineales | ✅ |

**HU5 — COMPLETADA**

---

## 12. Próximos pasos — Sprint 5

Con las 65 features predictoras disponibles el Sprint 5 puede proceder con el modelado:

1. **Modelo baseline** — regresión lineal o media móvil como referencia mínima
2. **Random Forest** — modelo de árboles robusto, usa `comuna_id` como entero
3. **XGBoost** — modelo boosting de alto rendimiento, usa `comuna_id` como entero
4. **LSTM** — red neuronal recurrente, usa One-Hot Encoding y variables normalizadas

Las features más importantes según el EDA y las refferencias del proyecto son: `cases_lag1` a `cases_lag4`, `temp_mean_lag4`, `semana_sin`, `semana_cos` e `is_epidemic_season`.
