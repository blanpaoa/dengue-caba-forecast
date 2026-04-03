# Registro de Decisiones Técnicas — Sprint 3

**Proyecto:** Predicción espacio-temporal de brotes de dengue en CABA  
**Sprint:** 3 — Análisis Exploratorio de Datos (EDA)
**Responsable:** Ing. Paola Andrea Blanco Blanco  

---

## DT-09: Codificación de la semana epidemiológica como variable cíclica

**Contexto**

El EDA confirmó que la semana del año es el predictor más importante de la estacionalidad del dengue. La SE13 (~marzo) tiene en promedio 1,797 casos mientras que la SE25 (~junio) tiene 1 caso. Sin embargo, si codificamos la semana como un número entero del 1 al 52, el modelo percibiría que la SE52 y la SE1 están "lejos" cuando en realidad son semanas consecutivas.

**Opciones evaluadas**

| Opción | Ventaja | Desventaja |
|--------|---------|------------|
| Semana como entero (1-52) | Simple | La SE52 y SE1 parecen lejanas |
| One-hot encoding (52 columnas) | Exacto | Demasiadas columnas, sobreajuste |
| Codificación cíclica (seno + coseno) | Captura la ciclicidad | Levemente más complejo |

**Decisión**

Usar codificación cíclica con seno y coseno:

```python
semana_sin = sin(2π × epi_week / 52)
semana_cos = cos(2π × epi_week / 52)
```

**Justificación**

Con seno y coseno, la SE52 y la SE1 quedan matemáticamente cerca en el espacio de features porque ambas tienen valores similares de seno y coseno. Esto permite que el modelo aprenda correctamente que diciembre y enero son épocas similares para el dengue.

**Implicación para el Sprint 4**

Agregar `semana_sin` y `semana_cos` como features en `src/features/lags.py`.

---

## DT-10: Creación de variable binaria de temporada epidémica

**Contexto**

El EDA mostró que la actividad del dengue se concentra casi exclusivamente en las semanas 1 a 17 (enero-abril). Entre la SE18 y la SE47 los casos son prácticamente cero en todos los años analizados.

**Decisión**

Crear una variable binaria `is_epidemic_season`:

```python
is_epidemic_season = 1  si epi_week <= 17 o epi_week >= 48
is_epidemic_season = 0  en caso contrario
```

**Justificación**

Esta variable le da al modelo una señal muy clara sobre el período de riesgo. Complementa la codificación cíclica — mientras seno y coseno capturan la posición gradual en el año, `is_epidemic_season` provee una señal binaria fuerte que el modelo puede usar directamente como condición en los árboles de decisión.

**Evidencia del EDA**

La diferencia entre temporada alta (promedio 1,797 casos en el pico) y baja (promedio 1 caso en invierno) es de 1,800 veces. Ninguna otra variable tiene tanta capacidad discriminativa.

---

## DT-11: Uso de anomalías climáticas como features primarias

**Contexto**

La matriz de correlaciones del EDA reveló que las variables climáticas absolutas (temperatura media, humedad) tienen correlaciones bajas con los casos de dengue (r = 0.06 a 0.12 con lag 0). Sin embargo, esta baja correlación puede deberse a que la temperatura y la humedad están confundidas con la estacionalidad — en verano hay más temperatura Y más dengue, pero no necesariamente por causalidad directa.

**Hallazgo del EDA**

La humedad mostró correlación negativa con lag 4 (r = -0.14), lo cual es contraintuitivo. Una explicación plausible es que en invierno la humedad es alta y la temperatura baja, y esa combinación inhibe al mosquito. El modelo podría estar aprendiendo la estacionalidad a través de la humedad en lugar del efecto directo de la humedad sobre el mosquito.

**Decisión**

Priorizar las **anomalías climáticas** sobre los valores absolutos como features principales para el modelo:

- `temp_mean_anomaly` en lugar de solo `temp_mean`
- `precipitation_anomaly` en lugar de solo `precipitation`
- `humidity_mean_anomaly` en lugar de solo `humidity_mean`

Incluir también los valores absolutos pero con menor prioridad.

**Justificación**

Las anomalías miden si el clima fue más caluroso, más lluvioso o más húmedo de lo normal para esa época del año. Esto separa el efecto climático puro del efecto estacional. Por ejemplo:

- Temperatura de 28°C en febrero = normal (anomalía ≈ 0)
- Temperatura de 28°C en julio = muy anormal (anomalía ≈ +16°C)

La segunda situación es mucho más relevante para predecir un brote fuera de temporada. Las anomalías capturan este efecto; los valores absolutos no.

**Respaldo en la literatura**

Sebastianelli et al. (2024) usan anomalías climáticas como features en su modelo de predicción de dengue, obteniendo mejor performance que con valores absolutos.

---

## DT-12: Lags óptimos por variable climática

**Contexto**

El análisis de correlaciones con rezagos del EDA identificó que el lag óptimo varía según la variable climática. Usar el lag incorrecto reduce la capacidad predictiva.

**Resultados del EDA**

| Variable | Lag óptimo | Correlación máxima | Interpretación biológica |
|----------|-----------|-------------------|--------------------------|
| Temperatura media | Lag 4w | r = +0.45 | El ciclo completo mosquito-virus-persona tarda ~4 semanas |
| Precipitación | Lag 1w | r = +0.25 | La lluvia genera criaderos en ~1 semana |
| Humedad relativa | Lag 4w | r = -0.14 | Resultado contraintuitivo — ver DT-11 |

**Decisión**

Incluir múltiples lags (1 a 4 semanas) para cada variable climática y dejar que el modelo determine cuáles son más importantes mediante la importancia de features. No preseleccionar un único lag por variable.

**Justificación**

Con solo 156 semanas de datos, preseleccionar lags basándose en correlaciones lineales puede descartar información útil. Los modelos de árboles (XGBoost, Random Forest) son robustos ante features redundantes y pueden seleccionar automáticamente los lags más informativos durante el entrenamiento.

---

## DT-13: Incorporación de población por comuna como feature

**Contexto**

El EDA reveló que la Comuna 4 aparece como tercera más afectada en términos absolutos (2,421 casos) pero su incidencia relativa es de 0.56 en 2023 y 1.21 en 2024 — valores moderados. La diferencia entre ranking absoluto y relativo sugiere que el tamaño poblacional de la comuna influye en el número de casos pero no está capturado en el dataset actual.

**Problema identificado**

Si el modelo aprende que "la Comuna 4 tiene muchos casos" sin saber que tiene mucha población, puede generar predicciones incorrectas. Una comuna pequeña con alta incidencia per cápita es más preocupante epidemiológicamente que una comuna grande con muchos casos pero baja incidencia per cápita.

**Decisión**

Incorporar en el Sprint 4 la población estimada de cada comuna (fuente: Censo 2022, INDEC) y calcular:

```python
incidencia_por_10000 = confirmed_cases / (poblacion_comuna / 10000)
```

Esta tasa de incidencia normalizada permite comparar comunas de diferente tamaño de forma justa.

**Fuente de datos**

Censo Nacional de Población, Hogares y Viviendas 2022 — INDEC. Datos disponibles por comuna de CABA en indec.gob.ar. No requiere descarga automatizada — son valores fijos que pueden incorporarse como constantes en el código.

**Limitación reconocida**

La población del Censo 2022 es una foto de ese año. Para 2023-2025 puede haber cambios menores. Para un trabajo académico esta aproximación es suficiente.

---

## DT-14: Features de autocorrelación temporal (lags de casos)

**Contexto**

El EDA no incluyó explícitamente el análisis de autocorrelación temporal de los casos, pero los gráficos de series temporales muestran claramente que los casos tienen fuerte inercia — si esta semana hay muchos casos, la semana siguiente probablemente también habrá muchos.

**Hipótesis**

Los lags de casos (casos en t-1, t-2, t-3, t-4 semanas) son probablemente los predictores más importantes del modelo — más que las variables climáticas. Esto se debe a que los casos pasados capturan tanto la dinámica epidémica actual como la estacionalidad implícitamente.

**Decisión**

Incluir como features obligatorios en el Sprint 4:

```python
cases_lag1 = confirmed_cases desplazado 1 semana hacia adelante
cases_lag2 = confirmed_cases desplazado 2 semanas hacia adelante
cases_lag3 = confirmed_cases desplazado 3 semanas hacia adelante
cases_lag4 = confirmed_cases desplazado 4 semanas hacia adelante
```

**Advertencia importante**

Estos lags deben calcularse **por comuna** — los casos de la semana pasada en la Comuna 1 predicen los casos de esta semana en la Comuna 1, no en la Comuna 9. Es un error común en series temporales espaciales usar lags sin respetar la dimensión espacial.

---

## Resumen de decisiones del Sprint 3

| Decisión | Impacto en Sprint 4 | Prioridad |
|----------|---------------------|-----------|
| DT-09: Codificación cíclica de semana | Agregar semana_sin, semana_cos | Alta |
| DT-10: Variable is_epidemic_season | Agregar feature binario | Alta |
| DT-11: Priorizar anomalías climáticas | Reordenar features en pipeline | Alta |
| DT-12: Múltiples lags climáticos | Incluir lags 1-4 para cada variable | Media |
| DT-13: Población por comuna | Incorporar datos del Censo 2022 | Media |
| DT-14: Lags de casos por comuna | Agregar cases_lag1 a cases_lag4 | Alta |
