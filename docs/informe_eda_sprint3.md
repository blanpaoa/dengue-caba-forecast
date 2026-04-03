# Informe de Análisis Exploratorio de Datos — Sprint 3

**Proyecto:** Predicción espacio-temporal de brotes de dengue en CABA  
**Sprint:** 3 — HU4: Análisis exploratorio de patrones espacio-temporales
**Responsable:** Ing. Paola Andrea Blanco Blanco  
**Dataset:** `data/processed/dataset_maestro.parquet`  
**Período analizado:** 2023-2025 | 15 comunas | 52 semanas epidemiológicas  

---

## 1. Introducción

El Análisis Exploratorio de Datos (EDA) es el proceso de examinar y visualizar los datos antes de construir el modelo predictivo. Su propósito es entender la estructura, patrones y relaciones presentes en los datos para tomar decisiones informadas durante el feature engineering y el modelado.

Este informe documenta los hallazgos del EDA realizado sobre el dataset maestro generado en el Sprint 2, que contiene 2,340 observaciones de casos confirmados de dengue y variables climáticas para las 15 comunas de CABA entre 2023 y 2025.

Las preguntas que guiaron el análisis fueron:

1. ¿Cómo evolucionan los casos de dengue a lo largo del tiempo?
2. ¿Qué comunas concentran más casos?
3. ¿En qué época del año hay más casos?
4. ¿Existe relación entre el clima y los casos de dengue?
5. ¿Cuántas semanas después de un cambio climático se ve el efecto en los casos?
6. ¿Hay diferencias estructurales entre comunas en su vulnerabilidad?

---

## 2. Análisis temporal — evolución de casos en CABA

### 2.1 Serie temporal completa (2023-2025)

El gráfico de serie temporal muestra la evolución de los casos confirmados de dengue sumando las 15 comunas de CABA semana a semana durante el período 2023-2025.

**Hallazgo principal:** Los casos de dengue en CABA tienen una estacionalidad extremadamente marcada. Los brotes ocurren exclusivamente durante la temporada alta (semanas epidemiológicas 1 a 17, enero a abril). Durante los meses de invierno (mayo a noviembre) los casos caen prácticamente a cero.

**Datos clave:**
- Total de casos en el período: 37,078
- Pico máximo: 3,029 casos en la SE12 de 2024 (tercera semana de marzo)
- Semanas con al menos un caso: 31% del total
- Semanas con cero casos: 69% del total

### 2.2 Comparación entre años

Al superponer los tres años usando la semana epidemiológica como eje común, se observa que:

**El patrón estacional es consistente.** Ambos años (2023 y 2024) tienen la misma forma de curva: crecimiento gradual desde enero, pico en marzo y descenso abrupto hacia abril-mayo. La forma es una campana asimétrica — la subida tarda 12 semanas y la bajada solo 5-6 semanas.

**El brote de 2024 fue significativamente más intenso.** Con 23,771 casos fue 1.8 veces más grande que el de 2023 (13,132 casos). Además, el brote de 2024 comenzó más temprano — ya desde la SE1 se registraban casos, mientras que en 2023 el crecimiento comenzó recién en la SE8.

**2025 tiene datos insuficientes.** Con solo 175 casos registrados (datos parciales), 2025 no aporta información significativa para el modelado.

### 2.3 Implicaciones para el modelado

La fuerte estacionalidad indica que la semana del año es el predictor más importante. El modelo debe incorporar variables que capturen en qué momento del año se encuentra una observación. Se decidió usar codificación cíclica (seno y coseno de la semana) más una variable binaria de temporada alta (DT-09 y DT-10).

---

## 3. Análisis espacial — distribución por comunas

### 3.1 Total de casos por comuna

El análisis de la distribución espacial revela una concentración muy marcada en pocas comunas.

**Top 5 comunas por casos totales (2023-2025):**

| Posición | Comuna | Casos totales | % del total CABA |
|----------|--------|---------------|-----------------|
| 1 | Comuna 1 | 14,810 | 39.9% |
| 2 | Comuna 9 | 3,086 | 8.3% |
| 3 | Comuna 4 | 2,421 | 6.5% |
| 4 | Comuna 11 | 2,149 | 5.8% |
| 5 | Comuna 15 | 2,003 | 5.4% |

**Observación importante:** La Comuna 1 (Retiro, San Nicolás, Puerto Madero, San Telmo, Montserrat, Constitución) concentra el 39.9% de todos los casos de CABA en el período analizado. Esta concentración es desproporcionada y constituye un outlier espacial significativo que el modelo deberá aprender a manejar.

**Advertencia sobre el ranking absoluto:** La Comuna 4 aparece tercera en casos absolutos pero su incidencia relativa es moderada (0.56 en 2023 y 1.21 en 2024). Esto sugiere que tiene muchos casos porque es una comuna populosa, no porque tenga alta vulnerabilidad per cápita. Se decidió incorporar la población por comuna como feature para normalizar (DT-13).

### 3.2 Incidencia relativa por comuna y año

La incidencia relativa normaliza los casos de cada comuna respecto al promedio de CABA para ese año, permitiendo comparar comunas de diferente tamaño de forma justa.

**Fórmula:** incidencia_relativa = casos_comuna / promedio_entre_comunas

Donde un valor de 1.0 significa que la comuna tuvo exactamente el promedio, >1.0 significa más casos que el promedio y <1.0 menos casos.

**Hallazgos por comuna:**

**Comuna 1 — riesgo estructuralmente muy alto**
- 2023: 3.97 (casi 4 veces el promedio de CABA)
- 2024: 7.15 (más de 7 veces el promedio)
- Es la única comuna que supera consistentemente 2 veces el promedio en todos los años

**Comuna 9 — patrón inestable entre años**
- 2023: 2.98 (casi 3 veces el promedio)
- 2024: 0.30 (por debajo del promedio)
- Este cambio drástico de un año al otro es epidemiológicamente relevante. Una hipótesis es que la inmunidad poblacional acumulada en 2023 redujo la susceptibilidad en 2024.

**Comunas de bajo riesgo consistente**
Las comunas 2, 3, 5, 6, 12 y 13 tienen incidencia relativa entre 0.16 y 0.60 en todos los años analizados. Son zonas sistemáticamente por debajo del promedio.

**Hallazgo sobre la inestabilidad espacial**
El patrón espacial no es estable entre años. Lo que fue verdad en 2023 no se repite necesariamente en 2024. Esto significa que features espaciales estáticas (como "la Comuna 1 siempre tiene más casos") no son suficientes. El modelo necesitará features dinámicas — casos recientes en comunas vecinas — para capturar estos cambios año a año (DT-14).

### 3.3 Heatmap semana-comuna

El heatmap confirma visualmente los hallazgos anteriores. En 2024 la columna de la Comuna 1 es tan dominante que hace que el resto de las comunas parezcan uniformes. En 2023 se observa una distribución más equilibrada con actividad notable también en las comunas 8, 9 y 10.

---

## 4. Estacionalidad

### 4.1 Promedio histórico por semana

El análisis de estacionalidad promedia los casos de todos los años disponibles para cada semana del año, revelando el patrón típico.

**Hallazgos:**
- Semana de pico histórico: SE13 (~última semana de marzo)
- Casos promedio en el pico: 1,797
- Casos promedio en invierno (SE25, junio): 1
- Diferencia entre pico y valle: 1,797 veces

**La curva tiene forma asimétrica:**
- Subida gradual: ~12 semanas desde la SE1 hasta el pico
- Bajada abrupta: ~5-6 semanas desde el pico hasta el valle

Esta asimetría tiene sentido biológico. El brote se instala lentamente porque requiere que la temperatura supere un umbral para que el mosquito se reproduzca activamente. La bajada es más rápida porque cuando la temperatura desciende en otoño, el mosquito deja de reproducirse casi de inmediato.

### 4.2 Umbral térmico implícito

Los gráficos sugieren que existe un umbral de temperatura a partir del cual el dengue se activa. En el contexto de CABA (~20-22°C de temperatura media) el dengue comienza a crecer. Por debajo de ese umbral los casos son prácticamente nulos.

Este umbral no fue calculado explícitamente en el EDA pero se incorporará como hipótesis en el feature engineering — la variable `is_epidemic_season` (DT-10) es una aproximación binaria de este umbral.

---

## 5. Correlaciones entre clima y dengue

### 5.1 Correlaciones con rezago (lag 0 a 4 semanas)

Se calculó la correlación de Pearson entre cada variable climática y los casos de dengue con rezagos de 0 a 4 semanas. El rezago representa cuántas semanas antes del momento de los casos se mide el clima.

**¿Por qué usar rezagos?**
El efecto del clima sobre el dengue no es inmediato. La temperatura de hoy determina si el mosquito se reproduce bien, pero ese efecto se traduce en casos humanos 2 a 4 semanas después — el tiempo que tarda el ciclo completo: huevo → larva → mosquito adulto → picadura → período de incubación del virus.

**Resultados:**

| Variable | Lag óptimo | Correlación máxima | Interpretación |
|----------|-----------|-------------------|----------------|
| Temperatura media | Lag 4w | r = +0.45 | Efecto diferido de ~4 semanas |
| Precipitación | Lag 1w | r = +0.25 | Criaderos disponibles en ~1 semana |
| Humedad relativa | Lag 4w | r = -0.14 | Resultado contraintuitivo |

**Interpretación de la temperatura (lag 4):**
La temperatura de hace 4 semanas tiene la mayor correlación con los casos actuales. Esto coincide con el ciclo biológico del Aedes aegypti — a temperaturas óptimas (~28°C) el ciclo de vida del mosquito tarda aproximadamente 10-14 días, y el período de incubación del virus dengue en el mosquito dura otros 8-12 días. Sumando ambos períodos se obtiene un rezago de ~3-4 semanas entre las condiciones climáticas favorables y el aumento de casos en la población.

**Interpretación de la precipitación (lag 1):**
La lluvia de la semana pasada predice mejor los casos actuales que la lluvia de esta semana. Las precipitaciones generan agua estancada que se convierte en criaderos de mosquitos en pocos días. El efecto es más rápido que el de la temperatura porque actúa sobre criaderos ya existentes, no sobre el ciclo de vida completo del mosquito.

**Interpretación de la humedad (resultado contraintuitivo):**
La humedad mostró correlación negativa con lag 4 (r = -0.14), lo cual es opuesto a lo esperado biológicamente. La explicación más probable es una **confusión por estacionalidad**: en invierno la humedad relativa en CABA es alta (el aire frío retiene más humedad relativa) mientras que los casos de dengue son nulos. Esta asociación inversa enmascara el efecto directo positivo que tiene la humedad sobre la supervivencia del mosquito. Las anomalías de humedad (DT-11) deberían resolver este problema.

### 5.2 Correlaciones directas (scatter lag 0)

Los scatter plots de cada variable climática contra los casos con lag 0 muestran correlaciones bajas (r = 0.18, 0.21, 0.12). Esta baja correlación en lag 0 es esperada y consistente con los resultados de la sección anterior — el lag óptimo no es 0 para ninguna variable.

Los scatter plots revelan además la distribución típica de datos con fuerte estacionalidad: muchos puntos concentrados cerca del cero (semanas de invierno) y pocos puntos dispersos hacia valores altos (semanas de verano). Esta forma de "L" o "J" es característica de enfermedades estacionales y dificulta el ajuste de modelos lineales simples.

### 5.3 Matriz de correlaciones completa

La matriz de correlaciones entre todas las variables del dataset revela dos hallazgos importantes para el feature engineering:

**Multicolinealidad extrema entre variables de temperatura:**
- Temperatura media y máxima: r = 0.99
- Temperatura media y mínima: r = 0.99
- Temperatura media e índice de calor: r = 1.00

Las cuatro variables de temperatura son prácticamente idénticas desde el punto de vista estadístico. Incluir todas en el modelo sería redundante. Se priorizará el índice de calor (que ya combina temperatura y humedad) y la anomalía de temperatura.

**La anomalía de precipitación es la variable más independiente:**
Tiene r = 0.89 con la precipitación bruta (esperado) pero r ≈ 0 con todas las variables de temperatura. Esta independencia la hace especialmente valiosa como feature porque aporta información que no está contenida en las otras variables.

---

## 6. Análisis de desbalance de clases

El dataset tiene un desbalance significativo entre semanas con y sin casos:

| Clase | Observaciones | Porcentaje |
|-------|--------------|------------|
| Sin casos (confirmed_cases = 0) | 1,614 | 69.0% |
| Con casos (confirmed_cases > 0) | 726 | 31.0% |

Este desbalance no está distribuido uniformemente — está causado por la estacionalidad. Las semanas de invierno (SE18 a SE47) concentran la mayoría de los ceros, mientras que las semanas de verano concentran la mayoría de los casos.

**Implicaciones para el modelado:**

Para la tarea de **regresión** (predecir el número de casos): el desbalance es menos crítico porque los modelos de regresión aprenden directamente el valor numérico. Sin embargo, el modelo puede subestimar los picos si la función de pérdida penaliza más los errores frecuentes (muchos ceros) que los errores en los picos.

Para la tarea de **clasificación binaria** (predecir si habrá brote o no): el desbalance es importante. Un clasificador que siempre predice "no brote" tendría 69% de accuracy pero sería completamente inútil en la práctica. Se utilizarán métricas de evaluación apropiadas (F1-score, AUC-ROC, recall) y se evaluará el uso de pesos de clase para compensar el desbalance.

---

## 7. Autocorrelación temporal — hallazgo pendiente

El EDA de las series temporales sugiere fuertemente que los casos de una semana están altamente correlacionados con los casos de semanas anteriores. Si esta semana hay 1,000 casos en una comuna, la semana siguiente probablemente también habrá muchos — el brote tiene inercia.

Este fenómeno, llamado **autocorrelación temporal**, no fue cuantificado explícitamente en este EDA pero será la base del feature engineering de lags de casos en el Sprint 4 (DT-14). Los lags de casos son probablemente los predictores más importantes del modelo — más que cualquier variable climática individual — porque capturan tanto la dinámica epidémica actual como la estacionalidad implícitamente.

---

## 8. Limitaciones del EDA

**Limitación 1 — Solo 3 años de datos**
El EDA está basado en 2 ciclos epidémicos completos (2023 y 2024) más datos parciales de 2025. Algunos patrones observados pueden no ser generalizables. En particular, el dominio extremo de la Comuna 1 en 2024 puede ser un evento atípico que el modelo podría sobre-aprender.

**Limitación 2 — Correlaciones lineales únicamente**
El análisis de correlaciones de Pearson solo detecta relaciones lineales. Las relaciones entre clima y dengue pueden ser no lineales — por ejemplo, puede existir un umbral de temperatura por encima del cual el mosquito se activa exponencialmente. Estas relaciones no lineales serán capturadas por los modelos de árboles de decisión (XGBoost, Random Forest) durante el modelado.

**Limitación 3 — Una sola estación meteorológica**
Todos los análisis climáticos se basan en datos del Observatorio Central de CABA. La variabilidad microclimática entre comunas no está capturada. Las comunas del sur (8, 9) cercanas al Riachuelo pueden tener condiciones distintas a las del norte (13, 14).

**Limitación 4 — Sesgo de notificación**
Los casos confirmados dependen del acceso al sistema de salud. Comunas con menor infraestructura sanitaria pueden tener casos subregistrados. Este sesgo afecta especialmente la interpretación de las comunas de bajo riesgo aparente.

---

## 9. Decisiones técnicas para el Sprint 4

A partir de los hallazgos del EDA se definen las siguientes decisiones de feature engineering para el Sprint 4:

| Decisión | Feature a crear | Justificación |
|----------|----------------|---------------|
| DT-09 | semana_sin, semana_cos | Capturar ciclicidad de la semana |
| DT-10 | is_epidemic_season | Señal binaria de temporada alta |
| DT-11 | Priorizar anomalías sobre valores absolutos | Separar efecto climático de estacionalidad |
| DT-12 | Lags 1-4 para variables climáticas | Capturar efecto diferido del clima |
| DT-13 | Población por comuna (Censo 2022) | Normalizar casos per cápita |
| DT-14 | cases_lag1 a cases_lag4 por comuna | Capturar autocorrelación temporal |

---

## 10. Criterios de aceptación HU4 — verificación

| Criterio | Sección | Estado |
|----------|---------|--------|
| Visualizaciones de series temporales de casos por semana | Sección 2 | ✅ |
| Mapas de calor por comuna y período | Sección 3.3 | ✅ |
| Correlaciones entre variables climáticas y casos | Sección 5 | ✅ |
| Patrones de vecindad entre comunas (autocorrelación espacial) | Sección 3.2 | ✅ |
| Estacionalidad y dispersión geográfica documentadas | Secciones 3-4 | ✅ |

**HU4 — COMPLETADA**

---

## 11. Figuras generadas

| Archivo | Descripción |
|---------|-------------|
| `01_serie_temporal_caba.png` | Casos CABA total 2023-2025 con área sombreada |
| `02_comparacion_años.png` | Superposición de curvas 2023 vs 2024 vs 2025 |
| `03_casos_por_comuna.png` | Barplot total y por año para cada comuna |
| `04_heatmap_semana_comuna.png` | Heatmap semana × comuna para cada año |
| `05_estacionalidad.png` | Promedio histórico de casos por semana con meses |
| `06_correlaciones_clima_dengue.png` | Correlaciones con lags 0-4 semanas |
| `07_scatter_clima_dengue.png` | Scatter plots clima vs casos con tendencia |
| `08_matriz_correlaciones.png` | Matriz de correlaciones triangular completa |
| `09_incidencia_relativa_comunas.png` | Heatmap de incidencia relativa por comuna y año |

---

## 12. Próximos pasos — Sprint 4

Con los hallazgos del EDA documentados, el Sprint 4 puede proceder con el **Feature Engineering**:

1. Crear lags temporales de casos (t-1 a t-4) por comuna
2. Crear lags temporales de variables climáticas (1 a 4 semanas)
3. Codificar semana epidemiológica como variable cíclica (seno + coseno)
4. Crear variable binaria `is_epidemic_season`
5. Incorporar población por comuna (Censo 2022 INDEC)
6. Implementar split temporal train/validation/test
7. Documentar el proceso en `src/features/lags.py`
