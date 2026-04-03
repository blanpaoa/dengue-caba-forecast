# Reporte de Calidad de Datos — Sprint 2

**Proyecto:** Predicción espacio-temporal de brotes de dengue en CABA  
**Sprint:** 2 — Limpieza y validación de datos (HU3) 
**Responsable:** Ing. Paola Andrea Blanco Blanco  

---

## 1. Descripción del dataset maestro

El dataset maestro resulta de la unificación de dos fuentes procesadas en el Sprint 1:

| Dataset origen | Archivo | Filas | Columnas |
|----------------|---------|-------|----------|
| Datos epidemiológicos (dengue) | `dengue_weekly_comuna.parquet` | 2,340 | 4 |
| Datos climáticos (Open-Meteo ERA5) | `clima_caba_semanal.parquet` | 210 | 13 |
| **Dataset maestro unificado** | **`dataset_maestro.parquet`** | **2,340** | **13** |

**Clave de unión:** `year` + `epi_week` (LEFT JOIN desde dengue hacia clima)

---

## 2. Estructura del dataset maestro

| Columna | Tipo | Descripción | Fuente |
|---------|------|-------------|--------|
| `year` | int64 | Año epidemiológico | Epidemiológica |
| `epi_week` | int64 | Semana epidemiológica (1-52) | Epidemiológica |
| `comuna_id` | int64 | ID de comuna CABA (1-15) | Epidemiológica |
| `confirmed_cases` | int64 | Casos confirmados de dengue | Epidemiológica |
| `temp_max_mean` | float64 | Promedio semanal de temperatura máxima (°C) | Climática |
| `temp_min_mean` | float64 | Promedio semanal de temperatura mínima (°C) | Climática |
| `temp_mean` | float64 | Temperatura media semanal (°C) | Climática |
| `precipitation` | float64 | Precipitación acumulada semanal (mm) | Climática |
| `humidity_mean` | float64 | Humedad relativa media semanal (%) | Climática |
| `heat_index_mean` | float64 | Índice de calor medio semanal (°C) | Derivada |
| `temp_mean_anomaly` | float64 | Anomalía de temperatura respecto a media histórica (°C) | Derivada |
| `precipitation_anomaly` | float64 | Anomalía de precipitación respecto a media histórica (mm) | Derivada |
| `humidity_mean_anomaly` | float64 | Anomalía de humedad respecto a media histórica (%) | Derivada |

---

## 3. Completitud del dataset

| Dimensión | Valor | Esperado | Estado |
|-----------|-------|----------|--------|
| Filas totales | 2,340 | 2,340 (15×52×3) | ✅ |
| Comunas cubiertas | 15 / 15 | 15 | ✅ |
| Años cubiertos | 2023, 2024, 2025 | 3 años | ✅ |
| Semanas epidemiológicas | 52 / 52 | 52 | ✅ |
| Valores faltantes | 0 | 0 | ✅ |
| Duplicados | 0 | 0 | ✅ |

---

## 4. Valores faltantes

**No se encontraron valores faltantes** en el dataset maestro.

Esto se debe a:
- Los datos epidemiológicos fueron completados con 0 en las semanas sin casos durante el Sprint 1
- Los datos climáticos de ERA5 (Open-Meteo) no tienen gaps por ser un modelo de reanálisis
- La alineación temporal fue exitosa: las 156 semanas del período epidemiológico (2023-2025) tienen cobertura climática completa

**Estrategia de imputación implementada (por precaución):**
Interpolación lineal para variables climáticas, aplicable en futuras actualizaciones del dataset si se incorporaran datos del SMN con posibles gaps.

---

## 5. Duplicados

**No se encontraron duplicados** en la clave `(year, epi_week, comuna_id)`.

---

## 6. Validación de rangos

Todos los valores están dentro de los rangos físicamente válidos:

| Variable | Rango válido | Mínimo observado | Máximo observado | Estado |
|----------|-------------|-----------------|-----------------|--------|
| `confirmed_cases` | [0, 5000] | 0 | 1,391 | ✅ |
| `temp_max_mean` | [-5, 45] | ~12°C | ~37.7°C | ✅ |
| `temp_min_mean` | [-5, 45] | ~-0.7°C | ~27°C | ✅ |
| `temp_mean` | [-5, 45] | ~5°C | ~30°C | ✅ |
| `precipitation` | [0, 500] | 0 | ~179.3 mm | ✅ |
| `humidity_mean` | [0, 100] | ~30% | ~90% | ✅ |
| `heat_index_mean` | [-5, 60] | ~5°C | ~31.2°C | ✅ |

---

## 7. Estadísticas descriptivas

### 7.1 Variable objetivo — casos confirmados de dengue

| Estadístico | Valor |
|-------------|-------|
| Total de casos en el período | 37,078 |
| Media por semana/comuna | 15.85 casos |
| Máximo en una semana/comuna | 1,391 casos |
| Semanas con al menos 1 caso | 726 de 2,340 (31.0%) |
| Semanas con cero casos | 1,614 de 2,340 (69.0%) |

**Distribución por año:**

| Año | Casos confirmados | % del total |
|-----|-------------------|-------------|
| 2023 | 13,132 | 35.4% |
| 2024 | 23,771 | 64.1% |
| 2025 | 175 | 0.5% |

**Nota:** 2024 concentra el 64% de los casos totales del período, correspondiente al brote epidémico histórico de Argentina. 2025 tiene datos parciales (semanas 1-45 aproximadamente).

**Top 5 semanas de mayor incidencia (CABA total):**

| Año | Semana epidemiológica | Casos CABA total | Período calendario |
|-----|-----------------------|-----------------|-------------------|
| 2024 | SE12 | 3,029 | 18-24 marzo 2024 |
| 2024 | SE13 | 2,894 | 25-31 marzo 2024 |
| 2024 | SE14 | 2,606 | 1-7 abril 2024 |
| 2024 | SE11 | 2,602 | 11-17 marzo 2024 |
| 2023 | SE13 | 2,479 | 27 mar - 2 abr 2023 |

### 7.2 Variables climáticas

| Variable | Media | Mín | Máx |
|----------|-------|-----|-----|
| Temperatura media (°C) | 18.2 | ~5 | ~30 |
| Precipitación semanal (mm) | 17.0 | 0 | ~179 |
| Humedad relativa (%) | 67.4 | ~30 | ~90 |
| Índice de calor (°C) | ~19 | ~5 | ~31 |

---

## 8. Análisis de desbalance de clases

El 69% de las filas tiene `confirmed_cases = 0`. Esto representa un **desbalance significativo** para la tarea de clasificación binaria (brote / no brote) que se implementará en el Sprint 5.

**Implicaciones para el modelado:**
- Los modelos de clasificación tenderán a predecir siempre "sin brote" si no se maneja este desbalance
- Se evaluarán técnicas de manejo de desbalance en el Sprint 4-5: pesos de clase, umbral de clasificación ajustado, métricas de evaluación adecuadas (F1, AUC-ROC en lugar de accuracy)
- Este hallazgo se documenta como limitación en la memoria del proyecto

---

## 9. Observación sobre la dimensión espacial

Las variables climáticas son **idénticas para las 15 comunas en cada semana** porque se usa una única estación de referencia (Observatorio Central CABA). Esto es una limitación reconocida y documentada en DT-05 del registro de decisiones técnicas del Sprint 1.

En el Sprint 4 (Feature Engineering) se agregarán variables espaciales que diferenciarán las comunas entre sí: casos en comunas vecinas, matriz de vecindad.

---

## 10. Criterios de aceptación HU3 — verificación

| Criterio | Estado |
|----------|--------|
| Valores faltantes identificados y documentados | ✅ |
| Proceso de imputación implementado y justificado | ✅ |
| Rangos de valores validados para todas las variables | ✅ |
| Formatos de fecha y códigos de comuna estandarizados | ✅ |
| Estadísticas descriptivas de calidad por fuente generadas | ✅ |
| Porcentajes de completitud por variable y período documentados | ✅ |

**HU3 — COMPLETADA**

---

## 11. Archivos generados

| Archivo | Ubicación | Descripción |
|---------|-----------|-------------|
| `dataset_maestro.parquet` | `data/processed/` | Dataset unificado y limpio para modelado |
| `dataset_maestro.csv` | `data/processed/` | Copia CSV para inspección manual |
| `cleaning.py` | `src/data/` | Script de unificación y limpieza reproducible |

---

## 12. Próximos pasos — Sprint 3

Con el dataset maestro validado y limpio, el Sprint 3 puede proceder con el **Análisis Exploratorio de Datos (EDA)** sobre `dataset_maestro.parquet`:

- Visualizaciones de series temporales por semana epidemiológica
- Mapas de calor de casos por comuna y período
- Correlaciones entre variables climáticas y casos de dengue
- Análisis de autocorrelación espacial entre comunas
- Documentación de patrones de estacionalidad y dispersión geográfica

---

## 13. Limitación crítica — suficiencia de datos históricos

### 13.1 Situación actual

El dataset maestro cubre **3 años epidemiológicos** (2023-2025), lo que representa **156 semanas únicas** de observación. Si bien el dataset tiene 2,340 filas, desde la perspectiva del modelado de series temporales lo relevante es la longitud de la serie temporal, no el número de filas totales.

### 13.2 Comparación con estándares de otros proyectos

| Estudio de referencia | Años de datos | Unidad espacial |
|----------------------|---------------|-----------------|
| Sebastianelli et al. (2024) — Brasil/Perú | 19 años (2001-2019) | 27 estados |
| Roster et al. (2022) — Brasil | 10 años | 790 ciudades |
| Salim et al. (2021) — Malasia | 8 años | Nacional |
| **Este proyecto — CABA** | **3 años (2023-2025)** | **15 comunas** |

Las referencias recomienda un mínimo de 5 años para capturar ciclos inter-epidémicos del dengue. Con 3 años solo se observaron 2 ciclos epidémicos completos.

### 13.3 Riesgos identificados

**Riesgo A — Sobreajuste al brote de 2024**

El año 2024 concentra el 64.1% de los casos totales del período (23,771 de 37,078). Fue un año epidémico históricamente atípico en Argentina. El modelo corre riesgo de aprender principalmente las condiciones de ese brote excepcional y tener menor performance en brotes de magnitud moderada, que son los más frecuentes históricamente.

**Riesgo B — Validación temporal limitada**

Una validación temporal correcta requiere al menos un año completo de datos de test no vistos durante el entrenamiento. Con 3 años, si se usa 2023-2024 para entrenamiento y 2025 para test, el conjunto de test tiene solo 175 casos confirmados — insuficiente para una evaluación estadísticamente robusta.

**Riesgo C — Ciclos inter-epidémicos no capturados**

El dengue presenta ciclos de varios años relacionados con la inmunidad poblacional acumulada y los serotipos circulantes. Con 3 años de datos el modelo no puede aprender estos patrones de largo plazo.

**Riesgo D — Ausencia de año sin brote significativo**

Los 3 años disponibles incluyen dos brotes importantes (2023 y 2024). No hay en el dataset un año de baja transmisión (como 2017-2018 a nivel nacional), lo que puede limitar la capacidad del modelo para distinguir condiciones de riesgo bajo.

### 13.4 Estrategias de mitigación implementadas

**1. Validación cruzada temporal con ventana expandible**

En lugar de un único split train/test, se usará una estrategia de validación con múltiples splits que maximice el uso de los 3 años disponibles, respetando siempre el orden cronológico de los datos.

**2. Modelos robustos a datasets pequeños**

Se priorizará XGBoost y Random Forest sobre LSTM en la comparación de modelos, ya que los modelos de árboles de decisión tienen mejor comportamiento con datasets pequeños. LSTM se incluirá como experimento complementario con las limitaciones documentadas.

**3. Métricas de evaluación conservadoras**

Las métricas de éxito definidas en el plan (>80% detección, >70% precisión) se revisarán en función de los resultados reales. El objetivo del trabajo académico es demostrar la **viabilidad del enfoque** y el **correcto diseño del pipeline**, no necesariamente alcanzar performance productiva con datos limitados.

**4. Documentación explícita en la memoria**

Esta limitación se documentará en la sección de limitaciones y trabajos futuros de la memoria del proyecto, con la recomendación de incorporar datos históricos adicionales (2018-2022) si estuvieran disponibles en el futuro.

### 13.5 Relación con el plan de proyecto

Esta limitación estaba anticipada en el plan original:

> *Supuesto:* "La cantidad de datos históricos disponibles es suficiente para entrenar modelos de ML con capacidad de generalización adecuada."

> *Riesgo 1:* "Insuficiencia o inconsistencia de datos históricos — Severidad: 8, Probabilidad: 7, RPN: 56"

La materialización parcial de este riesgo (disponibilidad de solo 3 años en lugar de los 5 ideales) se gestiona con las estrategias de mitigación descritas arriba y no compromete la viabilidad del trabajo académico.

### 13.6 Conclusión

Los datos disponibles son **suficientes para los objetivos académicos del trabajo final de especialización** — demostrar el pipeline completo de predicción espacio-temporal, aplicar correctamente algoritmos de ML y analizar resultados con rigor metodológico. Para una implementación productiva en el sistema de salud de CABA se recomienda incorporar al menos 5 años adicionales de datos históricos.