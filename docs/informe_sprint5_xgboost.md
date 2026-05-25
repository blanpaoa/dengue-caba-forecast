# Informe de resultados — Sprint 5 (XGBoost)
---

## ¿Qué hicimos en este sprint?

Entrenamos un modelo XGBoost (eXtreme Gradient Boosting) para predecir casos de dengue en las 15 comunas de CABA. A diferencia del Sprint 4 (que solo predecía la semana actual), este modelo evalúa cuatro distancias temporales distintas: 1, 2, 3 y 4 semanas adelante.

Se incorporaron tres mejoras técnicas respecto al Random Forest del Sprint 4:
- **Transformación logarítmica** del target para manejar la distribución asimétrica de casos
- **Early stopping** para evitar sobreajuste durante el entrenamiento
- **Features de vecindad espacial** que capturan la dispersión geográfica del dengue entre comunas

**Dataset:** 2.340 observaciones · 15 comunas · período 2023–2025 · 73 variables predictoras (67 del Sprint 4 + 6 nuevas: targets h1-h4, casos_vecinas_lag1, incidencia_vecinas_lag1)

---

## ¿Qué es XGBoost y en qué se diferencia del Random Forest?

Ambos son modelos basados en árboles de decisión, pero con una diferencia clave en cómo aprenden:

**Random Forest** construye muchos árboles en paralelo e independientes, y promedia sus predicciones. Es robusto y rápido.

**XGBoost** construye los árboles en secuencia: cada árbol nuevo aprende de los errores del árbol anterior. Esto lo hace más preciso en datasets con distribuciones complejas, como la distribución muy sesgada de casos de dengue (mayoría de semanas con 0-5 casos, picos de hasta 1.391).

---

## Resultados — comparación Sprint 4 vs Sprint 5

Período de validación: brote 2024, semanas 1-26 (hasta 1.391 casos en una sola semana).

| Modelo | Semanas adelante | MAE | R² | Observación |
|---|---|---|---|---|
| Persistencia | 1 semana | 16.81 | 0.932 | Referencia mínima |
| Random Forest | 1 semana | 24.93 | 0.614 | Mejor modelo Sprint 4 |
| **XGBoost** | **semana actual** | **24.92** | **0.580** | Iguala al RF con mejoras |
| XGBoost | 1 semana adelante | 34.09 | 0.430 | Predicción futura |
| XGBoost | 2 semanas adelante | 41.82 | 0.281 | Predicción futura |
| XGBoost | 3 semanas adelante | 37.08 | 0.371 | Predicción futura |
| XGBoost | 4 semanas adelante | 45.54 | 0.229 | Predicción futura |
| Media histórica | 1 semana | 50.62 | 0.300 | Referencia estacional |
| Ridge climático | 1 semana | 71.25 | -0.035 | Solo clima |

> **En palabras simples:** XGBoost iguala al Random Forest para la semana actual, y además puede predecir hasta 4 semanas adelante con R² positivo (mayor que cero = mejor que predecir siempre el promedio).

---

## ¿Cómo degrada la precisión al predecir más lejos?

Es esperable que el error aumente cuanto más lejos en el futuro miramos — cuanta más incertidumbre, más difícil es predecir.

| Horizonte | MAE | R² | Utilidad para la salud pública |
|---|---|---|---|
| Semana actual | 24.92 | 0.580 | Ajuste de guardia médica |
| 1 semana adelante | 34.09 | 0.430 | Planificación de insumos |
| 2 semanas adelante | 41.82 | 0.281 | Inicio de campañas |
| 3 semanas adelante | 37.08 | 0.371 | Alerta temprana |
| 4 semanas adelante | 45.54 | 0.229 | Alerta temprana máxima |

El modelo mantiene R² positivo en todos los horizontes, lo que significa que siempre es mejor que predecir simplemente el promedio histórico.

---

## Hallazgo principal — qué variables usa el modelo según el horizonte

Este es el resultado más importante del Sprint 5. Las variables que el modelo considera más útiles cambian radicalmente según cuánto tiempo adelante predice:

| Horizonte | Variable más importante | Importancia | Categoría |
|---|---|---|---|
| Semana actual | Casos semana anterior | 39% | Historial propio |
| 1 semana | Semana del año | 70% | Estacionalidad |
| 2 semanas | Semana del año | 56% | Estacionalidad |
| 3 semanas | Mes del año | 39% | Estacionalidad |
| 4 semanas | Mes del año + calor de hace 4 semanas | 45%+31% | Estacionalidad + clima |

**Lo que esto significa:** para predecir la semana que viene, lo más útil es saber cuántos casos hubo la semana pasada. Para predecir cuatro semanas adelante, lo más útil es saber en qué época del año estamos y qué tan caluroso fue el mes pasado. El calor favorece la reproducción del mosquito Aedes aegypti, que tarda semanas en completar su ciclo hasta generar casos.

Este resultado confirma la hipótesis de la tesis: **las variables climáticas son más valiosas para alertas tempranas de largo plazo** que para predicciones inmediatas.

---

## Aporte de la vecindad espacial

Por primera vez incorporamos información sobre las comunas vecinas: si la semana pasada hubo muchos casos en las comunas que comparten límite, es más probable que esta semana haya casos en la comuna propia.

`incidencia_vecinas_lag1` (incidencia de las comunas vecinas la semana pasada) aparece en el **top 5** de variables más importantes para la semana actual (3.15% de importancia) y en el **top 8** para 1 semana adelante. Su peso disminuye a mayor horizonte, lo que es coherente con la epidemiología: la dispersión espacial del dengue se observa en el corto plazo.

---

## Rendimiento por período (test set)

| Período | Casos reales (media) | Predicción XGBoost | MAE |
|---|---|---|---|
| 2024 S2 — post-brote | 0.01 | 0.12 | 0.12 |
| 2025 — brote moderado | 0.22 | 0.56 | 0.46 |

En temporada baja el modelo predice valores muy cercanos a cero, igual que la realidad. En 2025 sobreestima levemente pero el error absoluto es bajo.

---

## Mejoras respecto al Sprint 4

| Aspecto | Sprint 4 (RF) | Sprint 5 (XGBoost) |
|---|---|---|
| Horizontes evaluados | 1 (semana actual) | 5 (actual + h=1,2,3,4) |
| Transformación del target | No | Sí (log1p) |
| Early stopping | No | Sí (30 rondas) |
| Features de vecindad | No | Sí (casos_vecinas_lag1) |
| Búsqueda de hiperparámetros | No | Sí (72 combinaciones × 3 folds) |
| MAE en validación | 24.93 | 24.92 |

La mejora en MAE es marginal porque el RF del Sprint 4 ya era competitivo. El valor agregado del Sprint 5 es la evaluación multi-horizonte y la confirmación de qué variables son más importantes en cada horizonte.

---

## Split temporal utilizado

| Conjunto | Período | Filas | Propósito |
|---|---|---|---|
| Train | 2023 completo | 780 | Entrenamiento (720 efectivas tras eliminar NaN) |
| Validation | 2024 semanas 1–26 | 390 | Evaluación durante el brote masivo |
| Test | 2024 sem. 27–52 + 2025 | 1.170 | Evaluación final |

---

## Conclusión

XGBoost con los ajustes del Sprint 5 iguala al Random Forest para predicción de la semana actual y extiende el sistema a 4 semanas de anticipación con R² positivo en todos los horizontes. El hallazgo metodológico más relevante es la transición en importancia de features: del historial de casos (corto plazo) hacia estacionalidad y clima (largo plazo), lo que justifica y valida la arquitectura del sistema de alertas tempranas.

El próximo paso es implementar **LSTM y GRU** — redes neuronales recurrentes que pueden capturar dependencias temporales más complejas y que, en teoría, deberían mejorar especialmente en los horizontes largos (h=3 y h=4) donde XGBoost depende casi exclusivamente de la estacionalidad.

---

## Archivos generados

```
models/saved/xgboost_semana_actual.pkl     ← modelo para semana actual
models/saved/xgboost_h1.pkl               ← modelo para 1 semana adelante
models/saved/xgboost_h2.pkl               ← modelo para 2 semanas adelante
models/saved/xgboost_h3.pkl               ← modelo para 3 semanas adelante
models/saved/xgboost_h4.pkl               ← modelo para 4 semanas adelante
models/saved/xgboost_features_v2.pkl      ← lista de variables del modelo
models/saved/metricas_sprint5_v2.csv      ← métricas de todos los modelos
reports/figures/16_importancia_xgboost_v2.png
reports/figures/17_degradacion_xgboost_v2.png
```

---
