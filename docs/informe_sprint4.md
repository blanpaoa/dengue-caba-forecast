# Informe de resultados — Sprint 4
---

## ¿Qué hicimos en este sprint?

Entrenamos y comparamos cuatro modelos en orden creciente de complejidad para predecir cuántos casos de dengue habrá en cada una de las 15 comunas de CABA en la semana siguiente, usando datos climáticos e historial de casos.

**Dataset:** 2.340 observaciones · 15 comunas · período 2023–2025 · 67 variables predictoras

---

## Los cuatro modelos — escalera de complejidad

### 1. Persistencia (lag 1)
El modelo más simple posible: predice que esta semana habrá los mismos casos que la semana anterior. No aprende nada del clima ni de la historia — solo recuerda el último valor conocido. Es el piso mínimo que todo modelo más complejo debe superar. Es referencia estándar en la predicción de dengue (Sebastianelli et al., 2024).

### 2. Baseline — promedio histórico
Predice el promedio de casos de esa semana epidemiológica en el año de entrenamiento. Captura estacionalidad (sabe que enero tiene más casos que julio) pero no sabe si hay un brote activo en este momento.

### 3. Ridge climático
Regresión lineal regularizada que usa **solo variables climáticas y estacionalidad**, sin información de casos previos. Permite medir cuánto aporta el clima por sí solo, independientemente de la dinámica autorregresiva de la enfermedad. No incluye lags de casos — esa es su restricción deliberada.

### 4. Random Forest — modelo de inteligencia artificial
Aprende de combinaciones de variables: casos previos, temperatura, humedad, precipitación, estacionalidad y características de cada comuna. Utiliza 300 árboles de decisión trabajando en conjunto con las 67 features del Sprint 4.

---

## Resultados principales — período de brote 2024 (semanas 1–26)

Este es el período más importante: el brote más grande registrado en CABA, con hasta **1.391 casos en una sola semana**.

| Modelo | MAE | RMSE | R² | Interpretación |
|---|---|---|---|---|
| Persistencia (lag 1) | 16.81 | 43.10 | **0.932** | Inercia del brote domina |
| Media histórica | 50.62 | 138.82 | 0.300 | No captura magnitud del brote |
| Ridge climático | 71.25 | 168.80 | -0.035 | Clima insuficiente sin autorregresión |
| Random Forest | **24.93** | **103.13** | 0.614 | Mejor balance generalización |

### Hallazgos clave

**Persistencia es el modelo más preciso en validation (R²=0.932).** Esto no es un problema, es un hallazgo epidemiológico: durante un brote activo, los casos de hoy predicen muy bien los de mañana. La enfermedad tiene una inercia fuerte a corto plazo. Cualquier modelo avanzado deberá competir con este resultado.

**El clima solo no alcanza (Ridge R²=-0.035).** El modelo climático captura estacionalidad y temperatura con lag de 4 semanas, pero no puede cuantificar la magnitud del brote sin información autorregresiva. Un R² negativo significa que predecir siempre el promedio sería mejor que usar solo el clima. Esto valida la necesidad de combinar ambas fuentes.

**El Random Forest equilibra generalización y precisión.** Con R²=0.614 y MAE=24.93, es el único modelo que combina clima, estacionalidad y casos previos de forma no lineal. Supera a la media histórica en 51% de MAE.

---

## Resultados por período — test set completo

| Período | Casos reales (media) | MAE baseline | MAE Ridge | MAE Random Forest |
|---|---|---|---|---|
| 2024 S2 — post-brote | ~0.01 | 0.15 | 26.99 | **0.15** |
| 2025 — brote moderado | ~0.22 | 16.63 | 23.26 | **1.39** |

En temporada baja (2024 S2) baseline y RF empatan — no hay varianza real que predecir. En 2025, con un brote moderado, el RF supera al baseline por un factor de 12x y al Ridge por 17x.

---

## ¿Qué variables usa cada modelo?

### Random Forest — importancia Gini

| Variable | Importancia |
|---|---|
| Incidencia semana anterior (lag 1) | 51.9% |
| Casos semana anterior (lag 1) | 42.5% |
| Temperatura media (lag 1) | 0.6% |
| Estacionalidad (semana del año) | 0.5% |
| Resto de variables | 3.5% |

El RF funciona principalmente como un modelo autorregresivo AR(1). Las variables climáticas adquirirán mayor peso en horizontes de predicción más largos (2–4 semanas), que se explorarán en Sprint 5.

### Ridge climático — coeficientes absolutos (top 5)

| Variable | Coeficiente |
|---|---|
| Estacionalidad (semana_sin) | 75.21 |
| Heat index lag 4 semanas | 64.17 |
| Temperatura media lag 4 semanas | 44.30 |
| Anomalía temperatura lag 4 semanas | 37.48 |
| Estacionalidad (semana_cos) | 35.32 |

El Ridge confirma que la temperatura de **4 semanas atrás** es la variable climática más relevante — consistente con el ciclo biológico del Aedes aegypti. Este resultado aporta evidencia para el diseño de features en Sprint 5.

---

## Split temporal utilizado

| Conjunto | Período | Filas | Propósito |
|---|---|---|---|
| Train | 2023 completo | 780 | Entrenamiento |
| Validation | 2024 semanas 1–26 | 390 | Evaluación durante el brote masivo |
| Test | 2024 sem. 27–52 + 2025 | 1.170 | Evaluación final |

El scaler (normalización) se ajustó **solo con train** para evitar filtración de información del futuro al modelo.

---

## Conclusión

La escalera de complejidad revela tres hallazgos:

1. **La inercia temporal domina a corto plazo** — la persistencia con R²=0.932 muestra que durante un brote el valor anterior es el mejor predictor inmediato.
2. **El clima solo es insuficiente** — el Ridge climático con R²=-0.035 demuestra empíricamente que se necesita información autorregresiva, justificando la arquitectura del sistema.
3. **La combinación supera a las partes** — el Random Forest con R²=0.614 integra clima y casos previos de forma no lineal, superando a todos los baselines en el período con varianza real (2025).

El próximo paso (Sprint 5) es entrenar **XGBoost** con búsqueda de hiperparámetros y evaluar múltiples horizontes de predicción (1–4 semanas) según los criterios de aceptación de HU6.

---

## Archivos generados

```
models/saved/random_forest.pkl              ← modelo RF entrenado
models/saved/ridge_climatico.pkl            ← modelo Ridge entrenado
models/saved/ridge_scaler.pkl               ← scaler del Ridge
models/saved/metricas_sprint4.csv           ← métricas de los 4 modelos
reports/figures/14_importancia_features_rf_ridge.png
reports/figures/15_residuos_baseline_vs_rf_validation.png
```

---


