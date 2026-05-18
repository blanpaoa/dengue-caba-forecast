# Decisiones técnicas — Sprint 4

Registro de decisiones técnicas tomadas durante el Sprint 4 (feature engineering, modelos baseline y Random Forest). Cada decisión documenta el contexto, las alternativas consideradas y la justificación.

---

## DT-15 — Eliminación de filas con NaN en lugar de imputación con cero

**Contexto:** Las primeras semanas de cada comuna en el período de train no tienen valores de lags (`cases_lag1`, `cases_lag2`, etc.) porque no hay semanas previas disponibles. Esto genera 60 filas con NaN (4 semanas × 15 comunas).

**Alternativas consideradas:**
- Imputar con 0 — fácil pero introduce señal falsa (0 no significa "no hubo casos", significa "no tenemos dato").
- Imputar con la media de la comuna — más informativo pero introduce sesgo.
- Eliminar las filas — pierde 60 observaciones (7.7% del train).

**Decisión:** Eliminar filas con NaN en las features de lags antes del entrenamiento.

**Justificación:** Las primeras semanas de 2023 corresponden a temporada baja con casos cercanos a cero. Su eliminación no afecta la capacidad del modelo para aprender patrones de brote. Imputar con 0 sería epidemiológicamente incorrecto y podría sesgar los coeficientes de los lags hacia valores bajos.

**Impacto:** Train efectivo = 720 filas (92.3% del total).

---

## DT-16 — Escalera de cuatro modelos baseline en orden de complejidad creciente

**Contexto:** El criterio de aceptación de HU6 exige "al menos 3 algoritmos". La tesis propone comparar modelos de diferente naturaleza para justificar la progresión hacia XGBoost y LSTM.

**Alternativas consideradas:**
- Solo comparar baseline vs RF (mínimo requerido).
- Incluir SARIMA como tercer modelo — requiere stationarity tests y es más complejo de implementar.
- Escalera de 4 modelos: persistencia → media histórica → Ridge climático → RF.

**Decisión:** Implementar la escalera de 4 modelos donde cada uno agrega una dimensión nueva respecto al anterior.

**Justificación:** La escalera permite aislar el aporte de cada componente:
- Persistencia mide la inercia temporal pura.
- Media histórica mide el aporte de la estacionalidad.
- Ridge climático mide el aporte del clima sin autorregresión.
- Random Forest mide el aporte de combinar todo con relaciones no lineales.

Esta estructura es metodológicamente sólida y responde la pregunta de investigación: ¿qué aporta el clima más allá de la autocorrelación temporal?

---

## DT-17 — Modelo de persistencia usando cases_lag1 del dataset

**Contexto:** El modelo de persistencia predice `casos(t) = casos(t-1)`. Puede implementarse de dos formas: calculando el lag en tiempo de inferencia, o usando la variable `cases_lag1` ya presente en el dataset.

**Alternativas consideradas:**
- Calcular el lag en tiempo de inferencia — más explícito pero redundante.
- Usar `cases_lag1` del dataset — aprovecha el trabajo del Sprint 4, garantiza consistencia con los demás modelos.

**Decisión:** Usar `df_val["cases_lag1"].fillna(0)` directamente.

**Justificación:** `cases_lag1` fue calculado en `lags.py` con la misma lógica de agrupación por comuna (`groupby("comuna_id").shift(1)`), garantizando que el lag respeta la dimensión espacial. Usar la misma fuente que el RF asegura que la comparación es justa.

---

## DT-18 — Ridge (L2) en lugar de OLS para el modelo climático

**Contexto:** El modelo climático usa 42 variables climáticas y de estacionalidad, muchas de ellas correlacionadas entre sí (temperatura, heat index, anomalías térmicas). Con OLS clásico, la multicolinealidad genera coeficientes inestables y potencialmente enormes.

**Alternativas consideradas:**
- OLS (mínimos cuadrados ordinarios) — simple pero inestable con variables correlacionadas.
- Ridge (L2) — penaliza la magnitud de los coeficientes, estabiliza la solución.
- Lasso (L1) — genera coeficientes exactamente cero, útil para selección de features.
- ElasticNet (L1+L2) — combina ambas penalizaciones.

**Decisión:** Ridge con `alpha=1.0`.

**Justificación:** Ridge es la elección correcta cuando el objetivo es **predecir** (no seleccionar features). Con 720 filas efectivas y 42 variables, la regularización L2 evita sobreajuste sin eliminar variables que pueden ser relevantes en otros horizontes temporales. `alpha=1.0` es conservador — puede ajustarse mediante búsqueda en Sprint 5.

**Nota:** El scaler (StandardScaler) se ajusta solo con train y se aplica a val y test para evitar data leakage.

---

## DT-19 — Ridge climático sin lags de casos (diseño deliberado)

**Contexto:** El objetivo del Ridge climático es medir el aporte del clima de forma aislada, sin mezclar señal autorregresiva. Si se incluyeran `cases_lag1` e `incidencia_lag1`, el Ridge se volvería un modelo mixto y no podría responder la pregunta de investigación.

**Decisión:** Excluir explícitamente todos los lags de casos (`cases_lag1-4`, `incidencia_lag1-4`) del conjunto de features del Ridge.

**Justificación:** La hipótesis central de la tesis es que integrar datos climáticos mejora la predicción. Para validarla, es necesario tener un modelo de referencia que use solo clima. El resultado obtenido (R²=-0.035) demuestra empíricamente que el clima solo es insuficiente — lo que justifica la arquitectura completa del sistema.

---

## DT-20 — Hiperparámetros del Random Forest

**Contexto:** El RF requiere definir `n_estimators`, `max_depth` y `min_samples_leaf` antes del entrenamiento. En Sprint 5 se realizará búsqueda formal de hiperparámetros para XGBoost; para el RF de baseline se usan valores conservadores.

**Decisión:**

| Parámetro | Valor | Justificación |
|---|---|---|
| `n_estimators` | 300 | Suficientes árboles para estabilidad estadística |
| `max_depth` | 10 | Limita profundidad para evitar sobreajuste con 720 filas |
| `min_samples_leaf` | 5 | Cada hoja necesita al menos 5 muestras — reduce varianza |
| `random_state` | 42 | Reproducibilidad |
| `n_jobs` | -1 | Usa todos los núcleos disponibles |

**Justificación:** Con un dataset de 720 filas efectivas, la prioridad es evitar sobreajuste. `max_depth=10` y `min_samples_leaf=5` son restricciones conservadoras estándar para datasets de tamaño reducido. La búsqueda de hiperparámetros se realizará en Sprint 5 con validación cruzada temporal.

---

## DT-21 — Métrica por período para el test set heterogéneo

**Contexto:** El test set mezcla dos distribuciones muy distintas: 2024 S2 (casi sin casos) y 2025 (brote moderado). Las métricas globales como R² colapsan cuando la varianza es casi cero (2024 S2 con media=0.01).

**Decisión:** Reportar MAE desagregado por año además de las métricas globales.

**Justificación:** Una sola métrica global sobre el test set completo es engañosa. El MAE por período permite evaluar el modelo en contextos epidemiológicamente distintos: temporada baja vs. brote activo. Esta desagregación está alineada con el criterio de aceptación de HU7: "métricas de error por comuna y horizonte temporal".

---

## DT-22 — Guardado del scaler del Ridge por separado

**Contexto:** El Ridge climático requiere normalizar los datos de entrada con el mismo scaler ajustado en train. Para usar el modelo en producción o en Sprint 5, el scaler debe estar disponible junto al modelo.

**Decisión:** Guardar `ridge_scaler.pkl` en `models/saved/` junto a `ridge_climatico.pkl`.

**Justificación:** Sin el scaler, el modelo Ridge no puede hacer predicciones sobre datos nuevos. El RF no requiere esta precaución porque trabaja con los datos sin normalizar. Guardar ambos archivos juntos garantiza que el pipeline de inferencia sea reproducible.

---

## Resumen de hallazgos con impacto en Sprint 5

| Hallazgo | Implicación para Sprint 5 |
|---|---|
| Persistencia R²=0.932 | XGBoost debe superar este umbral para justificarse |
| Ridge R²=-0.035 | Confirma que los lags de casos son imprescindibles |
| `temp_mean_lag4` es la variable climática más importante en Ridge | Explorar lags climáticos más largos (hasta 6 semanas) en XGBoost |
| RF depende 94% de lags de casos | Evaluar horizonte h=2 y h=4 sin lag 1 disponible |
| Test set heterogéneo (baja + brote moderado) | Reportar métricas por período en todos los modelos de Sprint 5 |

---

*Documento generado al cierre del Sprint 4 · Próximo: DT-23 en adelante — Sprint 5 XGBoost*
