# Decisiones técnicas — Sprint 5

Registro de decisiones técnicas tomadas durante el Sprint 5 (XGBoost multi-horizonte, LSTM, GRU y data augmentation). Continúa desde DT-22 del Sprint 4. Cada decisión documenta el contexto, las alternativas consideradas y la justificación.

---

## DT-23 — Targets multi-horizonte generados en lags.py

**Contexto:** Para evaluar modelos a 1, 2, 3 y 4 semanas adelante se necesitan variables objetivo desplazadas temporalmente. Estas pueden generarse en cada script de modelo o una única vez en el pipeline de features.

**Alternativas consideradas:**
- Generar los targets en cada script de modelo (XGBoost, LSTM, GRU) — más flexible pero introduce riesgo de inconsistencias entre modelos.
- Generarlos una vez en `lags.py` con `shift(-h)` dentro de cada comuna — única fuente de verdad para todos los modelos.

**Decisión:** Generar `target_h1`, `target_h2`, `target_h3`, `target_h4` en `lags.py` usando `groupby("comuna_id")["confirmed_cases"].shift(-h)`.

**Justificación:** Los targets son datos preparados, no lógica de modelo. Generarlos en `lags.py` garantiza que XGBoost, LSTM y GRU usen exactamente los mismos targets, eliminando una fuente de variación no controlada en la comparación. Las últimas h semanas de cada comuna quedan con NaN y se excluyen automáticamente durante el entrenamiento de cada modelo.

**Impacto:** Consistencia garantizada en la comparación de los tres modelos del Sprint 5.

---

## DT-24 — Normalización diferenciada según tipo de modelo

**Contexto:** XGBoost y Random Forest son invariantes a la escala (solo hacen cortes binarios). LSTM y GRU son muy sensibles a la escala de los datos — valores sin normalizar pueden hacer que los gradientes exploten o desaparezcan durante el entrenamiento.

**Alternativas consideradas:**
- Normalizar todo por igual — simplifica el código pero aplica transformaciones innecesarias a los árboles.
- No normalizar nada — correcto para árboles pero causa inestabilidad en redes neuronales.
- Normalización diferenciada: StandardScaler para features en `lags.py` (sufijo `_norm`), MinMaxScaler para el target en LSTM/GRU.

**Decisión:** XGBoost no requiere normalización de features. LSTM y GRU usan las columnas `_norm` generadas en `lags.py` para features y un MinMaxScaler propio ajustado solo con train para el target.

**Justificación:** Aplicar normalización innecesaria a los árboles no mejora las métricas pero agrega complejidad. Para redes neuronales, un input de 1.391 casos vs 0.001 desestabiliza el entrenamiento por gradiente. El MinMaxScaler del target lleva los valores al rango [0, 1] — en la validación puede superar 1 (valores fuera del rango de train) lo que es correcto y esperado con distribution shift.

---

## DT-25 — Matriz de vecindad de CABA codificada directamente en lags.py

**Contexto:** El plan de tesis contempla features espaciales que consideren los casos en comunas vecinas. No existe un archivo de vecindad disponible en el repositorio.

**Alternativas consideradas:**
- Archivo externo GeoJSON con polígonos de comunas — requiere geopandas y cálculo de adyacencias.
- Calcular vecindad por proximidad de centroides — aproximación geométrica imprecisa.
- Codificar la matriz directamente en `lags.py` como diccionario `VECINOS_CABA` verificada manualmente.

**Decisión:** Codificar `VECINOS_CABA` directamente en `lags.py`, verificada contra el mapa oficial de comunas de CABA (01/09/2005).

**Justificación:** La vecindad entre comunas es información estática que no cambia. Codificarla directamente hace el código autocontenido. C7 (Flores/Parque Chacabuco) tiene 7 vecinos (máximo), C1/C8/C9 tienen 3 (mínimo).

**Resultado:** `incidencia_vecinas_lag1` aparece en el top 5 de importancia de features en XGBoost para la semana actual (3.15% de importancia).

---

## DT-26 — Transformación logarítmica del target en XGBoost

**Contexto:** La distribución de `confirmed_cases` es muy asimétrica — mayoría de semanas con 0-5 casos, pico de 1.391. XGBoost minimiza el error cuadrático medio, priorizando los picos extremos en detrimento de las semanas normales.

**Alternativas consideradas:**
- Sin transformación — XGBoost v1, MAE=41.06 en validación.
- log(1+x) — comprime los valores extremos, amplifica la señal moderada.
- Raíz cuadrada — más suave, menos habitual en conteos epidemiológicos.

**Decisión:** `np.log1p(y)` para entrenar y `np.expm1(pred)` para desnormalizar.

**Justificación:** `log1p` es el estándar para conteos con ceros — exactamente reversible con `expm1`. Comprime el pico de 1.391 casos (→7.24) sin eliminarlo, equilibrando el aprendizaje.

**Impacto:** MAE en validación mejoró un 39%: de 41.06 a 24.92.

---

## DT-27 — Early stopping en XGBoost con eval_set en validación

**Contexto:** Sin early stopping, XGBoost entrena todos los árboles siempre, generando sobreajuste al final del entrenamiento.

**Decisión:** `early_stopping_rounds=30` con `eval_set=[(X_val, y_val_log)]`. El eval_set usa el target en escala logarítmica para ser consistente con el entrenamiento.

**Justificación:** El early stopping evita sobreajuste y reduce el tiempo de entrenamiento. Usar validación como eval_set no introduce data leakage porque no forma parte del train. 30 rondas detecta estancamiento sin detener prematuramente.

---

## DT-28 — TimeSeriesSplit para búsqueda de hiperparámetros en XGBoost

**Contexto:** La validación cruzada aleatoria (K-Fold) en series temporales mezcla futuro con pasado — data leakage.

**Decisión:** `TimeSeriesSplit(n_splits=3)` con 72 combinaciones de hiperparámetros (grilla exhaustiva: 3×3×2×2×2).

**Justificación:** TimeSeriesSplit respeta la causalidad temporal — siempre evalúa con datos posteriores al entrenamiento. 72 combinaciones × 3 folds es un balance entre exhaustividad y tiempo de cómputo (~20 minutos totales para los 5 modelos).

---

## DT-29 — Lag de 1 semana en features de vecindad espacial

**Contexto:** Al calcular el promedio de casos de comunas vecinas, se puede usar el valor actual (semana t) o el de la semana anterior (lag 1).

**Decisión:** Aplicar `shift(1)` para obtener `casos_vecinas_lag1` e `incidencia_vecinas_lag1`.

**Justificación:** Los casos de las comunas vecinas de esta semana no están disponibles en tiempo real al momento de predecir. Con lag 1 se usa información de la semana pasada, que sí está disponible — evita data leakage y reproduce correctamente el escenario de uso real del sistema.

---

## DT-30 — Ventana temporal de 12 semanas para LSTM y GRU

**Contexto:** Los modelos LSTM y GRU procesan secuencias de longitud fija. Una ventana más larga captura más historia pero genera menos muestras de entrenamiento.

**Alternativas consideradas:**
- WINDOW_SIZE=8 — genera 660 secuencias. Las curvas de aprendizaje mostraron sobreajuste severo (train→0.01, val→0.093).
- WINDOW_SIZE=12 — genera 600 secuencias. Captura 3 meses de historia, suficiente para ver el inicio y pico de un brote típico.
- WINDOW_SIZE=16 — genera ~540 secuencias. Captura más historia pero reduce aún más las muestras disponibles.

**Decisión:** WINDOW_SIZE=12 semanas para todos los modelos recurrentes.

**Justificación:** Con 12 semanas el modelo puede observar el patrón completo de ascenso de un brote. Las curvas de aprendizaje mejoraron respecto a WINDOW_SIZE=8 — menor brecha entre train y val loss.

---

## DT-31 — Arquitecturas reducidas para LSTM y GRU

**Contexto:** Las arquitecturas originales con 64 unidades LSTM y Dropout=0.2 mostraron sobreajuste severo con 660 muestras de entrenamiento.

**Decisión:**
- LSTM simple: LSTM(32) → Dense(16) → Dense(8) → Dense(1), Dropout=0.1
- LSTM apilado: LSTM(32) → LSTM(16) → Dense(8) → Dense(1), Dropout=0.1
- GRU simple: GRU(32) → Dense(16) → Dense(8) → Dense(1), Dropout=0.1
- GRU apilado: GRU(32) → GRU(16) → Dense(8) → Dense(1), Dropout=0.1

**Justificación:** Reducir de 64 a 32 unidades disminuye ~25% los parámetros a aprender. Dropout=0.1 (antes 0.2) es más suave — con pocos datos el Dropout agresivo impedía el aprendizaje. La transición gradual (32→16→8→1) evita saltos abruptos en la dimensionalidad.

---

## DT-32 — Huber loss en lugar de MSE para LSTM y GRU

**Contexto:** Con el 60% de semanas con 0 casos, MSE hace que el modelo aprenda a predecir siempre cercano a cero — minimiza el error en la mayoría de los casos pero es inútil para detectar brotes.

**Decisión:** `tf.keras.losses.Huber(delta=0.1)` en escala normalizada, equivalente a ~130 casos reales.

**Justificación:** Huber loss combina MSE para errores pequeños (aprendizaje preciso) y MAE para errores grandes (no colapsa el entrenamiento). Con delta=0.1, errores hasta ~130 casos reciben MSE y errores mayores reciben MAE. Las curvas de aprendizaje mostraron convergencia más estable que con MSE.

---

## DT-33 — Sample weights para corregir el desbalanceo (60% de ceros)

**Contexto:** El dataset de train tiene 60% de semanas con 0 casos y solo 2.1% con más de 200 casos. Sin corrección el modelo aprende que predecir cero siempre minimiza el error promedio.

**Decisión:** `sample_weight = np.log1p(y_real) + 1.0` para cada secuencia de entrenamiento.

**Justificación:** Los pesos logarítmicos amplifican las semanas de brote (una semana con 500 casos pesa ~7.2x más que una semana con 0 casos) sin hacer que dominen completamente el entrenamiento. El logaritmo es la función estándar para este tipo de corrección — suave pero efectiva.

---

## DT-34 — Hiperparámetros de entrenamiento para LSTM y GRU

**Contexto:** La configuración original (lr=0.001, batch=32, patience=20) generaba convergencia prematura a mínimos locales subóptimos.

**Decisión:**

| Parámetro | Valor v1 | Valor final | Justificación |
|---|---|---|---|
| `learning_rate` | 0.001 | **0.0005** | Aprendizaje más lento y estable |
| `batch_size` | 32 | **16** | Más actualizaciones por época con dataset pequeño |
| `patience` EarlyStopping | 20 | **30** | Más tiempo para converger con lr bajo |
| `epochs` máximo | 200 | **300** | Permite más tiempo de convergencia |
| `factor` ReduceLROnPlateau | — | **0.5** | Reduce lr a la mitad si hay estancamiento |
| `patience` ReduceLROnPlateau | — | **10** | Activa tras 10 épocas sin mejora |

**Justificación:** Con lr=0.001 el modelo convergía en ~25 épocas a un mínimo local subóptimo. Con lr=0.0005 explora más el espacio de parámetros. Batch=16 genera el doble de actualizaciones por época respecto a batch=32.

---

## DT-35 — Data augmentation con factores x2 y x3 sobre semanas de brote

**Contexto:** El dataset de train (2023) tiene un máximo de 649 casos/semana. La validación (2024 S1) llega a 1.391 casos. Este distribution shift impide que los modelos recurrentes generalicen al brote inédito de 2024.

**Alternativas consideradas:**
- Sin augmentation — los modelos nunca ven brotes de magnitud comparable a 2024.
- Factor x2 solo — cubre parte del rango pero no llega a la magnitud de 2024.
- Factores x2 y x3 — cubre el rango completo (sintético x3 alcanza 2.110 casos).
- Factor aleatorio entre 1.5 y 4.0 — mayor diversidad pero menor control.

**Decisión:** Generar 2 copias de cada semana con `confirmed_cases > 50` (61 semanas), escalando valores de casos por 2.0 y 3.0, con ruido gaussiano del 5%. Solo se escalan las variables de casos — las climáticas no se modifican.

**Justificación:** El umbral de 50 casos garantiza que solo se amplifica brote real, no actividad baja. El ruido del 5% diversifica los patrones sintéticos sin distorsionar la dinámica epidemiológica. Las variables climáticas son independientes del número de casos — escalarlas sería epidemiológicamente incorrecto.

**Resultado:**
- Train original: 780 filas, 61 semanas de brote severo (7.8%)
- Train aumentado: 902 filas, 183 semanas de brote severo (20.3%)
- Máximo sintético: 2.110 casos
- LSTM simple h=4: MAE de 12.88 (sin aug) a 10.54 (con aug) — mejora del 18%
- LSTM simple h=3: MAE de 18.77 (sin aug) a 16.67 (con aug) — supera a la persistencia por primera vez

---

## DT-36 — LSTM simple como modelo definitivo de redes recurrentes

**Contexto:** Se compararon 4 arquitecturas (LSTM simple, LSTM apilado, GRU simple, GRU apilado) con y sin augmentation. Hay que seleccionar el modelo definitivo para el sistema de alertas tempranas.

**Resultados de la comparación con augmentation (MAE en validación):**

| Modelo | h=3 | h=4 | Estabilidad |
|---|---|---|---|
| LSTM simple + aug | **16.67** | **10.54** | Alta |
| LSTM apilado + aug | 16.67 | 29.88 | Baja (inestable) |
| GRU simple + aug | 21.98 | 10.59 | Alta |
| GRU apilado + aug | 21.59 | 10.82 | Baja (h=2: MAE=71) |

**Decisión:** LSTM simple con data augmentation como modelo definitivo de redes neuronales recurrentes.

**Justificación:** El LSTM simple supera al GRU simple en h=3 (16.67 vs 21.98) y es prácticamente igual en h=4 (10.54 vs 10.59). Las arquitecturas apiladas muestran inestabilidad con los datos aumentados. La arquitectura más simple generaliza mejor con este dataset de 902 filas.

---

## DT-37 — MAE como métrica primaria ante R² negativo

**Contexto:** Todos los modelos LSTM y GRU muestran R² negativos o cercanos a cero en el período de validación.

**Causas del R² negativo identificadas:**
1. Distribution shift: train tiene máximo 649 casos, validación tiene máximo 1.391. Los modelos subestiman sistemáticamente los picos.
2. Heterogeneidad de la validación: mezcla semanas de brote (200-1.391 casos) con semanas de descenso (0-50). El R² penaliza doble.
3. Varianza cercana a cero en comunas con pocos casos: cualquier predicción no-cero destruye el R² de esa comuna.

**Decisión:** Usar MAE como métrica primaria. R² se reporta como métrica secundaria con aclaración explícita de sus limitaciones en este contexto.

**Justificación:** El MAE mide el error promedio en casos reales — unidad directamente interpretable por los equipos de salud pública. Es robusto ante la heterogeneidad del período de validación. Esta decisión está alineada con Sebastianelli et al. (2024), que usa MAE como métrica primaria en predicción de dengue con datos limitados.

---

## Resumen de hallazgos con impacto en Sprint 6

| Hallazgo | Implicación para Sprint 6 |
|---|---|
| LSTM simple + aug es el mejor modelo recurrente | Usar para dashboard y evaluación HU7 |
| XGBoost gana a corto plazo (h=1), LSTM gana a largo (h=3,h=4) | Sistema de alertas multicapa: XGBoost para h=1, LSTM para h=3-4 |
| R² negativo por distribution shift y heterogeneidad de val | Evaluar HU7 por período (brote vs. descenso) en lugar de global |
| Vecindad espacial relevante a corto plazo | Incluir mapa de comunas en dashboard con propagación espacial |
| Augmentation mejora h=3,4 pero empeora h=1,2 | Augmentation selectivo por horizonte como línea de trabajo futuro |

---

*Documento generado al cierre del Sprint 5*
*Próximas decisiones: DT-38 en adelante — Sprint 6*
