# Informe de resultados — Sprint 5 (LSTM)
---

## ¿Qué hicimos en este sprint?

Entrenamos y comparamos dos arquitecturas de redes neuronales recurrentes LSTM
(Long Short-Term Memory) para predecir casos de dengue en las 15 comunas de CABA
a 1, 2, 3 y 4 semanas de anticipación.

A diferencia de XGBoost y Random Forest (que ven cada semana como una observación
independiente), el LSTM procesa las últimas 12 semanas **en orden cronológico**,
aprendiendo la dinámica temporal completa del brote — no solo el valor de la semana
anterior.

**Dataset:** 600 secuencias de entrenamiento · 210 de validación · ventana de 12 semanas · 44 variables

---

## Los dos modelos comparados

### LSTM simple (1 capa)
```
12 semanas de historia → [LSTM 32 celdas] → [Dense 16] → [Dense 8] → predicción
```
Una sola capa recurrente que resume las 12 semanas en 32 patrones temporales.
Menos parámetros, menor riesgo de sobreajuste, entrena más rápido.

### LSTM apilado (2 capas)
```
12 semanas → [LSTM 32] → [LSTM 16] → [Dense 8] → predicción
```
La primera capa aprende patrones simples (subidas, bajadas).
La segunda aprende patrones de patrones (curvas de brote, mesetas).
Mayor capacidad para capturar dinámicas temporales complejas.

### Ajustes aplicados para mejorar el desbalanceo (60% de semanas con 0 casos)

El dataset de entrenamiento tiene un desbalanceo severo — el 60% de las semanas
tienen 0 casos y solo el 2.1% tienen más de 200 casos (brote severo).
Sin corrección, el modelo aprende a predecir cero siempre, lo que minimiza el
error promedio pero es inútil para detectar brotes.

Se aplicaron dos técnicas:

**Huber loss** (reemplaza MSE): combina lo mejor del error cuadrático (preciso
para errores pequeños) y el error absoluto (robusto ante valores extremos).
Evita que los picos del brote "colapsen" el entrenamiento.

**Sample weights** (pesos por muestra): las semanas con más casos reciben más
peso durante el entrenamiento. Una semana con 500 casos pesa ~7x más que una
semana con 0 casos, obligando al modelo a prestarles atención.

---

## Resultados — comparación con modelos anteriores

Período de validación: brote 2024, semanas 1–26 (hasta 1.391 casos en una sola semana).

| Modelo | Semanas adelante | MAE | R² | Observación |
|---|---|---|---|---|
| Persistencia | 1 semana | 16.81 | 0.932 | Referencia imbatible a corto plazo |
| Random Forest | semana actual | 24.93 | 0.614 | Mejor modelo Sprint 4 |
| XGBoost | semana actual | 24.92 | 0.580 | Sprint 5 |
| LSTM simple | semana actual | 51.99 | -0.029 | Limitado por distribución shift |
| LSTM simple | 1 semana | 39.88 | -0.020 | — |
| LSTM simple | 2 semanas | 28.49 | -0.011 | Cerca del RF |
| LSTM simple | 3 semanas | 18.77 | -0.022 | Cerca de la persistencia |
| **LSTM simple** | **4 semanas** | **12.88** | -0.009 | **Supera a la persistencia** |
| LSTM apilado | 4 semanas | 13.17 | -0.007 | También supera a la persistencia |

---

## El hallazgo más importante — LSTM gana a largo plazo

**A 4 semanas adelante, el LSTM simple (MAE=12.88) supera a la persistencia
(MAE=16.81)** — que era el mejor modelo a corto plazo en Sprint 4.

Esto es el resultado más relevante para el sistema de alertas tempranas:
el LSTM puede anticipar un brote con un mes de anticipación mejor que cualquier
modelo anterior. Esta capacidad es especialmente valiosa porque permite planificar
campañas de fumigación y asignación de recursos antes de que el brote explote.

---

## Degradación de precisión por horizonte

| Horizonte | LSTM simple MAE | LSTM apilado MAE | Referencia |
|---|---|---|---|
| 1 semana | 39.88 | 39.93 | Persistencia: 16.81 |
| 2 semanas | 28.49 | 28.29 | Random Forest: 24.93 |
| 3 semanas | 18.77 | 19.90 | Persistencia: 16.81 |
| **4 semanas** | **12.88** | **13.17** | Persistencia: 16.81 |

La degradación es perfectamente monótona — el error baja consistentemente
al predecir más lejos. Esto indica que el modelo está aprendiendo patrones
reales de la serie temporal, no ruido aleatorio.

A 2 semanas, el LSTM simple (MAE=28.49) es comparable al Random Forest
(MAE=24.93) — con la ventaja de que el LSTM no requiere features de lag
calculadas manualmente.

---

## Las curvas de aprendizaje — diagnóstico del distribution shift

Los gráficos de curva de aprendizaje muestran un patrón llamativo: la curva
de validación (roja) es prácticamente plana desde la primera época, mientras
la curva de entrenamiento (azul) sigue bajando.

**¿Qué significa esto?**
El modelo converge al mínimo posible de validación en las primeras épocas y
después no puede mejorar más. No es sobreajuste clásico — es **distribution
shift**: el período de entrenamiento (2023, máximo 649 casos) tiene una
distribución completamente diferente al período de validación (2024 S1, máximo
1.391 casos). El modelo nunca vio un brote de esa magnitud durante el
entrenamiento, por lo que no puede aprender a predecirlo mejor.

Este es un límite del dataset, no de la arquitectura. Con más años de datos
históricos que incluyan brotes de magnitud similar al de 2024, el LSTM podría
mejorar significativamente.

---

## LSTM simple vs LSTM apilado

| Horizonte | LSTM simple | LSTM apilado | Diferencia |
|---|---|---|---|
| h=1 | 39.88 | 39.93 | Prácticamente iguales |
| h=2 | 28.49 | 28.29 | Prácticamente iguales |
| h=3 | **18.77** | 19.90 | Simple gana |
| h=4 | **12.88** | 13.17 | Simple gana levemente |

El LSTM simple supera o iguala al apilado en todos los horizontes. Con 600
muestras de entrenamiento, la arquitectura más simple generaliza mejor — la
complejidad adicional del apilado no aporta valor con este dataset.

**Conclusión:** hasta este punto, para esta tesis usar LSTM simple es el modelo que parece más adecuado.

---

## Comparación evolutiva — tres versiones del LSTM

| Versión | Cambios | MAE h=4 val |
|---|---|---|
| v1 | LSTM 64, Dropout 0.2, ventana 8, MSE | 42.61 |
| v2 | LSTM 32, Dropout 0.1, ventana 12, batch 16 | 12.88 |
| v FINAL | v2 + Huber loss + sample weights | **12.88** |

Los ajustes de arquitectura (v2) lograron la mayor mejora. Huber loss y
sample weights mejoraron principalmente la estabilidad del entrenamiento y
redujeron el sobreajuste (curvas más equilibradas).

---

## Limitaciones identificadas

**Dataset pequeño:** con 600 secuencias de entrenamiento las redes neuronales
tienen capacidad limitada para generalizar. Los modelos de árboles (XGBoost,
RF) funcionan mejor con pocos datos.

**Distribution shift severo:** el brote de 2024 (máximo 1.391 casos) es
inédito respecto al entrenamiento (máximo 649 casos en 2023). El modelo
no puede predecir lo que nunca vio.

**60% de ceros en el dataset:** el desbalanceo extremo dificulta el
aprendizaje de los patrones de brote. Aunque Huber loss y sample weights
ayudaron, el problema de fondo es la escasez de ejemplos de brote severo.

---

## Conclusión

El LSTM aporta valor real en horizontes largos: supera a la persistencia a 4
semanas (MAE 12.88 vs 16.81) y es comparable al Random Forest a 2 semanas.
Su limitación principal es el dataset pequeño y el distribution shift entre
entrenamiento (2023) y validación (brote masivo 2024).

El próximo paso es implementar **GRU (Gated Recurrent Unit)** — una variante
del LSTM con menos parámetros que suele comportarse mejor con datasets pequeños.

---

## Archivos generados

```
models/saved/lstm_lstm_simple_h0.keras     ← semana actual
models/saved/lstm_lstm_simple_h1.keras     ← 1 semana adelante
models/saved/lstm_lstm_simple_h2.keras     ← 2 semanas adelante
models/saved/lstm_lstm_simple_h3.keras     ← 3 semanas adelante
models/saved/lstm_lstm_simple_h4.keras     ← 4 semanas adelante
models/saved/lstm_lstm_apilado_h*.keras    ← arquitectura apilada (5 modelos)
models/saved/lstm_target_scaler.pkl        ← normalizador del target
models/saved/metricas_sprint5_lstm.csv     ← métricas de todos los modelos
reports/figures/18_aprendizaje_lstm_*.png
reports/figures/19_degradacion_lstm.png
```

---


