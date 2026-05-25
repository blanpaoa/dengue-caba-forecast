# Informe de resultados — Sprint 5 (GRU + Data Augmentation)
---

## ¿Qué hicimos en este sprint?

Entrenamos y comparamos dos arquitecturas GRU (Gated Recurrent Unit) con el
dataset aumentado generado por `augmentation.py`. El objetivo fue determinar
si GRU, al tener menos parámetros que LSTM, generaliza mejor con los datos
sintéticos del augmentation.

Todas las configuraciones son idénticas al LSTM para garantizar comparación justa:
mismas features (44 variables), misma ventana temporal (12 semanas), mismo
tratamiento del desbalanceo (Huber loss + sample weights), mismos datos de
entrenamiento (train_augmented.parquet, 902 filas).

**La única diferencia:** capa recurrente GRU() en lugar de LSTM().

---

## ¿Qué es GRU y en qué se diferencia del LSTM?

GRU (Gated Recurrent Unit) es una variante simplificada del LSTM publicada
en 2014. Ambas son redes neuronales recurrentes diseñadas para aprender
de secuencias temporales, pero con diferente número de compuertas internas:

| | LSTM | GRU |
|---|---|---|
| Compuertas | 3 (olvido, entrada, salida) | 2 (reset, update) |
| Parámetros aprox. | ~10.000 por capa (32 unidades) | ~7.500 por capa (25% menos) |
| Velocidad | Más lenta | Más rápida |
| Con datos escasos | Mayor sobreajuste | Menor sobreajuste |

Con menos parámetros, el GRU tiene menos riesgo de memorizar el dataset —
especialmente importante con solo 722 secuencias de entrenamiento.

---

## Dataset utilizado — train_augmented.parquet

| Métrica | Dataset original | Dataset aumentado |
|---|---|---|
| Total filas | 780 | **902** |
| Semanas con 0 casos | 468 (60%) | 468 (51.9%) |
| Semanas con >50 casos | 61 (7.8%) | **183 (20.3%)** |
| Máximo de casos | 649 | **2.110** (sintético) |

---

## Resultados — GRU con augmentation

Período de validación: brote 2024, semanas 1–26.

### GRU simple (1 capa)

| Horizonte | MAE val | RMSE val | R² val | Interpretación |
|---|---|---|---|---|
| Semana actual | 89.96 | 159.02 | -0.068 | Distribution shift severo |
| h=1 (1 semana) | 42.73 | 121.84 | -0.007 | — |
| h=2 (2 semanas) | 39.55 | 85.13 | -0.018 | — |
| h=3 (3 semanas) | 21.98 | 63.52 | -0.002 | Cerca de la persistencia |
| h=4 (4 semanas) | **10.59** | 45.56 | -0.043 | Supera a la persistencia |

### GRU apilado (2 capas)

| Horizonte | MAE val | RMSE val | R² val | Interpretación |
|---|---|---|---|---|
| Semana actual | 56.94 | 154.28 | -0.006 | Mejor que simple |
| h=1 (1 semana) | **42.13** | 121.97 | -0.009 | Levemente mejor que simple |
| h=2 (2 semanas) | 71.66 | 98.28 | -0.357 | Inestabilidad con aug |
| h=3 (3 semanas) | **21.59** | 63.55 | -0.003 | Levemente mejor que simple |
| h=4 (4 semanas) | 10.82 | 45.43 | -0.037 | Similar al simple |

---

## Comparación GRU sin aug vs GRU con aug

| Modelo | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| GRU simple sin aug | 40.11 | 27.86 | 19.98 | 13.14 |
| **GRU simple con aug** | 42.73 | 39.55 | 21.98 | **10.59** |
| GRU apilado sin aug | 39.84 | 28.05 | 20.11 | 14.70 |
| GRU apilado con aug | 42.13 | 71.66 | **21.59** | **10.82** |

El augmentation mejoró h=4 en ambas arquitecturas pero empeoró h=1 y h=2.
El mismo patrón observado en LSTM: los datos sintéticos ayudan al largo plazo
pero introducen ruido en el corto plazo.

---

## GRU con aug vs LSTM con aug — comparación directa

| Modelo | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| **LSTM simple + aug** | **38.09** | **36.74** | **16.67** | **10.54** |
| GRU simple + aug | 42.73 | 39.55 | 21.98 | 10.59 |
| GRU apilado + aug | 42.13 | 71.66 | 21.59 | 10.82 |

**El LSTM simple con augmentation supera al GRU en todos los horizontes.**

Esto es contraintuitivo — se esperaba que el GRU (con menos parámetros)
generalizara mejor. La explicación es que con los datos aumentados el LSTM
tiene más capacidad para extraer patrones de los ejemplos sintéticos de brote
sin sobreajustar, gracias a que Huber loss y sample weights ya controlan el
desbalanceo de forma efectiva.

---

## Curvas de aprendizaje — diagnóstico

Las curvas del GRU con augmentation muestran el mismo patrón estructural
que el LSTM con augmentation:

```
GRU simple:  train ≈ 0.018, val ≈ 0.001  — val por debajo de train
GRU apilado: train ≈ 0.017, val ≈ 0.002  — similar
```

La curva de val ya no es perfectamente plana (como era sin augmentation) —
tiene varianza suave y baja gradualmente. El augmentation atacó correctamente
el distribution shift en ambas arquitecturas.

Sin embargo, la brecha entre train y val sigue siendo grande. Esto refleja
que el distribution shift entre 2023 y el brote 2024 es estructural —
no completamente corregible con augmentation sin datos históricos adicionales.

---

## El R² negativo — por qué ocurre y cómo interpretarlo

Todos los modelos del Sprint 5 muestran R² negativo o cercano a cero en
el período de validación. Esto requiere una explicación explícita.

**¿Qué significa R²?**
R² mide qué fracción de la variación real explica el modelo.
R²=1.0 es predicción perfecta. R²=0.0 es igual al promedio histórico.
R² negativo significa que el modelo predice peor que el promedio histórico.

**¿Por qué ocurre en este caso?**
El período de validación (2024 S1) mezcla dos regímenes muy distintos:
- Semanas 1-13: brote masivo (casos 200-1.391) — el modelo subestima
- Semanas 14-26: descenso rápido (casos 0-50) — el modelo sobreestima

Cuando el modelo predice "muchos casos" (aprendió del brote sintético)
pero ya estamos en la fase de descenso, el error es enorme y el R² colapsa.

**¿Invalida los resultados?**
No. El MAE sigue siendo válido — mide el error promedio en casos reales,
una unidad directamente interpretable por los equipos de salud pública.
El R² es especialmente sensible a la heterogeneidad del período de
validación y no refleja el desempeño real del modelo en cada fase.

**Consistencia con la literatura:** trabajos similares sobre predicción
de dengue con datos limitados (Sebastianelli et al. 2024, Epitech 2023)
reportan R² negativos o cercanos a cero en períodos de brote inédito.
El MAE es la métrica primaria recomendada en estos contextos.

---

## GRU simple vs GRU apilado — conclusión

| Criterio | GRU simple | GRU apilado |
|---|---|---|
| MAE h=4 | **10.59** | 10.82 |
| MAE h=3 | 21.98 | **21.59** |
| Estabilidad con aug | Alta | Baja (h=2: MAE=71.66) |
| Curvas de aprendizaje | Equilibradas | Similares |

El GRU simple es más estable con el dataset aumentado. El GRU apilado
muestra inestabilidad en h=2 (MAE saltó de 28 a 71 con augmentation) —
la misma limitación observada en el LSTM apilado.

**Modelo definitivo GRU:** GRU simple con augmentation.

---

## Archivos generados

```
models/saved/gru_gru_simple_h0.keras     ← semana actual
models/saved/gru_gru_simple_h1.keras     ← 1 semana adelante
models/saved/gru_gru_simple_h2.keras     ← 2 semanas adelante
models/saved/gru_gru_simple_h3.keras     ← 3 semanas adelante
models/saved/gru_gru_simple_h4.keras     ← 4 semanas adelante
models/saved/gru_gru_apilado_h*.keras    ← arquitectura apilada (5 modelos)
models/saved/gru_target_scaler.pkl       ← normalizador del target
models/saved/metricas_sprint5_gru.csv    ← métricas completas
reports/figures/20_aprendizaje_gru_*.png
reports/figures/21_gru_vs_lstm_horizontes.png
```

---

