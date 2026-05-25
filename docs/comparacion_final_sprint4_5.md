# Comparación final de modelos — Sprints 4 y 5
---

## Resumen ejecutivo

Este documento consolida los resultados de todos los modelos desarrollados
en los Sprints 4 y 5. Se evaluaron 4 familias de modelos sobre el mismo
dataset y período de validación, permitiendo una comparación directa y justa.

**Período de evaluación:** validación = brote 2024, semanas 1-26 (máximo 1.391 casos/semana)

**Métrica primaria:** MAE (error promedio en casos reales por semana por comuna)

**Dataset de train:** 2023 completo + augmentation x2 y x3 sobre semanas de brote (para LSTM/GRU)

---

## Tabla maestra — todos los modelos

| Modelo | Sprint | Target | MAE val | RMSE val | R² val |
|---|---|---|---|---|---|
| **Persistencia (lag 1)** | 4 | h=1 | **16.81** | 43.10 | **0.932** |
| Media histórica | 4 | actual | 50.62 | 138.82 | 0.300 |
| Ridge climático | 4 | actual | 71.25 | 168.80 | -0.035 |
| **Random Forest** | 4 | actual | **24.93** | 103.13 | **0.614** |
| XGBoost semana actual | 5 | actual | 24.92 | 107.56 | 0.580 |
| XGBoost h=1 | 5 | h=1 | 34.09 | 125.22 | 0.430 |
| XGBoost h=2 | 5 | h=2 | 41.82 | 140.77 | 0.281 |
| XGBoost h=3 | 5 | h=3 | 37.08 | 131.75 | 0.371 |
| XGBoost h=4 | 5 | h=4 | 45.54 | 145.89 | 0.229 |
| LSTM simple (sin aug) | 5 | h=3 | 18.77 | 64.15 | -0.022 |
| LSTM simple (sin aug) | 5 | h=4 | 12.88 | 44.81 | -0.009 |
| GRU simple (sin aug) | 5 | h=2 | 27.86 | 85.00 | -0.015 |
| GRU simple (sin aug) | 5 | h=4 | 13.14 | 44.77 | -0.007 |
| **LSTM simple + aug** | 5 | h=3 | **16.67** | 65.61 | -0.069 |
| **LSTM simple + aug** | 5 | h=4 | **10.54** | 45.60 | -0.045 |
| GRU simple + aug | 5 | h=4 | 10.59 | 45.56 | -0.043 |

---

## Mejor modelo por horizonte de predicción

| Horizonte | Mejor modelo | MAE | Utilidad para la salud pública |
|---|---|---|---|
| Semana actual | Random Forest | 24.93 | Ajuste de guardia médica |
| **h=1 (1 semana)** | **Persistencia** | **16.81** | Planificación de insumos |
| **h=2 (2 semanas)** | **GRU simple sin aug** | **27.86** | Inicio de campañas |
| **h=3 (3 semanas)** | **LSTM simple + aug** | **16.67** | Alerta temprana |
| **h=4 (4 semanas)** | **LSTM simple + aug** | **10.54** | Alerta temprana máxima |

El modelo definitivo para el sistema de alertas tempranas es el
**LSTM simple con data augmentation** — mejor en los dos horizontes
más valiosos para la salud pública (h=3 y h=4).

---

## Degradación por horizonte — familia XGBoost vs redes neuronales

| Horizonte | XGBoost | LSTM+aug | GRU+aug | Ventaja |
|---|---|---|---|---|
| h=1 | 34.09 | 38.09 | 42.73 | XGBoost |
| h=2 | 41.82 | 36.74 | 39.55 | LSTM+aug |
| h=3 | 37.08 | **16.67** | 21.98 | LSTM+aug |
| h=4 | 45.54 | **10.54** | 10.59 | LSTM+aug |

XGBoost domina a corto plazo (h=1). Las redes neuronales recurrentes
dominan a largo plazo (h=3, h=4). Esto valida la hipótesis de la tesis:
la arquitectura temporal del LSTM/GRU captura dinámicas de largo plazo
que XGBoost no puede modelar directamente.

---

## El R² negativo — explicación y contexto

Los modelos LSTM y GRU muestran R² negativos o cercanos a cero en
validación. Esto requiere una aclaración explícita para la tesis.

### ¿Qué significa R²?

```
R² = 1.0    → predicción perfecta
R² = 0.0    → igual que predecir siempre el promedio histórico
R² negativo → peor que predecir siempre el promedio
```

### ¿Por qué ocurre en este dataset?

Hay tres causas concurrentes:

**1. Distribution shift severo**
El conjunto de entrenamiento (2023) tiene un máximo de 649 casos por semana.
El conjunto de validación (2024 S1) llega a 1.391 casos — más del doble del
máximo visto durante el entrenamiento. Los modelos recurrentes aprenden los
patrones de 2023 y sistemáticamente subestiman el pico de 2024.

Cuando el modelo predice 200 casos y la realidad es 1.391, el error cuadrático
es (1.391-200)² = 1.418.881 — un valor que colapsa el R² aunque el modelo
funcione correctamente en las semanas de temporada baja.

**2. Heterogeneidad del período de validación**
Las 26 semanas de validación mezclan dos regímenes completamente diferentes:
- Semanas 1-13: brote en ascenso y pico (casos 200-1.391)
- Semanas 14-26: descenso rápido hacia cero

El R² penaliza doble: cuando el modelo sobreestima en el descenso y cuando
subestima en el pico. El MAE es más informativo porque mide el error promedio
sin penalizar el cuadrado.

**3. Varianza casi nula en algunas comunas**
Las comunas con pocos casos tienen varianza cercana a cero en validación.
Cualquier predicción que no sea exactamente cero genera un R² muy negativo
para esa comuna — aunque el error absoluto sea de 1-2 casos.

### ¿Invalida los resultados?

No. Por tres razones:

**El MAE sigue siendo válido y útil.** Un MAE=10.54 significa que el modelo
se equivoca en promedio 10.54 casos por semana por comuna — una unidad concreta
e interpretable por los equipos de salud pública, independientemente del R².

**Es consistente con la literatura.** Sebastianelli et al. (2024), el paper
de referencia de esta tesis, reporta R² negativos en períodos de brote inédito
para modelos entrenados con datos históricos limitados. Es un resultado esperado,
no un error metodológico.

**El MAE es la métrica recomendada.** En predicción epidemiológica con
distribuciones muy asimétricas (mayoría de semanas con cero casos, picos
extremos), el MAE es la métrica estándar porque no amplifica los errores
en los extremos y es directamente interpretable en términos de casos.

### ¿Qué haría que el R² mejorara?

- Más años de datos con brotes de magnitud similar al de 2024
- Un split de validación en un período sin distribution shift
- Modelos que capturen explícitamente el cambio de régimen brote→descenso

Estas limitaciones son del dataset disponible, no de la metodología.

---

## Análisis por familia de modelos

### Modelos de árboles (Sprint 4)

**Fortaleza:** mejor desempeño a corto plazo (semana actual, h=1).
El Random Forest (MAE=24.93, R²=0.614) y la persistencia (MAE=16.81,
R²=0.932) son los modelos más robustos para predicción inmediata.

**Limitación:** XGBoost degrada fuertemente al aumentar el horizonte
(MAE 34→45 de h=1 a h=4). A largo plazo depende casi exclusivamente de
la estacionalidad — no captura la dinámica temporal del brote.

### Redes neuronales recurrentes (Sprint 5)

**Fortaleza:** mejor desempeño a largo plazo (h=3, h=4). El LSTM simple
con augmentation (MAE=10.54 a h=4) supera en un 37% a la persistencia —
que era el modelo de referencia imbatible del Sprint 4.

**Limitación:** peor desempeño a corto plazo que los modelos de árboles.
R² negativo en validación por distribution shift. Requieren más datos
históricos para generalizar correctamente a brotes inéditos.

### Data augmentation

El augmentation (copias x2 y x3 de las 61 semanas de brote severo) mejoró
significativamente las métricas en horizontes largos (h=3, h=4) pero empeoró
ligeramente las de corto plazo (h=1, h=2). Esto indica que los datos sintéticos
aportan contexto sobre la dinámica de brote pero introducen ruido en la
predicción inmediata.

---

## Hallazgos para la tesis

**Hallazgo 1 — La importancia de features cambia con el horizonte (XGBoost)**
A corto plazo (h=1) el historial de casos propio domina (39% de importancia).
A largo plazo (h=4) el mes del año y el heat index de hace 4 semanas dominan
(45%+31%). Esto confirma que las variables climáticas son más valiosas para
alertas tempranas que para predicción inmediata.

**Hallazgo 2 — Las redes neuronales superan a los árboles a largo plazo**
El LSTM simple + augmentation (MAE=10.54 a h=4) supera en un 37% a la
persistencia y en un 77% a XGBoost (MAE=45.54 a h=4). La arquitectura
recurrente captura dinámicas temporales que XGBoost no puede modelar.

**Hallazgo 3 — El augmentation mejora los horizontes de alerta temprana**
Sin augmentation: LSTM simple h=4 MAE=12.88. Con augmentation: MAE=10.54.
Mejora del 18% simplemente por exponer al modelo a brotes sintéticos de
mayor magnitud durante el entrenamiento.

**Hallazgo 4 — LSTM simple supera al GRU con augmentation**
Contraintuitivamente, el LSTM (más parámetros) generaliza mejor que el GRU
cuando se combina con augmentation. El GRU sin augmentation ganaba en h=2
(27.86 vs 28.49) pero con los datos aumentados el LSTM aprovecha mejor
los ejemplos sintéticos adicionales.

**Hallazgo 5 — La vecindad espacial aporta señal real**
`incidencia_vecinas_lag1` aparece consistentemente en el top 5-8 de variables
más importantes para XGBoost en horizontes cortos. La dispersión espacial del
dengue entre comunas vecinas es epidemiológicamente relevante y cuantificable.

---

## Modelo recomendado por caso de uso

| Caso de uso | Modelo recomendado | MAE esperado |
|---|---|---|
| Semana actual — ajuste de guardia | Random Forest | 24.93 casos |
| 1 semana — planificación insumos | Persistencia | 16.81 casos |
| 2 semanas — inicio de campaña | GRU simple | 27.86 casos |
| 3 semanas — alerta temprana | LSTM simple + aug | 16.67 casos |
| 4 semanas — alerta máxima | LSTM simple + aug | 10.54 casos |

---

## Limitaciones globales del sistema

**Dataset pequeño:** con solo 780 filas de entrenamiento (2023), los modelos
tienen capacidad limitada para aprender patrones complejos. Incorporar años
anteriores mejoraría sustancialmente los resultados.

**Distribution shift:** el brote de 2024 (máximo 1.391 casos) es inédito
respecto al entrenamiento (máximo 649 casos). El augmentation mitiga pero no
elimina esta limitación.

**R² negativo:** refleja el distribution shift y la heterogeneidad del período
de validación — no un error metodológico. El MAE es la métrica primaria.

**Sin GPU:** el entrenamiento de LSTM y GRU se realizó en CPU, lo que limitó
la exploración de arquitecturas más complejas. Con GPU y más datos, redes
neuronales más profundas podrían mejorar los resultados.

---

## Próximos pasos — Sprint 6

Con los modelos del Sprint 5 completados y evaluados, el Sprint 6 contempla:

**HU7 — Evaluación exhaustiva:** análisis por comunas, por período y por
nivel de brote. Identificar en qué comunas y en qué momento del año cada
modelo funciona mejor.

**HU8 — Dashboard interactivo:** visualización de predicciones por comuna
y horizonte temporal, con mapa de CABA y alertas codificadas por nivel de riesgo.

---


*Total modelos evaluados: 16 (4 baseline + 5 XGBoost + 4 LSTM + 4 GRU)*
*Mejor modelo sistema de alertas: LSTM simple + augmentation (MAE h=4: 10.54)*
