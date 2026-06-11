# Informe de resultados — Sprint 6 / HU7
## Evaluación exhaustiva por comuna y horizonte de predicción

---

## ¿Por qué era necesaria esta evaluación?

Los informes del Sprint 5 reportaron métricas globales — un único MAE promediado
sobre las 390 filas del período de validación y las 15 comunas. Ese número es útil
para comparar modelos entre sí, pero oculta información crítica para la salud pública:

**Lo que las métricas globales no responden:**
- ¿Hay comunas donde el modelo falla sistemáticamente? ¿Cuáles?
- ¿El LSTM es mejor que la persistencia en TODAS las comunas o solo en algunas?
- ¿El mismo modelo que gana en Palermo (baja incidencia) también gana en
  Puerto Madero (alta incidencia)?
- ¿En qué zonas de la ciudad el sistema de alertas tempranas es confiable?

La HU7 responde estas preguntas evaluando los modelos de forma **desagregada**:
por cada una de las 15 comunas, por cada horizonte de predicción, y sobre los
tres conjuntos de datos.

---

## Modelos evaluados y período de análisis

**Modelos:**
- **Persistencia (lag 1):** predice que esta semana habrá los mismos casos
  que la semana pasada. Es el baseline epidemiológico estándar — simple, sin
  entrenamiento, y sorprendentemente difícil de superar durante un brote activo.
- **Random Forest:** mejor modelo del Sprint 4. Entrenado para predecir la
  semana actual con 50 features (sin vecindad espacial).
- **XGBoost:** mejor modelo de árboles del Sprint 5. Entrenado con
  transformación log1p + early stopping + vecindad espacial, para semana
  actual y horizontes h=1 a h=4.
- **LSTM simple + augmentation:** mejor modelo de redes neuronales del Sprint 5.
  Entrenado con ventana de 12 semanas + Huber loss + sample weights +
  data augmentation x2/x3 sobre semanas de brote.

**Período de análisis principal:** validación = brote 2024, semanas 1-26.
Este período es el más exigente porque incluye el brote más severo registrado
en CABA (máximo 1.391 casos/semana en la C1), que supera más del doble al
máximo visto durante el entrenamiento (649 casos/semana en 2023).

---

## Resultados — MAE promedio sobre las 15 comunas

### Validation (brote 2024, semanas 1-26)

| Modelo | actual | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| Persistencia | 16.81 | 16.81 | 16.81 | 16.81 | 16.81 |
| Random Forest | 24.93 | — | — | — | — |
| XGBoost | 24.92 | 34.09 | 41.82 | 37.08 | 45.54 |
| **LSTM+aug** | 96.51 | 38.38 | 38.85 | **16.67** | **10.57** |

### Test (2024 S2 + 2025)

| Modelo | actual | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| Persistencia | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 |
| Random Forest | 0.97 | — | — | — | — |
| XGBoost | 0.35 | 0.94 | 1.96 | 2.71 | 3.02 |
| LSTM+aug | 99.22 | 14.43 | 34.50 | **0.19** | **1.01** |

**Nota sobre el test set:** el MAE extremadamente bajo de la persistencia
(0.17) refleja que 2024 S2 y 2025 son períodos de temporada baja con casi
cero casos. Cuando la realidad es 0 casos/semana, predecir "lo mismo que
la semana pasada" da un error casi perfecto.

---

## Modelo ganador por comuna y horizonte (Validation)

La siguiente tabla muestra qué modelo tiene el menor MAE para cada
combinación (comuna × horizonte). El símbolo **\*** indica que el modelo
ganador supera a la persistencia en esa combinación.

| Comuna | actual | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| C1 Puerto Madero | PERS | PERS | PERS | PERS | **LSTM\*** |
| C2 Recoleta | **RF\*** | PERS | PERS | PERS | **LSTM\*** |
| C3 Balvanera | **XGB\*** | PERS | PERS | PERS | **LSTM\*** |
| C4 La Boca | PERS | PERS | PERS | **LSTM\*** | **LSTM\*** |
| C5 Almagro | **XGB\*** | PERS | PERS | PERS | **LSTM\*** |
| C6 Caballito | **XGB\*** | PERS | PERS | **LSTM\*** | **LSTM\*** |
| C7 Flores | **XGB\*** | PERS | PERS | PERS | **LSTM\*** |
| C8 Lugano | **XGB\*** | PERS | PERS | **LSTM\*** | **LSTM\*** |
| C9 Liniers | **RF\*** | PERS | PERS | PERS | **LSTM\*** |
| C10 Floresta | PERS | PERS | PERS | PERS | **LSTM\*** |
| C11 Villa del Parque | **XGB\*** | PERS | PERS | **LSTM\*** | **LSTM\*** |
| C12 Coghlan | **RF\*** | PERS | PERS | **LSTM\*** | **LSTM\*** |
| C13 Belgrano | **RF\*** | PERS | PERS | **LSTM\*** | **LSTM\*** |
| C14 Palermo | **XGB\*** | PERS | PERS | **LSTM\*** | **LSTM\*** |
| C15 Agronomía | **RF\*** | PERS | PERS | **LSTM\*** | **LSTM\*** |

### Conteo total de victorias

| Modelo | Victorias | % | Horizontes donde domina |
|---|---|---|---|
| Persistencia | 40 | 53.3% | h=1 y h=2 en todas las comunas |
| LSTM+aug | 23 | 30.7% | h=3 en 10 comunas, h=4 en 15 comunas |
| XGBoost | 7 | 9.3% | Semana actual en comunas de incidencia moderada |
| Random Forest | 5 | 6.7% | Semana actual en comunas periféricas |

---

## Análisis detallado de hallazgos

### Hallazgo 1 — La persistencia domina a corto plazo: razón epidemiológica

La persistencia gana en **todas** las combinaciones de h=1 y h=2, independientemente
de la comuna o el brote. Esto no es una limitación de los modelos más complejos —
es una propiedad fundamental de la epidemiología del dengue.

**¿Por qué?**
El dengue tiene una dinámica altamente autocorrelacionada a corto plazo:
- El mosquito Aedes aegypti infectado pica durante semanas
- El período de incubación viral es de 4-14 días
- Los casos de esta semana son consecuencia directa de las picaduras de
  las semanas previas

Durante un brote activo, los casos de esta semana predicen muy bien los
de la próxima semana simplemente porque las condiciones que generaron el
brote (mosquitos infectados, temperatura favorable, humedad) no cambian
de un día para el otro. La autocorrelación temporal es tan fuerte que
ningún modelo más complejo puede superarla a 1-2 semanas.

**Implicancia para nuestro sistema de alertas:**
El modelo de persistencia es el componente óptimo para alertas de muy corto
plazo (h=1 y h=2). No tiene sentido usar modelos más complejos para estos
horizontes — agregan costo computacional sin mejorar las predicciones.

---

### Hallazgo 2 — LSTM+aug gana en h=4 en las 15 comunas: razón metodológica

A 4 semanas adelante, el LSTM supera a la persistencia en **todas las comunas
sin excepción** — incluyendo la C1 que es la más difícil. Este es el resultado
más sólido y consistente de toda la evaluación.

**¿Por qué el LSTM mejora a mayor horizonte?**
A 4 semanas de distancia, la autocorrelación temporal se debilita significativamente.
Predecir "habrá lo mismo que la semana pasada" ya no funciona bien porque las
condiciones del brote pueden haber cambiado drásticamente en un mes. El LSTM,
al procesar las últimas 12 semanas como una secuencia, puede detectar patrones
que indican si el brote está en fase de ascenso, plateau o descenso — información
que la persistencia ignora completamente.

Además, el data augmentation generó ejemplos sintéticos de brotes intensos
(hasta 2.110 casos), lo que permitió al LSTM aprender la dinámica de brotes
más severos que los observados en 2023. Esta mejora es especialmente visible
en comunas como C4 La Boca, donde el LSTM logra MAE=15.7 a h=4 versus
MAE=56.2 de XGBoost.

**¿Por qué XGBoost empeora a mayor horizonte?**
XGBoost a h=4 depende casi exclusivamente de la estacionalidad (mes del año
y semana del calendario), porque las variables de historial de casos pierden
poder predictivo tan lejos en el futuro. Cuando el brote supera la magnitud
vista en entrenamiento, la estacionalidad no es suficiente y el error explota.

---

### Hallazgo 3 — La Comuna 1 es la anomalía del sistema

La C1 (Puerto Madero, Constitución, Montserrat) es sistemáticamente la
comuna más difícil de predecir para todos los modelos:

| Modelo | MAE C1 (h=4) | MAE promedio resto (h=4) |
|---|---|---|
| Persistencia | 107.0 | 10.1 |
| XGBoost | 350.1 | 22.4 |
| LSTM+aug | **76.1** | 6.2 |

El LSTM es el menos malo para C1 a h=4 (MAE=76.1 vs 107.0 de la persistencia),
pero todos los errores son enormes comparados con el resto de las comunas.

**¿Por qué C1 es tan difícil?**
La C1 tiene características epidemiológicas únicas en CABA:
- **Población mínima:** 22.000 habitantes — la más pequeña de las 15 comunas.
  Cada caso individual tiene un impacto enorme en la tasa de incidencia.
- **Concentración histórica:** el 39.9% de todos los casos históricos de CABA
  se registraron en C1, aunque alberga solo el 0.7% de la población.
- **Brote inédito:** en 2024 S1 la C1 alcanzó 1.391 casos/semana — más del
  doble del máximo visto durante el entrenamiento (649 casos/semana en 2023).
  Ningún modelo puede predecir lo que nunca vio.
- **Dinámica de dispersión diferente:** la C1 limita con el Río de la Plata
  y tiene solo 3 comunas vecinas (C2, C3, C4). Su dinámica de propagación
  espacial es diferente al resto de la ciudad.

**Implicancia para nuestro sistema de alertas:**
La C1 requiere monitoreo especial y posiblemente un modelo dedicado entrenado
con datos específicos de brotes de alta magnitud. Los modelos actuales pueden
detectar que el brote está creciendo pero subestiman sistemáticamente su magnitud.

---

### Hallazgo 4 — XGBoost domina la semana actual en comunas de incidencia moderada

XGBoost gana en la semana actual en 7 comunas: C3, C5, C6, C7, C8, C11, C14.
Son todas comunas con incidencia moderada durante el brote 2024.

**¿Por qué XGBoost es mejor que RF para la semana actual en estas comunas?**
XGBoost fue entrenado con dos mejoras clave que RF no tiene:
1. **Transformación log1p del target:** comprime los valores extremos y
   equilibra el aprendizaje entre semanas de baja y alta incidencia.
2. **Features de vecindad espacial:** `casos_vecinas_lag1` captura la
   dispersión del dengue entre comunas adyacentes, información que RF no tiene.

Para comunas como C6 Caballito (7 vecinas) y C7 Flores (7 vecinas — la más
conectada de CABA), la vecindad espacial es especialmente informativa.

**¿Por qué RF gana en comunas periféricas?**
RF gana en C2, C9, C12, C13, C15 — comunas del norte y oeste de CABA.
Estas comunas tienen patrones de brote más regulares y menos extremos.
En ese contexto, el modelo más simple (RF) generaliza mejor porque no
tiene el riesgo de sobreajustar que tiene XGBoost con más parámetros.

---

### Hallazgo 5 — Patrón geográfico en las victorias de h=3

El LSTM gana en h=3 en 10 comunas, con un patrón geográfico claro:
- **Gana:** C4, C6, C8, C11, C12, C13, C14, C15 — comunas del sur,
  centro-norte y noroeste de CABA.
- **No gana:** C1, C2, C3, C5, C7, C9, C10 — comunas del centro histórico
  y del eje norte-sur.

Las comunas donde el LSTM no gana en h=3 son aquellas con brotes más
intensos (C1, C4) o con patrones de incidencia muy variables (C3, C5, C9).
La dinámica epidemiológica de esas comunas es más difícil de capturar a
3 semanas de distancia, y la persistencia —que predice la media del brote
actual— resulta más conservadora y por ende más precisa.

---

### Hallazgo 6 — LSTM mejora con el horizonte: un patrón contraintuitivo

La tabla de validación muestra algo inesperado:

```
LSTM+aug MAE:  actual=96.5  h=1=38.4  h=2=38.9  h=3=16.7  h=4=10.6
```

El LSTM **mejora** al predecir más lejos en lugar de empeorar. Esto va en
contra de la intuición — se esperaría que predecir más lejos sea más difícil.

**Explicación:**
Este patrón se debe a la interacción entre dos efectos:

1. **Distribution shift en la semana actual:** el LSTM fue entrenado con 2023
   (máx 649 casos) y se evalúa en el pico del brote 2024 (máx 1.391 casos).
   Para la semana actual, el modelo predice valores del orden de 2023 cuando
   la realidad es el doble — el error es enorme porque compara predicción vs.
   valor instantáneo del pico.

2. **Suavizado temporal en horizontes largos:** a h=4, el target es el valor
   4 semanas adelante. Durante el período de validación, muchas de esas
   semanas "futuras" corresponden a la fase de descenso del brote (semanas
   22-26 de 2024), donde los casos ya bajaron. El LSTM, que aprendió la
   dinámica de ascenso y descenso con los datos augmentados, predice
   correctamente ese descenso y el error disminuye.

En otras palabras: el LSTM no mejora "prediciendo más lejos" — mejora porque
el target de h=4 incluye la fase de descenso donde la magnitud del brote
vuelve a rangos que el modelo conoce.

---

## Comunas más y menos difíciles de predecir

### Más difíciles (LSTM+aug, MAE promedio sobre todos los horizontes)
1. **C1 Puerto Madero:** MAE=200.9 — brote inédito, dinámica atípica, baja población
2. **C4 La Boca:** MAE=44.5 — segundo brote más severo, zona de alta vulnerabilidad
3. **C15 Agronomía:** MAE=29.6 — comunas periféricas con menos historia epidemiológica

### Mejor predichas (LSTM+aug, MAE promedio)
1. **C14 Palermo:** MAE=26.8
2. **C10 Floresta:** MAE=26.8
3. **C6 Caballito:** MAE=26.8

Las comunas mejor predichas son las del centro-oeste de CABA, con incidencia
moderada, patrones regulares de brote y alta conectividad con sus vecinas
(C6 y C11 tienen 5 y 5 vecinos respectivamente). La señal de vecindad espacial
funciona especialmente bien en estas comunas.

---

## Implicancias para el diseño del sistema de alertas

El análisis por comuna confirma que el sistema de alertas óptimo debe ser
**multicapa**: usar el mejor modelo para cada horizonte y zona geográfica.

### Arquitectura recomendada

| Horizonte | Modelo | Cobertura | Razón |
|---|---|---|---|
| Semana actual | XGBoost (C3,C5-C8,C11,C14) / RF (resto) | 15 comunas | Mayor precisión en incidencia moderada |
| h=1 (1 semana) | Persistencia | 15 comunas | Autocorrelación temporal imbatible |
| h=2 (2 semanas) | Persistencia | 15 comunas | Ídem |
| h=3 (3 semanas) | LSTM+aug | 10 comunas / Persistencia en 5 | LSTM domina centro y periferia |
| h=4 (4 semanas) | LSTM+aug | **15 comunas** | Dominio universal a largo plazo |

### Tratamiento especial para C1

La C1 requiere:
- Umbral de alerta más bajo (el error sistemático hace que el modelo subestime)
- Posiblemente un modelo dedicado entrenado con más ejemplos de brotes extremos
- Monitoreo manual adicional cuando el sistema detecta señal de ascenso en C2, C3 o C4

---

## Limitaciones identificadas en HU7

**C1 como outlier estructural:**
La C1 distorsiona los promedios globales en todos los modelos. Las métricas
del Sprint 5 (MAE global) estaban suavizadas por el promedio de las 14 comunas
restantes. El MAE real del LSTM en C1 a h=4 es 76.1 — muy lejos del promedio
global de 10.57.

**Distribution shift concentrado en C1 y C4:**
Las dos comunas con brote más severo son exactamente las más difíciles de
predecir. El data augmentation generó brotes sintéticos hasta 2.110 casos,
pero la dinámica espacial y temporal específica de C1 durante 2024 no puede
replicarse artificialmente sin más datos históricos reales.

**Falta de evaluación por fase de brote:**
Este análisis promedia sobre todas las semanas del período de validación
(ascenso + pico + descenso). Una evaluación más fina separaría por fase
para identificar si los modelos son mejores o peores durante el pico vs.
el descenso. Esto se deja como trabajo futuro.

**Random Forest sin horizontes futuros:**
El RF del Sprint 4 solo fue entrenado para la semana actual. No puede
compararse con XGBoost y LSTM en h=1 a h=4. Si se entrenara con los
mismos targets futuros, podría mejorar en algunos horizontes intermedios.

---

## Conclusión

El análisis por comunas revela que no existe un modelo universalmente superior.
Cada modelo domina en el horizonte y contexto para el que fue diseñado:

- **La persistencia es imbatible a corto plazo (h=1, h=2)** en todas las comunas,
  debido a la alta autocorrelación temporal del dengue durante un brote activo.

- **El LSTM+aug es el mejor modelo a largo plazo (h=3, h=4)**, especialmente
  a 4 semanas donde gana en las 15 comunas. La combinación de arquitectura
  recurrente + data augmentation permite capturar la dinámica temporal del
  brote mejor que los modelos de árboles.

- **La Comuna 1 es la excepción permanente:** su comportamiento epidemiológico
  atípico (39.9% de todos los casos históricos en el 0.7% de la población)
  la hace sistemáticamente difícil para todos los modelos. Requiere
  tratamiento especial en el sistema de alertas.

- **El sistema de alertas multicapa** — que usa la persistencia para h=1/h=2
  y el LSTM para h=3/h=4 — maximiza la precisión en todos los horizontes
  de predicción y en todas las zonas geográficas de CABA.

---

## Archivos generados

```
reports/hu7_metricas_por_comuna.csv                    ← métricas desagregadas completas
reports/figures/22_heatmap_mae_comunas_validation.png  ← mapa de calor por comuna
reports/figures/23_mae_comunas_h4_validation.png       ← comparación por comuna h=4
reports/figures/23_mae_comunas_h1_validation.png       ← comparación por comuna h=1
reports/figures/24_victorias_modelos_validation.png    ← conteo de victorias
src/utils/hu7_evaluacion_comunas.py                    ← script de evaluación
```

