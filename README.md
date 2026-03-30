# Predicción espacio-temporal de brotes de dengue en CABA

**Trabajo Final — Carrera de Especialización en Inteligencia Artificial**  
Facultad de Ingeniería — Universidad de Buenos Aires (FIUBA)  
Autora: Ing. Paola Andrea Blanco Blanco  
Directora: Esp. Lic. María Carina Roldán  

---

## Descripción

Sistema de predicción de brotes de dengue en la Ciudad Autónoma de Buenos Aires (CABA) que integra datos epidemiológicos, climáticos y geográficos para generar alertas tempranas a nivel de **comuna** y **semana epidemiológica**, con un horizonte predictivo de 1 a 4 semanas.

El modelo se basa en la arquitectura ensemble propuesta por [Sebastianelli et al. (2024)](https://www.nature.com/articles/s41598-024-52796-9), adaptada al contexto de CABA con datos de fuentes públicas argentinas.

---

## Estructura del proyecto

```
dengue-caba-forecast/
│
├── data/
│   ├── raw/            # Datos originales sin modificar (BEN, BES, SMN)
│   ├── processed/      # Datos limpios y alineados por semana epidemiológica
│   └── external/       # GeoJSON comunas, normales climáticas históricas
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Análisis exploratorio espacio-temporal
│   ├── 02_feature_engineering.ipynb  # Construcción de features
│   ├── 03_model_baseline.ipynb   # Modelo baseline (SARIMA / Poisson)
│   ├── 04_model_xgboost.ipynb    # XGBoost con variables rezagadas
│   ├── 05_model_lstm.ipynb       # LSTM para series temporales
│   └── 06_ensemble.ipynb         # Modelo ensemble final
│
├── src/
│   ├── data/
│   │   ├── ingestion.py          # Descarga y carga de fuentes (BEN, BES, SMN)
│   │   ├── cleaning.py           # Limpieza, imputación, validación
│   │   └── spatial.py            # Georreferenciación y matriz de vecindad
│   │
│   ├── features/
│   │   ├── lags.py               # Variables rezagadas temporales (t-1 a t-4)
│   │   ├── climate.py            # Índices climáticos y anomalías
│   │   └── spatial_feats.py      # Features de comunas vecinas
│   │
│   ├── models/
│   │   ├── baseline.py           # Modelos estadísticos de referencia
│   │   ├── xgboost_model.py      # XGBoost / LightGBM
│   │   ├── lstm_model.py         # LSTM / GRU (TensorFlow/Keras)
│   │   └── ensemble.py           # Modelo ensemble (basado en Sebastianelli 2024)
│   │
│   └── utils/
│       ├── metrics.py            # MAE, RMSE, MAPE, AUC-ROC por comuna
│       ├── alerts.py             # Sistema de alertas estratificadas
│       └── visualization.py      # Mapas, series temporales, heatmaps
│
├── models/
│   └── saved/          # Modelos entrenados serializados (.pkl, .h5)
│
├── reports/
│   └── figures/        # Gráficos y visualizaciones exportadas
│
├── tests/              # Tests unitarios
├── dashboard/          # Dashboard interactivo (opcional)
│
├── config.yaml         # Parámetros globales del proyecto
├── requirements.txt    # Dependencias Python
├── Makefile            # Comandos de automatización
└── .gitignore
```

---

## Fuentes de datos

| Fuente | Datos | Acceso |
|--------|-------|--------|
| [datos.gob.ar](https://datos.gob.ar) | Casos confirmados de dengue | Libre |
| Boletín Epidemiológico Nacional (BEN) | Semanas epidemiológicas nacionales | Público |
| Boletín Epidemiológico Semanal CABA (BES) | Casos por comuna CABA | Público |
| [Servicio Meteorológico Nacional (SMN)](https://www.smn.gob.ar) | Temperatura, humedad, precipitaciones | Libre |
| [BA en Datos](https://data.buenosaires.gob.ar) | Variables climáticas GCBA | Open Data |

---

## Modelo de referencia

Este proyecto adapta la arquitectura de:

> Sebastianelli et al. (2024). *A reproducible ensemble machine learning approach to forecast dengue outbreaks*. Scientific Reports, 14, 3807.  
> DOI: [10.1038/s41598-024-52796-9](https://doi.org/10.1038/s41598-024-52796-9)  
> Repositorio original: [ESA-PhiLab/ESA-UNICEF_DengueForecastProject](https://github.com/ESA-PhiLab/ESA-UNICEF_DengueForecastProject)

**Adaptaciones principales:**
- Unidad espacial: comunas de CABA (15) en lugar de estados/departamentos
- Resolución temporal: semana epidemiológica en lugar de mensual
- Fuentes de datos: BEN/BES/SMN/BA en Datos (datos públicos argentinos)
- Horizonte predictivo: 1 a 4 semanas

---

## Instalación

```bash
git clone https://github.com/tu-usuario/dengue-caba-forecast.git
cd dengue-caba-forecast
pip install -r requirements.txt
```

---

## Uso rápido

```bash
# Descargar y preparar datos
make data

# Ejecutar análisis exploratorio
jupyter notebook notebooks/01_EDA.ipynb

# Entrenar modelo
make train

# Generar predicciones y alertas
make predict
```

---

## Métricas de éxito

| Métrica | Objetivo |
|---------|----------|
| Tasa de detección de brotes | > 80% |
| Precisión de alertas | > 70% |
| Horizonte predictivo útil | ≥ 2 semanas |
| MAE por comuna | Minimizar |

---

## Stack tecnológico

- **Python 3.10+**
- `scikit-learn`, `xgboost`, `lightgbm` — modelos ML
- `tensorflow` / `keras` — LSTM/GRU
- `pandas`, `numpy` — procesamiento de datos
- `geopandas`, `shapely` — datos espaciales
- `plotly`, `matplotlib`, `folium` — visualización
- `pytest` — testing

---

## Consideraciones éticas

Este sistema genera **recomendaciones**, no decisiones automáticas. Ver sección 12.2 del plan de proyecto para el análisis completo de sesgos y medidas de mitigación. Las predicciones deben siempre ser validadas por profesionales de salud.

---

## Licencia

MIT License — uso libre para fines académicos y de investigación.

---

## Contacto

Paola Andrea Blanco Blanco — FIUBA, Carrera de Especialización en IA  
Dirección: Esp. Lic. María Carina Roldán
