# Makefile — dengue-caba-forecast
# Comandos de automatización del pipeline

.PHONY: help setup data clean train predict test lint format

## help: muestra esta ayuda
help:
	@echo "Comandos disponibles:"
	@echo ""
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' | sed -e 's/^/ /'

## setup: instala dependencias
setup:
	pip install -r requirements.txt

## data: descarga y procesa datos (ejecutar Sprint 1-2)
data:
	python src/data/ingestion.py --config config.yaml
	python src/data/cleaning.py  --config config.yaml
	python src/data/spatial.py   --config config.yaml
	@echo "Datos listos en data/processed/"

## features: genera features espacio-temporales (Sprint 4)
features:
	python -c "from src.features.lags import build_lag_features; build_lag_features()"
	python -c "from src.features.climate import build_climate_features; build_climate_features()"
	python -c "from src.features.spatial_feats import build_spatial_features; build_spatial_features()"
	@echo "Features generadas en data/processed/features.parquet"

## train: entrena todos los modelos (Sprint 4-5)
train: features
	python src/models/baseline.py   --config config.yaml
	python src/models/xgboost_model.py  --config config.yaml
	python src/models/lstm_model.py --config config.yaml
	python src/models/ensemble.py   --config config.yaml
	@echo "Modelos guardados en models/saved/"

## predict: genera predicciones y alertas (Sprint 6-7)
predict:
	python -c "from src.utils.alerts import generate_alerts; generate_alerts()"
	@echo "Predicciones y alertas generadas en reports/"

## evaluate: evalúa modelos (Sprint 6)
evaluate:
	python -c "from src.utils.metrics import evaluate_all; evaluate_all()"

## test: corre tests unitarios
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

## lint: verifica estilo de código
lint:
	flake8 src/ tests/ --max-line-length=88

## format: formatea código
format:
	black src/ tests/ notebooks/
	isort src/ tests/

## clean: limpia archivos temporales
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete
	@echo "Limpieza completada"

## notebook: abre Jupyter Lab
notebook:
	jupyter lab notebooks/
