.PHONY: format lint test check

QUALITY_PATHS = src/config.py src/inference src/training src/features/graph_features.py deployment tests scripts/check_deployment.py

format:
	ruff format $(QUALITY_PATHS)

lint:
	ruff check $(QUALITY_PATHS)

test:
	pytest -q

check:
	ruff format --check $(QUALITY_PATHS)
	ruff check $(QUALITY_PATHS)
	pytest -q
