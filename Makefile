.PHONY: setup generate train serve test lint clean

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

generate:
	.venv/bin/python scripts/generate_data.py

train:
	.venv/bin/python scripts/train.py

serve:
	.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	.venv/bin/pytest --cov=src

lint:
	.venv/bin/ruff check src tests
	.venv/bin/mypy src

clean:
	rm -rf artifacts data/raw/*.parquet data/processed/*.parquet
