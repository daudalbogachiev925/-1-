# Demand Forecast & Inventory Optimization

End-to-end иерархическое прогнозирование спроса с оптимизацией страховых запасов.

## Бизнес-задача
- Снизить stockout rate с 15% до ≤7.5% (−50%).
- Снизить overstock (избыточные запасы) на 20%.
- Дать API для интеграции с WMS/ERP.

## Иерархия

Total
├── Region (North, South, East, West)
│ └── Category (Electronics, Apparel, Grocery)
│ └── SKU (40 SKU)


## Модели
- LightGBM (global, с категориальными признаками)
- Prophet (per-SKU baseline)
- PyTorch MLP (global, embeddings)
- Иерархическая согласованность: MinT (OLS, WLS, MinT-Sample)

## Метрики
- WAPE, sMAPE, MASE, Bias
- Temporal cross-validation (rolling origin, 4 folds)
- Stockout rate, Overstock rate, Fill rate, Service level

## Быстрый старт
```bash
make setup
make generate
make train
make serve
curl http://localhost:8000/forecast?sku=SKU-001&horizon=28

стек
Python 3.11, Polars, DuckDB, LightGBM, Prophet, PyTorch, FastAPI, Pydantic v2, Docker.

---

## `pyproject.toml`

```toml
[project]
name = "demand-forecast-inventory"
version = "0.1.0"
description = "Hierarchical demand forecasting and inventory optimization"
requires-python = ">=3.11"
dependencies = [
    "polars>=0.20.0",
    "duckdb>=0.10.0",
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.4",
    "lightgbm>=4.3",
    "prophet>=1.1.5",
    "torch>=2.2",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "great-tables>=0.6",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "joblib>=1.3",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.3",
    "mypy>=1.9",
]

[project.scripts]
dfi = "src.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true

