PYTHON ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install test lint typecheck run-text serve frontend frontend-build clean

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

install: $(BIN)/python
	$(BIN)/pip install -q -e ".[dev,aws]"

test:
	$(BIN)/pytest --cov=agent --cov-report=term-missing --cov-fail-under=90

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

typecheck:
	$(BIN)/mypy src

run-text:
	$(BIN)/python -m agent.interfaces.cli $(ARGS)

serve:
	$(BIN)/uvicorn agent.interfaces.api:create_app --factory --reload --port 8000

frontend/node_modules: frontend/package.json
	cd frontend && npm install --no-audit --no-fund
	touch frontend/node_modules

frontend: frontend/node_modules
	cd frontend && npm run dev

frontend-build: frontend/node_modules
	cd frontend && npm run build

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage
