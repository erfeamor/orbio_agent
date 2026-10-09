PYTHON ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install test lint typecheck run-text serve clean

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

install: $(BIN)/python
	$(BIN)/pip install -q -e ".[dev]"

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

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage
