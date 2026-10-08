PYTHON ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install test lint typecheck clean

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

install: $(BIN)/python
	$(BIN)/pip install -q -e ".[dev]"

test:
	$(BIN)/pytest --cov=agent --cov-report=term-missing

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

typecheck:
	$(BIN)/mypy src

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage
