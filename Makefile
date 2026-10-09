PYTHON ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install lock test lint typecheck run-text serve frontend frontend-build api-types clean

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

# Exact versions from the lockfile, then the project itself without re-resolving.
install: $(BIN)/python
	$(BIN)/pip install -q -r requirements/dev.lock
	$(BIN)/pip install -q --no-deps -e .

# Re-pin after changing dependencies in pyproject.toml; commit the .lock files.
lock: $(BIN)/python
	$(BIN)/pip-compile -q --strip-extras --extra dev --extra aws -o requirements/dev.lock pyproject.toml
	$(BIN)/pip-compile -q --strip-extras --extra aws -o requirements/runtime.lock pyproject.toml

test:
	$(BIN)/pytest --cov=agent --cov-report=term-missing --cov-fail-under=90

lint:
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

typecheck:
	$(BIN)/mypy src

run-text:
	$(BIN)/python -m agent.interfaces.cli $(ARGS)

serve:
	$(BIN)/uvicorn agent.interfaces.api:app_factory --factory --reload --port 8000

frontend/node_modules: frontend/package.json
	cd frontend && npm install --no-audit --no-fund
	touch frontend/node_modules

frontend: frontend/node_modules
	cd frontend && npm run dev

frontend-build: frontend/node_modules
	cd frontend && npm run build

# Regenerate the frontend's TypeScript types after changing an API model.
api-types: frontend/node_modules
	$(BIN)/python scripts/export_openapi.py
	cd frontend && npm run api-types

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage
