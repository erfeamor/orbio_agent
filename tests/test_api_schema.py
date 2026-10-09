"""The frontend's TypeScript types are generated from frontend/openapi.json; keep it current."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_committed_openapi_schema_matches_the_api() -> None:
    spec = importlib.util.spec_from_file_location(
        "export_openapi", ROOT / "scripts" / "export_openapi.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    committed = (ROOT / "frontend" / "openapi.json").read_text(encoding="utf-8")
    assert committed == module.openapi_schema(), (
        "frontend/openapi.json is stale: run `make api-types` and commit the result"
    )
