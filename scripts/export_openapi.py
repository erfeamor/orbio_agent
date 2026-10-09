"""Write the API's OpenAPI schema to frontend/openapi.json (input for the TypeScript types).

Run with ``make api-types``. tests/test_api_schema.py fails if the committed file is stale.
"""

import json
import sys
import tempfile
from pathlib import Path

from agent.config import Settings
from agent.interfaces.api import create_app

OUTPUT = Path(__file__).parents[1] / "frontend" / "openapi.json"


def openapi_schema() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        # Building the app makes no network calls; keys are irrelevant to the schema.
        settings = Settings(_env_file=None, conversations_dir=Path(tmp))  # type: ignore[call-arg]
        schema = create_app(settings=settings).openapi()
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    OUTPUT.write_text(openapi_schema(), encoding="utf-8")
    print(f"wrote {OUTPUT}", file=sys.stderr)
