import json
import os
from pathlib import Path
from typing import Any

class JSONRenderer:
    def __init__(self, schema_dir: Path = None):
        if schema_dir is None:
            self.schema_dir = Path(__file__).parent / "response_schemas"

    def render(self, file_name: str) -> Any:
        """Render JSON data."""
        try:
            with open(os.path.join(self.schema_dir, file_name), 'r', encoding='utf-8') as f:
                self.data = json.load(f)
                return self.data
        except Exception as e:
            print(f"Error loading JSON: {e}")
