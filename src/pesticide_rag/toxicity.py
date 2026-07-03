import pandas as pd

from .config import WHO_CLASS_I_PATH
from .text_utils import normalize_name


class ToxicityChecker:
    def __init__(self, path=WHO_CLASS_I_PATH):
        self.items = []
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                name = normalize_name(str(row.get("active_ingredient", "")))
                if name:
                    self.items.append(
                        {
                            "active_ingredient": name,
                            "who_class": str(row.get("who_class", "Class I")),
                            "notes": str(row.get("notes", "")),
                        }
                    )

    def flag(self, text: str) -> dict:
        haystack = normalize_name(text)
        matches = [item for item in self.items if item["active_ingredient"] in haystack]
        return {
            "is_highly_hazardous": bool(matches),
            "matches": matches,
        }

