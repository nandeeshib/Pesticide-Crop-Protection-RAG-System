import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9.%/-]+", text.lower())


def normalize_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

