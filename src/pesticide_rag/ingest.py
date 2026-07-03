import json
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal environments
    def tqdm(iterable=None, **kwargs):
        return iterable

from .config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, MANIFEST_PATH, PROCESSED_DIR, RAW_DIR


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        pages.append({"page": page_no, "text": normalize_text(page.extract_text() or "")})
    return pages


def read_text(path: Path) -> list[dict]:
    return [{"page": 1, "text": normalize_text(path.read_text(encoding="utf-8", errors="ignore"))}]


def format_structured_recommendation(row: pd.Series) -> str:
    fields = {
        "Crop": row.get("crop", ""),
        "Pest": row.get("pest", ""),
        "Recommended pesticide": row.get("recommended_pesticide", ""),
        "Dose": row.get("dose", ""),
        "Dose per hectare": row.get("dose_per_ha", ""),
        "Waiting period": row.get("waiting_period", ""),
        "Water dilution": row.get("water_dilution", ""),
        "Safety precautions": row.get("safety_precautions", ""),
        "Source": row.get("source", ""),
    }
    return "\n\n".join(f"{key}:\n{value}" for key, value in fields.items() if str(value).strip())


def read_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    structured_cols = {"crop", "pest", "recommended_pesticide", "dose", "waiting_period"}
    if structured_cols.issubset({col.lower() for col in df.columns}):
        df.columns = [col.lower() for col in df.columns]
        return [
            {"page": int(i) + 1, "text": normalize_text(format_structured_recommendation(row))}
            for i, row in df.iterrows()
        ]

    rows = []
    for i, row in df.iterrows():
        text = " | ".join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
        rows.append({"page": int(i) + 1, "text": normalize_text(text)})
    return rows


def load_document(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".csv", ".tsv"}:
        return read_csv(path)
    if suffix in {".txt", ".md"}:
        return read_text(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
        else:
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + chunk_size])
                start += max(1, chunk_size - overlap)
            current = ""
    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = []
    previous_tail = ""
    for chunk in chunks:
        merged = f"{previous_tail} {chunk}".strip()
        overlapped.append(merged[: chunk_size + overlap])
        previous_tail = chunk[-overlap:]
    return overlapped


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["priority"] = pd.to_numeric(manifest.get("priority", 1.0), errors="coerce").fillna(1.0)
    return manifest


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    chunks = []
    for _, meta in tqdm(manifest.iterrows(), total=len(manifest), desc="Ingesting documents"):
        path = RAW_DIR / str(meta["file_name"])
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        pages = load_document(path)
        for page in pages:
            for chunk_no, chunk in enumerate(chunk_text(page["text"]), start=1):
                if len(chunk.strip()) < 80:
                    continue
                chunk_id = f"{path.stem}-p{page['page']}-c{chunk_no}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "file_name": path.name,
                        "page": page["page"],
                        "source_type": str(meta.get("source_type", "")),
                        "crop": str(meta.get("crop", "")),
                        "pest": str(meta.get("pest", "")),
                        "document_title": str(meta.get("document_title", path.stem)),
                        "publisher": str(meta.get("publisher", "")),
                        "year": str(meta.get("year", "")),
                        "url": str(meta.get("url", "")),
                        "priority": float(meta.get("priority", 1.0)),
                    }
                )

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
