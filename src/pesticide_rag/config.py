from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
MANIFEST_PATH = DATA_DIR / "metadata_manifest.csv"
WHO_CLASS_I_PATH = DATA_DIR / "who_class_i_seed.csv"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

