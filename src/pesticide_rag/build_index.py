import json
import pickle

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import CHUNKS_PATH, EMBED_MODEL_NAME, INDEX_DIR
from .text_utils import tokenize


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("No chunks found. Run python -m src.pesticide_rag.ingest first.")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    tokenized = [tokenize(c["text"]) for c in tqdm(chunks, desc="Building BM25")]
    bm25 = BM25Okapi(tokenized)

    with (INDEX_DIR / "chunks.pkl").open("wb") as f:
        pickle.dump(chunks, f)
    with (INDEX_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    print(f"Indexed {len(chunks)} chunks in {INDEX_DIR}")


if __name__ == "__main__":
    main()

