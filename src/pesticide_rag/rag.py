import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import CHUNKS_PATH, EMBED_MODEL_NAME, INDEX_DIR, RERANK_MODEL_NAME
from .text_utils import tokenize
from .toxicity import ToxicityChecker


RECOMMENDATION_HINTS = re.compile(
    r"(dose|dosage|spray|apply|application|ml|g/ha|kg/ha|l/ha|ppm|waiting|pre[- ]harvest|phi|interval|safety)",
    re.IGNORECASE,
)


class PesticideRAG:
    def __init__(self, use_reranker: bool = True):
        self.simple_mode = not (INDEX_DIR / "faiss.index").exists()
        self.index = None
        self.bm25 = None
        self.embedder = None
        self.reranker = None

        if self.simple_mode:
            self.chunks = self.load_jsonl_chunks(CHUNKS_PATH)
        else:
            import faiss
            from sentence_transformers import CrossEncoder, SentenceTransformer

            self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
            with (INDEX_DIR / "chunks.pkl").open("rb") as f:
                self.chunks = pickle.load(f)
            with (INDEX_DIR / "bm25.pkl").open("rb") as f:
                self.bm25 = pickle.load(f)
            self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
            if use_reranker:
                try:
                    self.reranker = CrossEncoder(RERANK_MODEL_NAME)
                except Exception:
                    self.reranker = None

        self.toxicity = ToxicityChecker()

    def load_jsonl_chunks(self, path: Path) -> list[dict]:
        import json

        if not path.exists():
            raise RuntimeError("No chunks found. Run python -m src.pesticide_rag.ingest first.")
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def dense_search(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        vector = self.embedder.encode([query], normalize_embeddings=True)
        vector = np.asarray(vector, dtype="float32")
        scores, ids = self.index.search(vector, k)
        return [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx >= 0]

    def bm25_search(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def hybrid_search(self, query: str, crop: str = "", pest: str = "", k: int = 8) -> list[dict]:
        if self.simple_mode:
            return self.simple_search(query, crop=crop, pest=pest, k=k)

        dense = self.dense_search(query, k=40)
        lexical = self.bm25_search(query, k=40)
        fused = defaultdict(float)

        for rank, (idx, score) in enumerate(dense, start=1):
            fused[idx] += 0.60 * (1 / (60 + rank)) + 0.10 * score
        for rank, (idx, score) in enumerate(lexical, start=1):
            fused[idx] += 0.40 * (1 / (60 + rank)) + 0.02 * score

        crop_l = crop.lower().strip()
        pest_l = pest.lower().strip()
        for idx in list(fused):
            chunk = self.chunks[idx]
            if crop_l and crop_l not in {"all", str(chunk.get("crop", "")).lower()} and crop_l not in chunk["text"].lower():
                fused[idx] *= 0.75
            if pest_l and pest_l not in {"all", str(chunk.get("pest", "")).lower()} and pest_l not in chunk["text"].lower():
                fused[idx] *= 0.75
            fused[idx] *= float(chunk.get("priority", 1.0))

        candidates = sorted(fused, key=fused.get, reverse=True)[:20]

        if self.reranker and candidates:
            pairs = [(query, self.chunks[idx]["text"]) for idx in candidates]
            rerank_scores = self.reranker.predict(pairs)
            candidates = [
                idx for idx, _ in sorted(zip(candidates, rerank_scores), key=lambda item: item[1], reverse=True)
            ]

        results = []
        for idx in candidates[:k]:
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(fused[idx]), 4)
            results.append(chunk)
        return results

    def simple_search(self, query: str, crop: str = "", pest: str = "", k: int = 8) -> list[dict]:
        query_tokens = set(tokenize(" ".join([query, crop, pest])))
        scored = []
        for chunk in self.chunks:
            text = chunk["text"].lower()
            text_tokens = set(tokenize(text))
            overlap = len(query_tokens & text_tokens)
            if crop and crop.lower() in text:
                overlap += 5
            if pest and pest.lower() in text:
                overlap += 5
            if "recommended pesticide:" in text:
                overlap += 4
            if "waiting period:" in text:
                overlap += 2
            if "waiting period:\nnot stated" not in text and "waiting period:" in text:
                overlap += 5
            if "ppqs/cib&rc" in text:
                overlap += 2
            if overlap > 0:
                row = dict(chunk)
                row["score"] = round(overlap * float(chunk.get("priority", 1.0)), 4)
                scored.append(row)
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:k]

    def extract_evidence_lines(self, chunks: list[dict], max_lines: int = 5) -> list[dict]:
        evidence = []
        for chunk in chunks:
            if "Recommended pesticide:" in chunk["text"] and "Waiting period:" in chunk["text"]:
                evidence.append(
                    {
                        "text": chunk["text"],
                        "source": self.format_source(chunk),
                        "page": chunk.get("page", ""),
                        "url": chunk.get("url", ""),
                    }
                )
                if len(evidence) >= max_lines:
                    return evidence
                continue

            lines = re.split(r"(?<=[.;])\s+|\n+", chunk["text"])
            selected = [line.strip() for line in lines if RECOMMENDATION_HINTS.search(line)]
            if not selected:
                selected = [chunk["text"][:350].strip()]
            for line in selected[:2]:
                evidence.append(
                    {
                        "text": line,
                        "source": self.format_source(chunk),
                        "page": chunk.get("page", ""),
                        "url": chunk.get("url", ""),
                    }
                )
                if len(evidence) >= max_lines:
                    return evidence
        return evidence

    def format_source(self, chunk: dict) -> str:
        page = chunk.get("page", "")
        title = chunk.get("document_title", chunk.get("file_name", "source"))
        publisher = chunk.get("publisher", "")
        year = chunk.get("year", "")
        return f"{title}, {publisher} {year}, p. {page}".strip(", ")

    def answer(self, query: str, crop: str = "", pest: str = "") -> dict:
        chunks = self.hybrid_search(query, crop=crop, pest=pest)
        evidence = self.extract_evidence_lines(chunks)
        answer_text = self.compose_answer(query, evidence)
        risk = self.toxicity.flag(answer_text + "\n" + "\n".join(e["text"] for e in evidence))
        return {
            "query": query,
            "answer": answer_text,
            "risk": risk,
            "evidence": evidence,
            "contexts": chunks,
        }

    def parse_structured_evidence(self, text: str) -> dict:
        labels = [
            "Crop",
            "Pest",
            "Recommended pesticide",
            "Dose",
            "Dose per hectare",
            "Waiting period",
            "Water dilution",
            "Safety precautions",
            "Source",
        ]
        pattern = r"(?m)^(" + "|".join(re.escape(label) for label in labels) + r"):\s*$"
        matches = list(re.finditer(pattern, text))
        parsed = {}
        for i, match in enumerate(matches):
            label = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            parsed[label] = text[start:end].strip()
        return parsed

    def recommendation_sentence(self, item: dict, rank: int) -> str:
        crop = item.get("Crop", "the crop")
        pest = item.get("Pest", "the pest")
        pesticide = item.get("Recommended pesticide", "the listed pesticide")
        dose = item.get("Dose", "the dose stated in the source")
        waiting = item.get("Waiting period", "not stated in the retrieved source")
        water = item.get("Water dilution", "")
        precautions = item.get("Safety precautions", "")
        source = item.get("Source", "")

        if rank == 1:
            prefix = f"For {pest} in {crop}, the most relevant retrieved recommendation is {pesticide}."
        else:
            prefix = f"Another retrieved option is {pesticide} for {pest} in {crop}."

        parts = [
            prefix,
            f"Use it at {dose}.",
        ]
        if water and water.lower() not in {"not stated", "not stated in source"}:
            parts.append(f"The source lists water dilution as {water}.")
        if waiting and "not stated" not in waiting.lower():
            parts.append(f"Keep a waiting period of {waiting} before harvest.")
        else:
            parts.append("The waiting period was not stated in this retrieved source, so verify it on the product label before use.")
        if precautions:
            parts.append(f"Safety note: {precautions}")
        if source:
            parts.append(f"Source: {source}.")
        return " ".join(parts)

    def compose_answer(self, query: str, evidence: list[dict]) -> str:
        if not evidence:
            return (
                "I could not find strong cited evidence for this crop-pest query. "
                "Check the registered label and local extension guidance before use."
            )

        structured = [self.parse_structured_evidence(item["text"]) for item in evidence]
        structured = [item for item in structured if item.get("Recommended pesticide")]

        if structured:
            lines = [
                "Here is the pesticide guidance I found from the retrieved sources. Use it only if the crop, pest, formulation, and local label match your field situation.",
                "",
            ]
            for i, item in enumerate(structured[:3], start=1):
                lines.append(f"{i}. {self.recommendation_sentence(item, i)}")
            if len(structured) > 3:
                lines.append("")
                lines.append(f"I found {len(structured) - 3} more cited options in the evidence table below.")
            return "\n".join(lines)

        lines = [
            "I found related evidence, but it was not structured enough to confidently turn into a pesticide recommendation. Please verify the dose and waiting period from the cited source before use.",
            "",
        ]
        for i, item in enumerate(evidence, start=1):
            lines.append(f"{i}. {item['text']} Source: {item['source']}")
        return "\n".join(lines)
