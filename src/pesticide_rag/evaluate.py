import pandas as pd
import site
import sys

from .config import DATA_DIR
from .rag import PesticideRAG


user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.append(user_site)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    eval_path = DATA_DIR / "eval" / "eval_queries.csv"
    out_path = DATA_DIR / "eval" / "eval_results.csv"
    summary_path = DATA_DIR / "eval" / "eval_summary.md"
    df = pd.read_csv(eval_path)
    rag = PesticideRAG(use_reranker=False)

    predictions = []
    risk_flags = []
    risk_matches = []
    top_sources = []
    for _, row in df.iterrows():
        result = rag.answer(str(row["query"]))
        predictions.append(result["answer"])
        risk_flags.append(result["risk"]["is_highly_hazardous"])
        risk_matches.append(
            "; ".join(f"{m['active_ingredient']} ({m['who_class']})" for m in result["risk"]["matches"])
        )
        if result["evidence"]:
            top_sources.append(result["evidence"][0]["source"])
        else:
            top_sources.append("")

    df["predicted_answer"] = predictions
    df["toxicity_flag"] = risk_flags
    df["toxicity_matches"] = risk_matches
    df["top_source"] = top_sources

    refs = df["reference_answer"].fillna("").astype(str).tolist()
    bertscore_status = "not run"
    if all(ref.strip() for ref in refs):
        try:
            from bert_score import score

            precision, recall, f1 = score(
                predictions,
                refs,
                model_type="distilbert-base-uncased",
                batch_size=4,
                verbose=True,
            )
            df["bertscore_precision"] = precision.tolist()
            df["bertscore_recall"] = recall.tolist()
            df["bertscore_f1"] = f1.tolist()
            print(f"Mean BERTScore F1: {float(f1.mean()):.4f}")
            bertscore_status = "run"
        except Exception as exc:
            print(f"BERTScore skipped: {exc}")
            bertscore_status = f"skipped: {exc}"
    else:
        print("Reference answers are incomplete, so BERTScore was skipped.")
        bertscore_status = "skipped: incomplete reference answers"

    if "bertscore_f1" not in df.columns:
        df["fallback_token_f1"] = [
            token_f1(pred, ref) for pred, ref in zip(predictions, refs, strict=False)
        ]

    df.to_csv(out_path, index=False)
    metric_col = "bertscore_f1" if "bertscore_f1" in df.columns else "fallback_token_f1"
    mean_score = float(df[metric_col].mean())
    toxicity_count = int(df["toxicity_flag"].sum())
    summary_path.write_text(
        "\n".join(
            [
                "# RAG Evaluation Summary",
                "",
                f"- Query pairs evaluated: {len(df)}",
                f"- BERTScore status: {bertscore_status}",
                f"- Reported metric: {metric_col}",
                f"- Mean score: {mean_score:.4f}",
                f"- Toxicity-flagged answers: {toxicity_count}",
                f"- Results CSV: {out_path}",
                "",
                "Reference answers are curated from the ingested PPQS/CIB&RC and extension-source recommendations used in this MVP.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
