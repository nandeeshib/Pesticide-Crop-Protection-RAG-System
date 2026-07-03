import pandas as pd
import streamlit as st

from src.pesticide_rag.config import CHUNKS_PATH
from src.pesticide_rag.rag import PesticideRAG
from src.pesticide_rag.translator import KannadaTranslator


st.set_page_config(page_title="Pesticide RAG", layout="wide")


@st.cache_resource
def load_rag():
    return PesticideRAG(use_reranker=True)


@st.cache_resource
def load_translator():
    return KannadaTranslator()


st.title("Pesticide & Crop Protection RAG")
st.caption("Evidence-first pesticide guidance with source citations and WHO Class I risk flags.")
st.info("Answer mode: Plain English or Kannada recommendations generated from retrieved structured chunks.")

if not CHUNKS_PATH.exists():
    st.error("Chunks not found. Run ingestion first.")
    st.code("python -m src.pesticide_rag.ingest", language="powershell")
    st.stop()

rag = load_rag()

with st.sidebar:
    st.header("Query Controls")
    crop = st.text_input("Crop filter", placeholder="rice, cotton, tomato")
    pest = st.text_input("Pest filter", placeholder="brown planthopper, bollworm")
    output_language = st.radio("Output language", ["English", "Kannada"], horizontal=True)
    st.divider()
    st.write("Use exact crop and pest names where possible for safer retrieval.")

query = st.text_area(
    "Ask a crop-pest pesticide question / ಬೆಳೆ-ಕೀಟ ಪ್ರಶ್ನೆ ಕೇಳಿ",
    placeholder="Example: What pesticide and dose is recommended for brown planthopper in rice? / ಭತ್ತದಲ್ಲಿ ಕಂದು ಜಿಗಿ ಹುಳಿಗೆ ಯಾವ ಕೀಟನಾಶಕ ಮತ್ತು ಪ್ರಮಾಣ?",
    height=100,
)

if st.button("Get cited recommendation", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Enter a question first.")
        st.stop()

    with st.spinner("Retrieving and checking evidence..."):
        translator = load_translator() if output_language == "Kannada" else None
        retrieval_query = query
        retrieval_crop = crop
        retrieval_pest = pest
        display_question = query
        if translator:
            retrieval_query = translator.to_english(query)
            retrieval_crop = translator.translate_filter(crop)
            retrieval_pest = translator.translate_filter(pest)
            display_question = translator.question_to_kannada(query)

        result = rag.answer(retrieval_query, crop=retrieval_crop, pest=retrieval_pest)
        answer = result["answer"]
        if output_language == "Kannada":
            answer = translator.answer_to_kannada(answer, result["evidence"])

    risk = result["risk"]
    if risk["is_highly_hazardous"]:
        names = ", ".join(f"{m['active_ingredient']} ({m['who_class']})" for m in risk["matches"])
        if output_language == "Kannada":
            st.error(f"WHO Class I ಅತ್ಯಂತ ಅಪಾಯಕಾರಿ ಕೀಟನಾಶಕ ಎಚ್ಚರಿಕೆ: {names}")
        else:
            st.error(f"WHO Class I highly hazardous pesticide flag: {names}")
    else:
        if output_language == "Kannada":
            st.success("ಹಿಂತೆಗೆದ ಉತ್ತರದಲ್ಲಿ WHO Class I ಹೊಂದಾಣಿಕೆ ಕಂಡುಬಂದಿಲ್ಲ.")
        else:
            st.success("No WHO Class I match detected in the retrieved answer text.")

    if output_language == "Kannada":
        st.subheader("ಕನ್ನಡ ಪ್ರಶ್ನೆ")
        st.write(display_question)
        st.subheader("ಕನ್ನಡ ಉತ್ತರ")
    else:
        st.subheader("Plain English Answer")
    st.write(answer)

    st.subheader("Cited Evidence")
    evidence_df = pd.DataFrame(result["evidence"])
    st.dataframe(evidence_df, use_container_width=True, hide_index=True)

    st.subheader("Retrieved Contexts")
    context_rows = [
        {
            "score": c.get("score"),
            "source_type": c.get("source_type"),
            "crop": c.get("crop"),
            "pest": c.get("pest"),
            "source": rag.format_source(c),
            "text": c.get("text", "")[:700],
        }
        for c in result["contexts"]
    ]
    st.dataframe(pd.DataFrame(context_rows), use_container_width=True, hide_index=True)
