from __future__ import annotations

import os
import warnings
import logging
import json
import time
from pathlib import Path

import streamlit as st


# ==========================================================
# QUIET THIRD-PARTY LIBRARIES
# ==========================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


# ==========================================================
# IMPORTS
# ==========================================================

from extractor.prayer_extractor import PrayerExtractor
from ocr_v2 import OCRProcessor
from extractor.extractor import LegalExtractor
from extractor.survey_location import SurveyLocationExtractor

from RAG.chunker import LegalTextChunker
from RAG.embedding import EmbeddingModel
from RAG.retriever import LegalRetriever
from RAG.hybrid_retreiver import HybridRetriever
from RAG.vector_store import QdrantVectorStore
from RAG.act_extractor import ActExtractor


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Legal AI Extractor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    """
    <style>

    /* ======================================================
       APP
       ====================================================== */

    .stApp {
        background: #0b0f14;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1250px;
        padding: 2rem 2.5rem 3rem;
    }

    header[data-testid="stHeader"] {
        background: #0b0f14;
    }

    h1,
    h2,
    h3 {
        color: #f8fafc !important;
    }

    /* ======================================================
       HEADER
       ====================================================== */

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 1.8rem;
    }

    /* ======================================================
       PANELS
       ====================================================== */

    .panel {
        background: #11161d;
        border: 1px solid #232a34;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    /* ======================================================
       FILE INFO
       ====================================================== */

    .file-info {
        background: #161c24;
        border: 1px solid #293241;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        margin-top: 0.6rem;
        color: #cbd5e1;
        font-size: 0.82rem;
    }

    /* ======================================================
       RESULT TITLE
       ====================================================== */

    .result-title {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem;
    }

    /* ======================================================
       ITEMS
       ====================================================== */

    .item {
        background: #161c24;
        border: 1px solid #252d38;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        margin-bottom: 0.45rem;
        color: #dbe2ea;
        font-size: 0.84rem;
    }

    /* ======================================================
       CHIPS
       ====================================================== */

    .chip {
        display: inline-block;
        background: #161c24;
        border: 1px solid #293241;
        border-radius: 6px;
        padding: 0.35rem 0.55rem;
        margin: 0.15rem;
        color: #cbd5e1;
        font-size: 0.78rem;
    }

    /* ======================================================
       EMPTY
       ====================================================== */

    .empty {
        color: #64748b;
        background: #11161d;
        border: 1px dashed #293241;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        font-size: 0.82rem;
    }

    /* ======================================================
       PRAYER
       ====================================================== */

    .prayer-panel {
        background: #11161d;
        border: 1px solid #232a34;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
    }

    .prayer-text {
        color: #dbe2ea;
        font-size: 0.88rem;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    /* ======================================================
       MAPPING
       ====================================================== */

    .mapping {
        background: #11161d;
        border: 1px solid #232a34;
        border-radius: 9px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }

    .mapping-act {
        color: #f8fafc;
        font-size: 0.84rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    /* ======================================================
       PROCESSING
       ====================================================== */

    .processing {
        background: #11161d;
        border: 1px solid #293241;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        color: #cbd5e1;
        font-size: 0.85rem;
        margin: 1rem 0;
    }

    .success {
        background: #0f241b;
        border: 1px solid #1f5138;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        color: #86efac;
        font-size: 0.82rem;
        margin: 1rem 0;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        color: #475569;
        text-align: center;
        font-size: 0.72rem;
        margin-top: 2rem;
    }

    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: #0f141a;
        border: 1px dashed #303946;
        border-radius: 9px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #64748b;
    }

    /* ======================================================
       BUTTON
       ====================================================== */

    div.stButton > button {
        height: 2.7rem;
        border-radius: 8px;
        font-weight: 600;
    }

    button[kind="primary"] {
        background: #2563eb !important;
        border-color: #2563eb !important;
    }

    button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }

    /* ======================================================
       DATAFRAME
       ====================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #232a34;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ======================================================
       EXPANDER
       ====================================================== */

    .stExpander {
        background: #11161d !important;
        border: 1px solid #232a34 !important;
        border-radius: 8px !important;
    }

    /* ======================================================
       ALERTS
       ====================================================== */

    .stAlert {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FOLDER = BASE_DIR / "section_output"

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==========================================================
# CACHED RESOURCES
# ==========================================================

@st.cache_resource(
    show_spinner="Loading BGE-M3 embedding model..."
)
def get_embedding_model():

    return EmbeddingModel()


@st.cache_resource
def get_ocr_processor():

    return OCRProcessor(
        output_folder=OUTPUT_FOLDER,
        tesseract_path=TESSERACT_PATH,
        search_pages=30,
        fast_dpi=150,
        full_dpi=220,
        max_workers=8,
        page_start=2,
        page_end=13,
        prefer_text_layer=True,
        denoise=True,
    )


@st.cache_resource
def get_legal_extractor():

    return LegalExtractor()


# ==========================================================
# PIPELINE
# ==========================================================

def run_pipeline(
    pdf_bytes: bytes,
    filename: str,
):

    pipeline_start = time.perf_counter()

    timings = {}

    # ======================================================
    # 1. OCR
    # ======================================================

    stage_start = time.perf_counter()

    ocr_processor = get_ocr_processor()

    ocr_text = ocr_processor.process_bytes(
        pdf_bytes=pdf_bytes,
        filename=filename,
    )

    timings["OCR"] = (
        time.perf_counter() - stage_start
    )

    txt_path = (
        OUTPUT_FOLDER
        / f"{Path(filename).stem}.txt"
    )

    if not txt_path.exists():

        raise RuntimeError(
            "OCR completed but the output text "
            "file could not be found."
        )

    # ======================================================
    # 2. LEGAL EXTRACTION
    # ======================================================

    stage_start = time.perf_counter()

    document_name = Path(filename).stem

    legal_extractor = get_legal_extractor()

    base_result = legal_extractor.extract(
        text=ocr_text,
        case_number=document_name,
    )

    timings["Legal Extraction"] = (
        time.perf_counter() - stage_start
    )

    # ======================================================
    # 3. PRAYER
    # ======================================================

    stage_start = time.perf_counter()

    prayer_extractor = PrayerExtractor()

    prayer = prayer_extractor.extract(
        ocr_text
    )

    base_result["prayer"] = prayer

    timings["Prayer Extraction"] = (
        time.perf_counter() - stage_start
    )

    # ======================================================
    # 4. BGE-M3
    # ======================================================

    stage_start = time.perf_counter()

    embedding_model = get_embedding_model()

    timings["Embedding Model"] = (
        time.perf_counter() - stage_start
    )

    # ======================================================
    # 5. CHUNKING
    # ======================================================

    stage_start = time.perf_counter()

    chunker = LegalTextChunker(
        chunk_size=450,
        overlap=50,
        tokenizer=embedding_model.tokenizer,
    )

    chunks = chunker.chunk_file(
        txt_path
    )

    if not chunks:

        raise RuntimeError(
            "No chunks were created. "
            "Check LegalTextChunker section parsing."
        )

    timings["Chunking"] = (
        time.perf_counter() - stage_start
    )

    # ======================================================
    # 6. EMBEDDINGS
    # ======================================================

    stage_start = time.perf_counter()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        batch_size=12,
    )

    timings["Embeddings"] = (
        time.perf_counter() - stage_start
    )

    # ======================================================
    # 7. QDRANT
    # ======================================================

    stage_start = time.perf_counter()

    vector_store = QdrantVectorStore()

    vector_store.insert(
        chunks,
        embeddings,
    )

    timings["Qdrant"] = (
        time.perf_counter() - stage_start
    )

    # ======================================================
    # 8. SURVEY LOCATIONS
    # ======================================================

    stage_start = time.perf_counter()

    survey_numbers = base_result.get(
        "survey_numbers",
        [],
    )

    if survey_numbers:

        location_extractor = (
            SurveyLocationExtractor(
                ocr_text
            )
        )

        survey_locations = (
            location_extractor.extract(
                survey_numbers
            )
        )

    else:

        survey_locations = []

    base_result[
        "survey_locations"
    ] = survey_locations

    timings["Survey Location"] = (
        time.perf_counter() - stage_start
    )

    # ======================================================
    # 9. HYBRID RETRIEVER
    # ======================================================

    stage_start = time.perf_counter()

    retriever = HybridRetriever(
        vector_retriever=LegalRetriever(
            embedding_model=embedding_model,
        )
    )

    retriever.build_bm25(
        chunks
    )

    timings["BM25 + Retriever"] = (
        time.perf_counter() - stage_start
    )

    # ======================================================
    # 10. ACT EXTRACTION
    # ======================================================

    stage_start = time.perf_counter()

    act_extractor = ActExtractor(
        chunks,
        retriever=retriever,
    )

    act_result = act_extractor.extract(
        document=document_name,
        sections=base_result.get(
            "sections",
            [],
        ),
        top_k=5,
    )

    timings["Act Extraction"] = (
        time.perf_counter() - stage_start
    )

    # ======================================================
    # 11. FINAL RESULT
    # ======================================================

    base_result["acts"] = (
        act_result.get(
            "acts",
            [],
        )
    )

    base_result["sections"] = (
        act_result.get(
            "sections",
            base_result.get(
                "sections",
                [],
            ),
        )
    )

    base_result[
        "act_section_mapping"
    ] = act_result.get(
        "act_section_mapping",
        [],
    )

    timings["Total"] = (
        time.perf_counter()
        - pipeline_start
    )

    return (
        base_result,
        timings,
        ocr_text,
        chunks,
    )


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="app-title">⚖️ Legal AI Extractor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">Karnataka High Court document information extraction</div>',
    unsafe_allow_html=True,
)


# ==========================================================
# UPLOAD PANEL
# ==========================================================

st.markdown(
    '<div class="panel"><div class="panel-title">Upload Document</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose PDF",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file:

    size_kb = (
        uploaded_file.size / 1024
    )

    st.markdown(
        f"""
        <div class="file-info">
            📄 {uploaded_file.name}
            &nbsp;·&nbsp;
            {size_kb:.2f} KB
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ==========================================================
# ANALYZE BUTTON
# ==========================================================

if uploaded_file:

    extract = st.button(
        "⚡ Extract Information",
        type="primary",
        width="stretch",
    )

else:

    extract = False

    st.markdown(
        '<div class="empty">Upload a PDF to begin extraction.</div>',
        unsafe_allow_html=True,
    )


# ==========================================================
# SESSION STATE
# ==========================================================

if "result" not in st.session_state:

    st.session_state.result = None

if "timings" not in st.session_state:

    st.session_state.timings = None

if "ocr_text" not in st.session_state:

    st.session_state.ocr_text = None

if "chunks" not in st.session_state:

    st.session_state.chunks = None


# ==========================================================
# EXTRACTION
# ==========================================================

if uploaded_file and extract:

    processing = st.empty()

    processing.markdown(
        """
        <div class="processing">
            ⏳ Processing document with OCR and AI extraction...
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress = st.progress(
        0,
        text="Starting extraction..."
    )

    try:

        progress.progress(
            10,
            text="Running OCR..."
        )

        (
            result,
            timings,
            ocr_text,
            chunks,
        ) = run_pipeline(
            uploaded_file.getvalue(),
            uploaded_file.name,
        )

        progress.progress(
            100,
            text="Extraction complete."
        )

        processing.empty()

        st.session_state.result = result

        st.session_state.timings = timings

        st.session_state.ocr_text = ocr_text

        st.session_state.chunks = chunks

    except Exception as exc:

        processing.empty()

        progress.empty()

        st.error(
            "Document extraction failed."
        )

        st.exception(exc)


# ==========================================================
# RESULT
# ==========================================================

result = st.session_state.result

if result is None:

    st.stop()


# ==========================================================
# SUCCESS
# ==========================================================

st.markdown(
    """
    <div class="success">
        ✓ Extraction completed successfully
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# CASE INFORMATION
# ==========================================================

st.markdown(
    '<div class="result-title">📋 Case Information</div>',
    unsafe_allow_html=True,
)

case_number = result.get(
    "case_number"
)

case_col, _ = st.columns(
    [1, 2]
)

with case_col:

    st.markdown(
        f"""
        <div class="item">
            <strong>Case Number</strong><br>
            {case_number or "N/A"}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# PRAYER
# ==========================================================

st.markdown(
    '<div class="result-title">🙏 Prayer</div>',
    unsafe_allow_html=True,
)

prayer = result.get(
    "prayer",
    "",
)

if prayer:

    st.markdown(
        f"""
        <div class="prayer-panel">
            <div class="prayer-text">
                {prayer}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        '<div class="empty">No Prayer identified.</div>',
        unsafe_allow_html=True,
    )


# ==========================================================
# ACTS + SECTIONS
# ==========================================================

left, right = st.columns(
    2,
    gap="large",
)


# ==========================================================
# ACTS
# ==========================================================

with left:

    st.markdown(
        '<div class="result-title">📜 Acts</div>',
        unsafe_allow_html=True,
    )

    acts = result.get(
        "acts",
        [],
    )

    if acts:

        for act in acts:

            if isinstance(
                act,
                dict,
            ):

                act_name = act.get(
                    "name",
                    "",
                )

            else:

                act_name = str(act)

            if act_name:

                st.markdown(
                    f"""
                    <div class="item">
                        📜 {act_name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:

        st.markdown(
            '<div class="empty">No Acts identified.</div>',
            unsafe_allow_html=True,
        )


# ==========================================================
# SECTIONS
# ==========================================================

with right:

    st.markdown(
        '<div class="result-title">🔢 Sections</div>',
        unsafe_allow_html=True,
    )

    sections = result.get(
        "sections",
        [],
    )

    if sections:

        for section in sections:

            st.markdown(
                f"""
                <span class="chip">
                    § {section}
                </span>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.markdown(
            '<div class="empty">No Sections identified.</div>',
            unsafe_allow_html=True,
        )


# ==========================================================
# SURVEY NUMBERS
# ==========================================================

st.markdown(
    '<div class="result-title">📍 Survey Numbers</div>',
    unsafe_allow_html=True,
)

survey_numbers = result.get(
    "survey_numbers",
    [],
)

if survey_numbers:

    for survey_number in survey_numbers:

        st.markdown(
            f"""
            <span class="chip">
                {survey_number}
            </span>
            """,
            unsafe_allow_html=True,
        )

else:

    st.markdown(
        '<div class="empty">No Survey Numbers identified.</div>',
        unsafe_allow_html=True,
    )


# ==========================================================
# SURVEY LOCATIONS
# ==========================================================

st.markdown(
    '<div class="result-title">🗺️ Survey Locations</div>',
    unsafe_allow_html=True,
)

survey_locations = result.get(
    "survey_locations",
    [],
)

if survey_locations:

    st.dataframe(
        survey_locations,
        width="stretch",
        hide_index=True,
        height=min(
            500,
            100
            + len(survey_locations) * 45,
        ),
    )

else:

    st.markdown(
        '<div class="empty">No survey locations found.</div>',
        unsafe_allow_html=True,
    )


# ==========================================================
# ACT → SECTION MAPPING
# ==========================================================

st.markdown(
    '<div class="result-title">🔗 Act → Section Mapping</div>',
    unsafe_allow_html=True,
)

mappings = result.get(
    "act_section_mapping",
    [],
)

if mappings:

    for item in mappings:

        if not isinstance(
            item,
            dict,
        ):
            continue

        act = item.get(
            "act",
            "Unknown Act",
        )

        mapped_sections = item.get(
            "sections",
            [],
        )

        st.markdown(
            f"""
            <div class="mapping">

                <div class="mapping-act">
                    📜 {act}
                </div>
            """,
            unsafe_allow_html=True,
        )

        if mapped_sections:

            for section in mapped_sections:

                st.markdown(
                    f"""
                    <span class="chip">
                        § {section}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.caption(
                "No sections mapped."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

else:

    st.markdown(
        '<div class="empty">No direct mappings found.</div>',
        unsafe_allow_html=True,
    )


# ==========================================================
# TECHNICAL OUTPUT
# ==========================================================

with st.expander(
    "🔍 Technical Output"
):

    st.subheader(
        "Final JSON"
    )

    st.json(
        result
    )

    st.subheader(
        "OCR Text"
    )

    ocr_text = (
        st.session_state.ocr_text
    )

    if ocr_text:

        st.text_area(
            "OCR",
            value=ocr_text,
            height=400,
            label_visibility="collapsed",
        )

    st.subheader(
        "Document Chunks"
    )

    chunks = (
        st.session_state.chunks
    )

    if chunks:

        st.caption(
            f"{len(chunks)} chunks generated."
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            with st.expander(
                f"Chunk {index}"
            ):

                st.write(
                    chunk.get(
                        "text",
                        "",
                    )
                )


# ==========================================================
# DOWNLOAD
# ==========================================================

st.download_button(
    "⬇️ Download JSON",
    data=json.dumps(
        result,
        indent=4,
        ensure_ascii=False,
    ).encode("utf-8"),
    file_name=(
        f"{Path(uploaded_file.name).stem}.json"
    ),
    mime="application/json",
    width="stretch",
)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        Legal AI · High Court Document Information Extraction System
    </div>
    """,
    unsafe_allow_html=True,
)