from __future__ import annotations

import json
import time

import streamlit as st

from extractor.extractor import LegalExtractor
from extractor.survey_location import SurveyLocationExtractor

from RAG.chunker import LegalTextChunker
from RAG.embedding import EmbeddingModel
from RAG.vector_store import QdrantVectorStore
from RAG.act_extractor import ActExtractor


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="CCMS - Legal Document Analysis",
    page_icon="⚖️",
    layout="wide",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 13px;
    }

    .result-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">⚖️ CCMS Legal Document Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "OCR Text → Extraction → Chunking → Embeddings → Qdrant → "
    "Survey Locations → Act Extraction"
    "</div>",
    unsafe_allow_html=True,
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## Document")

    uploaded_file = st.file_uploader(
        "Upload OCR Text File",
        type=["txt"],
    )

    st.markdown("---")

    st.markdown("### Pipeline")

    st.caption("1. Read OCR text")
    st.caption("2. Extract case information")
    st.caption("3. Chunk document")
    st.caption("4. Generate embeddings")
    st.caption("5. Store in Qdrant")
    st.caption("6. Extract survey locations")
    st.caption("7. Extract Acts")
    st.caption("8. Generate final result")


# ==========================================================
# NO FILE
# ==========================================================

if uploaded_file is None:

    st.info(
        "Upload an OCR `.txt` file from the sidebar to begin."
    )

    st.stop()


# ==========================================================
# READ FILE
# ==========================================================

try:

    text = uploaded_file.read().decode(
        "utf-8"
    )

except UnicodeDecodeError:

    try:

        uploaded_file.seek(0)

        text = uploaded_file.read().decode(
            "cp1252"
        )

    except UnicodeDecodeError:

        uploaded_file.seek(0)

        text = uploaded_file.read().decode(
            "latin-1"
        )


document_name = uploaded_file.name.rsplit(
    ".",
    1,
)[0]


# ==========================================================
# DOCUMENT INFORMATION
# ==========================================================

st.markdown(
    '<div class="section-title">Document Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-value">{document_name}</div>
            <div class="metric-label">Document</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-value">{len(text):,}</div>
            <div class="metric-label">Characters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-value">{len(text.split()):,}</div>
            <div class="metric-label">Words</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# RUN PIPELINE
# ==========================================================

run_pipeline = st.button(
    "Run Legal Document Analysis",
    type="primary",
    use_container_width=True,
)


if not run_pipeline:

    st.stop()


# ==========================================================
# PIPELINE TIMER
# ==========================================================

pipeline_start = time.perf_counter()


# ==========================================================
# PROGRESS
# ==========================================================

progress = st.progress(
    0
)

status = st.empty()


# ==========================================================
# 1. EXTRACT CASE INFORMATION
# ==========================================================

status.write(
    "Step 1/7 — Extracting case information..."
)

legal_extractor = LegalExtractor()

base_result = legal_extractor.extract(
    text=text,
    case_number=document_name,
)

progress.progress(
    10
)


# ==========================================================
# 2. CHUNK DOCUMENT
# ==========================================================

status.write(
    "Step 2/7 — Chunking document..."
)

temp_path = None

try:

    import tempfile

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt",
        mode="w",
        encoding="utf-8",
    ) as temp_file:

        temp_file.write(text)

        temp_path = temp_file.name

    chunker = LegalTextChunker(
        chunk_size=450,
        overlap=50,
    )

    chunks = chunker.chunk_file(
        temp_path
    )

finally:

    if temp_path:

        from pathlib import Path

        temp_file_path = Path(
            temp_path
        )

        if temp_file_path.exists():

            temp_file_path.unlink()


if not chunks:

    st.error(
        "No chunks were created."
    )

    st.stop()


progress.progress(
    25
)


# ==========================================================
# 3. GENERATE EMBEDDINGS
# ==========================================================

status.write(
    "Step 3/7 — Generating embeddings..."
)

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts
)

progress.progress(
    45
)


# ==========================================================
# 4. STORE IN QDRANT
# ==========================================================

status.write(
    "Step 4/7 — Storing embeddings in Qdrant..."
)

vector_store = QdrantVectorStore()

vector_store.insert(
    chunks,
    embeddings,
)

progress.progress(
    60
)


# ==========================================================
# 5. SURVEY LOCATION EXTRACTION
# ==========================================================

status.write(
    "Step 5/7 — Extracting survey locations..."
)

survey_numbers = base_result.get(
    "survey_numbers",
    [],
)

if survey_numbers:

    location_extractor = (
        SurveyLocationExtractor(
            text
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

progress.progress(
    70
)


# ==========================================================
# 6. ACT EXTRACTION
# ==========================================================

status.write(
    "Step 6/7 — Extracting Acts and Act-Section mappings..."
)

act_extractor = ActExtractor(
    chunks
)

act_result = act_extractor.extract(
    document=document_name,
    sections=base_result.get(
        "sections",
        [],
    ),
    top_k=5,
)


# ==========================================================
# ADD ACT RESULT
# ==========================================================

base_result["acts"] = act_result.get(
    "acts",
    [],
)

base_result[
    "act_section_mapping"
] = act_result.get(
    "act_section_mapping",
    [],
)

progress.progress(
    90
)


# ==========================================================
# TOTAL TIME
# ==========================================================

pipeline_end = time.perf_counter()

total_time = (
    pipeline_end - pipeline_start
)


progress.progress(
    100
)

status.success(
    f"Pipeline completed in {total_time:.2f} seconds"
)


# ==========================================================
# RESULTS SUMMARY
# ==========================================================

st.markdown(
    '<div class="section-title">Analysis Summary</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Chunks",
        len(chunks),
    )

with col2:

    st.metric(
        "Survey Numbers",
        len(
            survey_numbers
        ),
    )

with col3:

    st.metric(
        "Acts",
        len(
            base_result.get(
                "acts",
                [],
            )
        ),
    )

with col4:

    st.metric(
        "Sections",
        len(
            base_result.get(
                "sections",
                [],
            )
        ),
    )


# ==========================================================
# TABS
# ==========================================================

st.markdown(
    '<div class="section-title">Extracted Information</div>',
    unsafe_allow_html=True,
)

tab_overview, tab_surveys, tab_acts, tab_sections, tab_mapping, tab_json = st.tabs(
    [
        "Overview",
        "Survey Locations",
        "Acts",
        "Sections",
        "Act → Section",
        "JSON",
    ]
)


# ==========================================================
# OVERVIEW
# ==========================================================

with tab_overview:

    st.subheader(
        "Document"
    )

    st.write(
        document_name
    )

    st.subheader(
        "Processing Time"
    )

    st.write(
        f"{total_time:.2f} seconds"
    )

    st.subheader(
        "Chunk Information"
    )

    st.write(
        f"{len(chunks)} chunks created"
    )

    st.write(
        f"Embedding shape: {embeddings.shape}"
    )


# ==========================================================
# SURVEY LOCATIONS
# ==========================================================

with tab_surveys:

    survey_locations = base_result.get(
        "survey_locations",
        [],
    )

    if survey_locations:

        st.json(
            survey_locations
        )

    else:

        st.info(
            "No survey locations found."
        )


# ==========================================================
# ACTS
# ==========================================================

with tab_acts:

    acts = base_result.get(
        "acts",
        [],
    )

    if acts:

        for act in acts:

            if isinstance(
                act,
                dict,
            ):

                st.markdown(
                    f"""
                    <div class="result-box">
                        <strong>{act.get("name", "")}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.write(
                    act
                )

    else:

        st.info(
            "No Acts found."
        )


# ==========================================================
# SECTIONS
# ==========================================================

with tab_sections:

    sections = base_result.get(
        "sections",
        [],
    )

    if sections:

        st.json(
            sections
        )

    else:

        st.info(
            "No Sections found."
        )


# ==========================================================
# ACT → SECTION MAPPING
# ==========================================================

with tab_mapping:

    mappings = base_result.get(
        "act_section_mapping",
        [],
    )

    if mappings:

        for mapping in mappings:

            act_name = mapping.get(
                "act",
                "",
            )

            mapped_sections = mapping.get(
                "sections",
                [],
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <strong>{act_name}</strong>
                    <br><br>
                    Sections: {", ".join(mapped_sections)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.info(
            "No Act-Section mappings found."
        )


# ==========================================================
# COMPLETE JSON
# ==========================================================

with tab_json:

    st.json(
        base_result
    )

    st.download_button(
        label="Download JSON",
        data=json.dumps(
            base_result,
            indent=4,
            ensure_ascii=False,
        ),
        file_name=f"{document_name}_result.json",
        mime="application/json",
        use_container_width=True,
    )