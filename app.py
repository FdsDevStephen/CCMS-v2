"""
Streamlit Application for AI-Powered Legal Document Analysis.
"""

from pathlib import Path
import json
import tempfile

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from extractor.extractor import LegalExtractor


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI-Powered Legal Document Analysis",
    layout="wide",
)

# ==========================================================
# TITLE
# ==========================================================

st.title("AI-Powered Legal Document Analysis System")

st.write(
    "Upload a Karnataka High Court judgment PDF to extract legal information."
)

# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

# ==========================================================
# RUN PIPELINE
# ==========================================================

if uploaded_file is not None:

    st.success(f"Selected File: **{uploaded_file.name}**")

    with st.spinner("Analyzing document..."):

        # --------------------------------------------------
        # Save uploaded PDF temporarily
        # --------------------------------------------------

        with tempfile.TemporaryDirectory() as temp_dir:

            pdf_path = Path(temp_dir) / uploaded_file.name

            pdf_path.write_bytes(uploaded_file.getbuffer())

            extractor = LegalExtractor()

            result = extractor.extract(pdf_path)

    # ======================================================
    # EXTRACTION RESULT
    # ======================================================

    st.divider()

    st.header("Extraction Result")

    st.subheader("Case Number")
    st.write(result.get("case_number", ""))

    st.subheader("Survey Numbers")

    survey_numbers = result.get("survey_numbers", [])

    if survey_numbers:

        for survey in survey_numbers:
            st.write(f"• {survey}")

    else:

        st.write("No Survey Numbers Found.")

    st.subheader("Sections")

    sections = result.get("sections", [])

    if sections:

        for section in sections:
            st.write(f"• {section}")

    else:

        st.write("No Sections Found.")

    st.subheader("Acts")

    acts = result.get("acts", [])

    if acts:

        for act in acts:
            st.write(f"• {act}")

    else:

        st.write("No Acts Found.")

    st.subheader("Primary Act")

    st.write(result.get("primary_act") or "None")

    # ======================================================
    # PDF PREVIEW
    # ======================================================

    st.divider()

    st.header("PDF Preview")

    pdf_viewer(
        input=uploaded_file.getvalue(),
        width=900,
        height=900,
    )

    # ======================================================
    # DOWNLOAD JSON
    # ======================================================

    st.divider()

    st.header("Download JSON")

    json_string = json.dumps(
        result,
        indent=4,
        ensure_ascii=False,
    )

    st.download_button(
        label="Download JSON",
        data=json_string,
        file_name=f"{Path(uploaded_file.name).stem}.json",
        mime="application/json",
    )

    # ======================================================
    # RAW JSON
    # ======================================================

    st.divider()

    st.header("Raw JSON")

    st.json(result)