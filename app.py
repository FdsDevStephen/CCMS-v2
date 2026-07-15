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

st.write("Upload a Karnataka High Court judgment PDF to extract legal information.")

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

    st.header("📄 Extraction Result")

    # ------------------------------------------------------
    # Case Number
    # ------------------------------------------------------

    st.subheader("Case Number")

    st.info(result.get("case_number", "Not Available"))

    # ------------------------------------------------------
    # Two Columns
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    # ======================================================
    # LEFT COLUMN
    # ======================================================

    with col1:

        st.subheader("📍 Survey Numbers")

        surveys = result.get("survey_numbers", [])

        if surveys:

            for survey in surveys:
                st.write(f"• {survey}")

        else:

            st.caption("No Survey Numbers Found.")

        st.divider()

    # ======================================================
    # RIGHT COLUMN
    # ======================================================

    with col2:

        st.subheader("⚖️ Acts")

        acts = result.get("acts", [])

        if acts:

            for act in acts:
                st.write(f"• {act}")

        else:

            st.caption("No Acts Found.")

        st.divider()

    # ======================================================
    # ACT → SECTION MAPPING
    # ======================================================

    st.divider()

    st.subheader("📚 Act → Section Mapping")

    mapping = result.get("act_section_mapping", [])

    if mapping:

        for item in mapping:

            with st.container(border=True):

                st.markdown(f"### {item['act']}")

                if item["sections"]:

                    for section in item["sections"]:

                        st.write(f"• {section}")

                else:

                    st.caption("No Associated Sections")

    else:

        st.caption("No Mapping Found.")
