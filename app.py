"""
Streamlit app for AI-powered legal document analysis.
"""

from pathlib import Path
import json
import tempfile
from unittest import result

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from extractor.extractor import LegalExtractor


APP_TITLE = "AI-Powered Legal Document Analysis"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource
def get_extractor() -> LegalExtractor:
    """Load the extractor once and reuse it across reruns."""
    return LegalExtractor()


def save_uploaded_pdf(uploaded_file, temp_dir: str) -> Path:
    """Save uploaded PDF to a temporary path."""
    safe_name = Path(uploaded_file.name).name
    pdf_path = Path(temp_dir) / safe_name
    pdf_path.write_bytes(uploaded_file.getbuffer())
    return pdf_path


def render_items(title: str, items: list, empty_message: str, style: str = "info") -> None:
    """Render a list of extracted items."""
    st.subheader(title)

    if not items:
        st.caption(empty_message)
        return

    for item in items:
        if style == "success":
            st.success(item)
        elif style == "warning":
            st.warning(item)
        else:
            st.info(item)


def render_summary(result: dict) -> None:
    """Render extraction summary metrics."""
    # st.header("📄 Extraction Summary")

    # col1, col2, col3, col4, col5 = st.columns(5)

    # col1.metric("Survey Numbers", len(result.get("survey_numbers", [])))
    # col2.metric("Survey Locations", len(result.get("survey_locations", [])))
    # col3.metric("Sections", len(result.get("sections", [])))
    # col4.metric("Acts", len(result.get("acts", [])))


def render_act_section_mapping(mapping: list) -> None:
    """Render act-to-section mapping."""
    st.header("📚 Act → Section Mapping")

    if not mapping:
        st.warning("No mapping found.")
        return

    for item in mapping:
        act = item.get("act", "Unknown Act")
        sections = item.get("sections", [])

        with st.expander(act, expanded=True):
            if not sections:
                st.caption("No associated sections.")
                continue

            columns = st.columns(4)

            for index, section in enumerate(sections):
                columns[index % 4].info(section)


def render_download(result: dict, file_name: str) -> None:
    """Render JSON download button and raw JSON preview."""
    st.header("⬇️ Download JSON")

    json_string = json.dumps(result, indent=4, ensure_ascii=False)

    st.download_button(
        label="Download JSON",
        data=json_string,
        file_name=f"{Path(file_name).stem}.json",
        mime="application/json",
    )

    with st.expander("📦 Raw JSON", expanded=False):
        st.json(result)

def render_survey_locations(locations: list[dict]) -> None:
    """Render Survey Number -> Location details."""
    st.header("📍 Survey Locations")

    if not locations:
        st.warning("No survey locations found.")
        return

    st.dataframe(
        locations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "survey_number": "Survey Number",
            "village": "Village",
            "hobli": "Hobli",
            "taluk": "Taluk",
            "district": "District",
        },
    )

def analyze_pdf(uploaded_file) -> dict:
    """Run the extraction pipeline on the uploaded PDF."""
    extractor = get_extractor()

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = save_uploaded_pdf(uploaded_file, temp_dir)
        return extractor.extract(pdf_path)


def main() -> None:
    st.title(f"⚖️ {APP_TITLE}")

    uploaded_file = st.file_uploader(
        label="Upload PDF",
        type=["pdf"],
    )

    if uploaded_file is None:
        st.info("Please upload a PDF file to begin analysis.")
        return

    st.success(f"Selected file: **{uploaded_file.name}**")

    try:
        with st.spinner("Analyzing document..."):
            result = analyze_pdf(uploaded_file)

    except Exception as error:
        st.error("Something went wrong while analyzing the document.")
        st.exception(error)
        return

    # st.divider()
    # render_summary(result)

    st.divider()
    st.subheader("📌 Case Number")
    st.info(result.get("case_number") or "Not available")

    st.divider()

    left_column, right_column = st.columns(2)

    with left_column:
        render_items(
            title="📍 Survey Numbers",
            items=result.get("survey_numbers", []),
            empty_message="No survey numbers found.",
            style="success",
        )


        # render_items(
        #     title="📜 Sections",
        #     items=result.get("sections", []),
        #     empty_message="No sections found.",
        #     style="info",
        # )

    with right_column:
        render_items(
            title="⚖️ Acts",
            items=result.get("acts", []),
            empty_message="No acts found.",
            style="success",
        )

        # st.divider()

        # st.subheader("⭐ Primary Act")
        # primary_act = result.get("primary_act")

        # if primary_act:
        #     st.success(primary_act)
        # else:
        #     st.warning("None")
        
    st.divider()
    render_survey_locations(result.get("survey_locations", []))

    st.divider()
    render_act_section_mapping(result.get("act_section_mapping", []))

    st.divider()
    st.header("📄 PDF Preview")

    pdf_viewer(
        input=uploaded_file.getvalue(),
        width=900,
        height=900,
    )

    st.divider()
    render_download(result, uploaded_file.name)


if __name__ == "__main__":
    main()