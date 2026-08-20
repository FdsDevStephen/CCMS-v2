import streamlit as st
import requests

st.set_page_config(
    page_title="Legal AI Extractor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = "http://101.53.140.252/extract"

st.markdown("""
<style>
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

h1, h2, h3 {
    color: #f8fafc !important;
}

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

.file-info {
    background: #161c24;
    border: 1px solid #293241;
    border-radius: 8px;
    padding: 0.65rem 0.8rem;
    margin-top: 0.6rem;
    color: #cbd5e1;
    font-size: 0.82rem;
}

.metric {
    background: #11161d;
    border: 1px solid #232a34;
    border-radius: 10px;
    padding: 0.9rem 1rem;
}

.metric-label {
    color: #64748b;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 0.35rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.result-title {
    color: #f8fafc;
    font-size: 1.15rem;
    font-weight: 600;
    margin: 1.5rem 0 0.8rem;
}

.item {
    background: #161c24;
    border: 1px solid #252d38;
    border-radius: 8px;
    padding: 0.65rem 0.8rem;
    margin-bottom: 0.45rem;
    color: #dbe2ea;
    font-size: 0.84rem;
}

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

.empty {
    color: #64748b;
    background: #11161d;
    border: 1px dashed #293241;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    font-size: 0.82rem;
}

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

.footer {
    color: #475569;
    text-align: center;
    font-size: 0.72rem;
    margin-top: 2rem;
}

[data-testid="stFileUploader"] {
    background: #0f141a;
    border: 1px dashed #303946;
    border-radius: 9px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #64748b;
}

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

div[data-testid="stDataFrame"] {
    border: 1px solid #232a34;
    border-radius: 8px;
    overflow: hidden;
}

.stExpander {
    background: #11161d !important;
    border: 1px solid #232a34 !important;
    border-radius: 8px !important;
}

.stAlert {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="app-title">⚖️ Legal AI Extractor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">High Court document information extraction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="panel"><div class="panel-title">Upload Document</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file:

    size_kb = uploaded_file.size / 1024

    st.markdown(
        f'<div class="file-info">📄 {uploaded_file.name} &nbsp;·&nbsp; {size_kb:.2f} KB</div>',
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:

    extract = st.button(
        "⚡ Extract Information",
        type="primary",
        use_container_width=True
    )

else:

    extract = False

    st.markdown(
        '<div class="empty">Upload a PDF to begin extraction.</div>',
        unsafe_allow_html=True
    )

if uploaded_file and extract:

    processing = st.empty()

    processing.markdown(
        '<div class="processing">⏳ Processing document with OCR and AI extraction...</div>',
        unsafe_allow_html=True
    )

    try:

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        response = requests.post(
            API_URL,
            files=files,
            timeout=180
        )

        processing.empty()

        if response.status_code != 200:

            st.error(
                f"Extraction API returned HTTP {response.status_code}"
            )

            try:
                st.json(response.json())
            except ValueError:
                st.code(response.text)

            st.stop()

        try:

            data = response.json()

        except ValueError:

            st.error(
                "The extraction server returned invalid JSON."
            )

            st.code(response.text)
            st.stop()

    except requests.exceptions.ConnectionError as e:

        processing.empty()

        st.error(
            "Could not connect to the extraction server."
        )

        st.code(str(e))
        st.stop()

    except requests.exceptions.Timeout:

        processing.empty()

        st.error(
            "The extraction request timed out."
        )

        st.stop()

    except requests.exceptions.RequestException as e:

        processing.empty()

        st.error(
            "Request failed."
        )

        st.code(str(e))
        st.stop()

    except Exception as e:

        processing.empty()

        st.exception(e)
        st.stop()

    case_number = data.get("case_number")
    acts = data.get("acts", [])
    sections = data.get("sections", [])
    survey_locations = data.get("survey_locations", [])
    mappings = data.get("act_section_mapping", [])

    st.markdown(
        '<div class="success">✓ Extraction completed successfully</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-title">Extraction Summary</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="metric"><div class="metric-label">CASE NUMBER</div><div class="metric-value">{case_number or "N/A"}</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="metric"><div class="metric-label">ACTS</div><div class="metric-value">{len(acts)}</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="metric"><div class="metric-label">SECTIONS</div><div class="metric-value">{len(sections)}</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="metric"><div class="metric-label">SURVEY LOCATIONS</div><div class="metric-value">{len(survey_locations)}</div></div>',
            unsafe_allow_html=True
        )

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="result-title">📜 Acts</div>',
            unsafe_allow_html=True
        )

        if acts:

            for act in acts:

                st.markdown(
                    f'<div class="item">📜 {act}</div>',
                    unsafe_allow_html=True
                )

        else:

            st.markdown(
                '<div class="empty">No Acts identified.</div>',
                unsafe_allow_html=True
            )

    with right:

        st.markdown(
            '<div class="result-title">🔢 Sections</div>',
            unsafe_allow_html=True
        )

        if sections:

            for section in sections:

                st.markdown(
                    f'<span class="chip">§ {section}</span>',
                    unsafe_allow_html=True
                )

        else:

            st.markdown(
                '<div class="empty">No Sections identified.</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="result-title">🗺️ Survey Locations</div>',
        unsafe_allow_html=True
    )

    if survey_locations:

        st.dataframe(
            survey_locations,
            use_container_width=True,
            hide_index=True,
            height=min(
                500,
                100 + len(survey_locations) * 45
            )
        )

    else:

        st.markdown(
            '<div class="empty">No survey locations found.</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="result-title">🔗 Act → Section Mapping</div>',
        unsafe_allow_html=True
    )

    if mappings:

        for item in mappings:

            act = item.get(
                "act",
                "Unknown Act"
            )

            mapped_sections = item.get(
                "sections",
                []
            )

            st.markdown(
                f'<div class="mapping"><div class="mapping-act">📜 {act}</div>',
                unsafe_allow_html=True
            )

            if mapped_sections:

                for section in mapped_sections:

                    st.markdown(
                        f'<span class="chip">§ {section}</span>',
                        unsafe_allow_html=True
                    )

            else:

                st.caption("No sections mapped.")

            st.markdown("</div>", unsafe_allow_html=True)

    else:

        st.markdown(
            '<div class="empty">No direct mappings found.</div>',
            unsafe_allow_html=True
        )

    with st.expander("🔍 Raw JSON"):

        st.json(data)

st.markdown(
    '<div class="footer">Legal AI · High Court Document Information Extraction System</div>',
    unsafe_allow_html=True
)