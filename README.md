# AI-Powered Legal Document Analysis System

An AI-powered legal document analysis system designed to extract structured information from Karnataka High Court documents. The system combines OCR, rule-based extraction, and Large Language Models (LLMs) to automatically identify and organize legal entities into a structured JSON format.

---

## Overview

Legal documents are often lengthy, unstructured, and difficult to analyze manually. This project automates the extraction of key legal information from High Court PDF documents, making them easier to process, search, and integrate into downstream legal workflows.

The system currently extracts:

- Case Number
- Survey Numbers
- Survey Locations
  - Village
  - Hobli
  - Taluk
  - District
- Acts
- Sections

The extracted information is returned as clean, validated JSON.

---

## Features

- PDF document processing
- OCR support for scanned documents
- Survey Number extraction using Regex
- Survey Location extraction using an LLM
- Automatic mapping of Village, Hobli, Taluk and District to survey numbers
- Act extraction using LLM
- Section extraction using Regex
- Automatic normalization of extracted values
- Validation against predefined legal data
- JSON output generation
- Interactive Streamlit interface

---

## System Architecture

```
                PDF Document
                     │
                     ▼
             OCR (Tesseract)
                     │
                     ▼
            Extracted Raw Text
                     │
                     ▼
            Regex Extraction
          ┌──────────────────┐
          │ Survey Numbers   │
          │ Sections         │
          └──────────────────┘
                     │
                     ▼
      Survey Location Extraction (LLM)
                     │
                     ▼
         Act Extraction (Chunked LLM)
                     │
                     ▼
        Normalization & Validation
                     │
                     ▼
              Structured JSON
```

---

## Project Structure

```
CCMS/
│
├── app.py
├── config.py
├── requirements.txt
│
├── extractor/
│   ├── extractor.py
│   ├── survey_extractor.py
│   ├── survey_location.py
│   ├── section_extractor.py
│   ├── regex_extractor.py
│   ├── normalizer.py
│   ├── validator.py
│   └── llm/
│
├── prompts/
│   ├── act_prompt.py
│   └── location_prompt.py
│
├── output_json/
├── output_text/
├── sample_documents/
└── assets/
```

---

## Technologies Used

- Python
- Streamlit
- Ollama
- Qwen Models
- PyMuPDF
- Tesseract OCR
- pdf2image
- Regular Expressions
- JSON

---

## Extraction Pipeline

1. Upload a PDF document.
2. OCR extracts text from scanned pages.
3. Survey Numbers and Sections are extracted using Regex.
4. Survey locations are extracted using an LLM.
5. Acts are extracted using chunk-wise LLM processing.
6. Results are normalized.
7. Validation is performed.
8. Final structured JSON is generated.

---

## Example Output

```json
{
  "case_number": "WP-202220-2023",

  "survey_locations": [
    {
      "survey_number": "82",
      "village": "Dattagalli",
      "hobli": "Kasaba",
      "taluk": "Mysuru",
      "district": "Mysuru"
    }
  ],

  "acts": [
    "Karnataka Land Revenue Act, 1964"
  ],

  "sections": [
    "Section 95"
  ]
}
```

---

## Running the Project

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Future Improvements

- Deploy using vLLM on NVIDIA A100
- Faster parallel LLM inference
- Improved OCR correction
- Better location extraction for complex legal documents
- Support additional Indian legal document formats

---

## Author

**Stephen Fernandes**

AI Engineer Intern at
CEG (Centre for e-Governance)


---

## License

This project is intended for educational and research purposes.
