from fastapi import FastAPI, UploadFile, File
from extractor.extractor import LegalExtractor
from pathlib import Path
import tempfile
import shutil


app = FastAPI(
    title="AI-Powered Legal Document Analysis API",
    description="Extract structured information from Karnataka High Court documents.",
    version="1.0.0"
)

extractor = LegalExtractor()

@app.get("/")
def home():
    return {
        "message": "Welcome to the Legal Document Extraction API!"
    }
    
@app.post("/extract")
async def extract_pdf(file: UploadFile = File(...)):

    with tempfile.TemporaryDirectory() as temp_dir:

        pdf_path = Path(temp_dir) / file.filename

        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = extractor.extract(pdf_path)

    return result