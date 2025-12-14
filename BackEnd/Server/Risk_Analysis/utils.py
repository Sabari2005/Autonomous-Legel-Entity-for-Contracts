import os
import subprocess
import re
import pdfplumber
from PyPDF2 import PdfReader
from docx import Document
from BackEnd.Server.Risk_Analysis.config import UPLOAD_DIR

def extract_text_from_file(file_path: str, file_ext: str) -> str:
    extracted_text = ""
    if file_ext == "pdf":
        with open(file_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_ext == "docx":
        doc = Document(file_path)
        extracted_text = "\n".join([para.text for para in doc.paragraphs])
    return extracted_text.strip()

def docx_to_pdf(input_path: str, output_pdf: str):
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", input_path, "--outdir", os.path.dirname(output_pdf)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error converting DOCX to PDF: {e}")

def pdf_to_html(pdf_path: str, output_html: str) -> str:
    html_content = "<html><body>"
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                html_content += "<p>" + text.replace("\n", "<br>") + "</p>"
    html_content += "</body></html>"
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_content

def clean_extracted_text(text):
    text = text.replace('\r', '')
    text = re.sub(r'\n+', '\n', text)
    return re.sub(r'\s+', ' ', text).strip()