from fastapi import FastAPI, File, UploadFile,WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil

import os
import subprocess
import pdfplumber
import multiprocessing 
import os
import fitz  
import docx 
import torch
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from pyngrok import ngrok
import random
import time
import datetime
import uvicorn
import shap
import asyncio
from groq import Groq
import socket
from PyPDF2 import PdfReader
from docx import Document
from fastapi.middleware.gzip import GZipMiddleware
nltk.download('stopwords')
groq_API_KEY  = "PUT YOUR GROQ API KEY"
client =Groq(api_key=groq_API_KEY)


from fastapi.staticfiles import StaticFiles

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


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
        print(f" Converted '{input_path}' to PDF: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting DOCX to PDF: {e}")


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
    
    print(f"Converted '{pdf_path}' to HTML: {output_html}")
    return html_content  



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "docx"]:
        return JSONResponse(content={"message": "Unsupported file format"}, status_code=400)

    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())


    time.sleep(100000000)
    doc_data={[]}

    return JSONResponse(content={
        "filename": file.filename,
        "fileUrl": file_location,
        "message": "File uploaded successfully",
        "extracted_text": doc_data["extracted_text"],
        "htmlUrl": doc_data["htmlUrl"],
        "original_text": doc_data["extracted_text"],
        "modified_text": doc_data["modified_text"],
        "explained_text": doc_data["explained_text"]
    })


if __name__ == "__main__":
    # ngrok_tunnel = ngrok.connect(8000)
    # print(f"FastAPI Public URL: {ngrok_tunnel.public_url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)

