import transformers
from langchain_ollama import OllamaLLM
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from pyngrok import ngrok
import time
import os
from fastapi.staticfiles import StaticFiles


from PyPDF2 import PdfReader
from docx import Document
import uvicorn
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
llama_model = OllamaLLM(model = "llama3.3")

from langchain_ollama import OllamaLLM
from typing import List
import re

class ContinuousContractSummarizer:
    def __init__(self, model: str = "llama3.3"):
        self.llm = OllamaLLM(model=model)

    def _split_contract(self, text: str) -> List[str]:
        sections = re.split(r'\n\s*(ARTICLE|SECTION)\s+[IVXLCDM]+\s*\n', text)
        if len(sections) > 1:
            return sections
        
        return re.split(r'\n\d+\.\s+[A-Z][A-Z\s]+\n', text) or [text]

    def _generate_prompt(self, text: str, is_final: bool = False) -> str:
        if is_final:
            return f"""
            Create a FLOWING CONTRACT SUMMARY with these requirements:
            
            1. PRESERVE EXACTLY:
               - Numbers ($10,000) and dates (2025-01-01)
               - Key obligations (shall/must/will)
               - Conditional terms (if/unless)
            
            2. ORGANIZE NATURALLY by:
               - Who is involved
               - What must be done
               - When payments occur
               - How to terminate
            
            3. WRITE IN CONTINUOUS PROSE that reads like a narrative.
            
            CONTENT TO SUMMARIZE:
            {text}


            - Exclude explantion
            """
        else:
            return f"""
            Extract CRUCIAL CONTRACT TERMS for later summarization:
            
            RULES:
            - include parties details
            - keep termination, financial and risks
            - Copy VERBATIM: amounts, dates, deadlines
            - Keep obligations and conditions intact
            - Ignore boilerplate and signatures
            
            TEXT:
            {text}
            """

    # def summarize(self, contract_text: str):
    #     chunks = self._split_contract(contract_text)
        
    #     extracted_terms = []
    #     for chunk in chunks:
    #         prompt = self._generate_prompt(chunk)
    #         for chunk in self.llm.stream(prompt):
    #             print(chunk, end="", flush=True)
    #             extracted_terms.append(chunk)
        
    #     combined_terms = "\n".join(extracted_terms)
    #     final_prompt = self._generate_prompt(combined_terms, is_final=True)
        
    #     print("\n\n================ FINAL SUMMARY ==================")
    #     for chunk in self.llm.stream(final_prompt):
    #         print(chunk, end="")

    #     return final_prompt
    def summarize(self, contract_text: str) -> str:
        chunks = self._split_contract(contract_text)

        extracted_terms = []
        for chunk in chunks:
            prompt = self._generate_prompt(chunk)
            for response_chunk in self.llm.stream(prompt):
                extracted_terms.append(response_chunk)

        combined_terms = "\n".join(extracted_terms)
        final_prompt = self._generate_prompt(combined_terms, is_final=True)

        final_summary = []
        for response_chunk in self.llm.stream(final_prompt):
            final_summary.append(response_chunk)

        return "".join(final_summary)
    


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




@app.post("/summerizer_text")
async def receive_text(payload: dict):  
    text = payload.get("text", "")
    summarizer = ContinuousContractSummarizer()
    contract_text = text
    summary = summarizer.summarize(contract_text)
    return {"response": "msg received", "text": summary}

@app.post("/summerizer_upload_file")
async def summerizer_upload_file(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1].lower()
    summarizer = ContinuousContractSummarizer()
    # contract_text = 
    # summarizer.summarize(contract_text)
    # with open(f"uploads/{file.filename}", "wb") as f:
    #     f.write(await file.read())
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    extracted_text = extract_text_from_file(file_location, file_ext)
    contract_text = extracted_text
    summary = summarizer.summarize(contract_text)
    return {"response": "file uploaded successfully", "filename": summary}


if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print(f"FastAPI Public URL: {ngrok_tunnel.public_url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
