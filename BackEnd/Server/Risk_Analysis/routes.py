from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
from BackEnd.Server.Risk_Analysis.config import UPLOAD_DIR
from BackEnd.Server.Risk_Analysis.utils import extract_text_from_file, docx_to_pdf, pdf_to_html, clean_extracted_text
from BackEnd.Server.Risk_Analysis.services import process_chunk, save_results_as_html

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "docx"]:
        return JSONResponse(content={"message": "Unsupported format"}, status_code=400)

    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())


    return JSONResponse(content={"message": "File processed"})

@router.post("/convert/")
async def html_to_pdf(request: Request):

    return FileResponse("output.pdf")