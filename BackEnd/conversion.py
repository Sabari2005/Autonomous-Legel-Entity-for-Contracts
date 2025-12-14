from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
import pdfkit
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from bs4 import BeautifulSoup

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HTMLRequest(BaseModel):
    html_content: str




@app.post("/generate_pdf")
async def generate_pdf(request: HTMLRequest):
    html_content = request.html_content
    config = pdfkit.configuration(wkhtmltopdf="C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe")
    pdfkit.from_string(html_content, "output.pdf", configuration=config)

    pdf_path = "generated.pdf"
    print(html_content)
# async def generate_pdf():
    soup = BeautifulSoup(html_content, "html.parser")

    pdf_file = "output.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    y_position = 750  
    for tag in soup.find_all(["h1", "p"]):
        text = tag.get_text()
        c.drawString(100, y_position, text)
        y_position -= 20

    c.save()

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
