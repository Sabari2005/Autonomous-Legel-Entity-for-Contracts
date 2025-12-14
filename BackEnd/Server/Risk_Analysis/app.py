from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pyngrok import ngrok
from BackEnd.Server.Risk_Analysis.config import UPLOAD_DIR
from BackEnd.Server.Risk_Analysis.routes import router

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.include_router(router)

if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print(f"FastAPI Public URL: {ngrok_tunnel.public_url}")
    uvicorn.run(app, host="0.0.0.0", port=8000)