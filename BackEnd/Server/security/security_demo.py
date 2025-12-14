from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import re
import asyncio
import httpx

app = FastAPI()
templates = Jinja2Templates(directory="templates")

BLACKLISTED_IPS = {"192.168.0.100", "10.0.0.25"}
RECAPTCHA_SECRET_KEY = "your_recaptcha_secret_key"

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

@app.middleware("http")
async def ddos_middleware(request: Request, call_next):
    user_agent = request.headers.get("user-agent", "")
    client_ip = request.client.host

    if client_ip in BLACKLISTED_IPS:
        return JSONResponse(status_code=403, content={"detail": "Access denied."})

    if not user_agent or re.search(r"bot|crawl|spider|scanner", user_agent, re.IGNORECASE):
        return JSONResponse(status_code=403, content={"detail": "Bot traffic blocked"})

    if "suspicious-header" in request.headers:
        await asyncio.sleep(2)

    response = await call_next(request)

    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

@app.get("/", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/accept")
@limiter.limit("5/minute")
async def accept_message(
    request: Request,
    user_input: str = Form(...),
    g_recaptcha_response: str = Form(alias="g-recaptcha-response")
):
    async with httpx.AsyncClient() as client:
        captcha_response = await client.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": RECAPTCHA_SECRET_KEY, "response": g_recaptcha_response}
        )
        captcha_result = captcha_response.json()
        if not captcha_result.get("success"):
            raise HTTPException(status_code=400, detail="Invalid reCAPTCHA")

    if re.search(r"(select|union|insert|drop|--|;|')", user_input, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Suspicious input detected.")

    print(f"Received message: {user_input}")
    return {"message": "Message received securely"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, workers=2)
