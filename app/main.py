"""
Основной модуль FastAPI приложения.

Содержит эндпоинты:
- GET /health — проверка статуса, устройство и наличие весов;
- POST /detect — детекция логотипа на изображении.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import asyncio

from .models import DetectionResponse, ErrorResponse
from .inference import YOLODetector
from .settings import get_settings


settings = get_settings()

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("api")

app = FastAPI(title="T-Bank Logo Detector", version="1.0.0")

# CORS (на случай фронтенд-клиента)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


detector = YOLODetector()
semaphore = asyncio.Semaphore(settings.MAX_CONCURRENCY)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Опциональный тёплый прогон модели для снижения задержки первого запроса
if settings.WARMUP:
    try:
        from PIL import Image
        from io import BytesIO

        im = Image.new("RGB", (64, 64), color=(0, 0, 0))
        buf = BytesIO()
        im.save(buf, format="PNG")
        _ = detector.detect(buf.getvalue())
    except Exception:
        logger.warning("Warmup failed, continuing without warmup")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await asyncio.wait_for(call_next(request), timeout=settings.REQUEST_TIMEOUT_S)
            return response
        except asyncio.TimeoutError:
            logger.error("Превышен таймаут обработки запроса")
            return JSONResponse(status_code=504, content=ErrorResponse(error="Timeout", detail="Request processing timed out").model_dump())
        except Exception as e:
            logger.exception("Ошибка обработчика запроса: %s", e)
            return JSONResponse(status_code=500, content=ErrorResponse(error="Internal error", detail=str(e)).model_dump())


app.add_middleware(RequestLoggingMiddleware)


@app.get("/health", response_model=dict)
async def health() -> dict:
    """Простой health-check с диагностической информацией."""
    info = detector.runtime_info()
    return {"status": "ok", **info}


@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)) -> DetectionResponse:
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp",
    }:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Ограничение размера входного файла
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (> {settings.MAX_FILE_SIZE_MB} MB)")

    # Проверка, что это валидное изображение и ограничение по числу пикселей
    try:
        from PIL import Image
        from io import BytesIO

        im = Image.open(BytesIO(image_bytes))
        im.verify()  # проверка целостности
        im = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = im.size
        if w * h > settings.MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image pixels exceed limit")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        async with semaphore:
            detections = detector.detect(image_bytes)
        return DetectionResponse(detections=detections)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Weights not found", detail=str(e)).model_dump())
    except Exception as e:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Internal error", detail=str(e)).model_dump())


