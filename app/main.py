from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from .models import DetectionResponse, ErrorResponse
from .inference import YOLODetector


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


@app.get("/health", response_model=dict)
async def health() -> dict:
    return {"status": "ok"}


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

    try:
        detections = detector.detect(image_bytes)
        return DetectionResponse(detections=detections)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Weights not found", detail=str(e)).model_dump())
    except Exception as e:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Internal error", detail=str(e)).model_dump())


