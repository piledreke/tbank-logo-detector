import os
from functools import lru_cache


class Settings:
    """Конфигурация приложения"""

    # Путь к весам по умолчанию (локальные веса обученной модели)
    DEFAULT_WEIGHTS_PATH: str = os.getenv("WEIGHTS_PATH", "weights/best.pt")

    # Опциональная ссылка на веса (если размещены в облаке)
    WEIGHTS_URL: str | None = os.getenv("WEIGHTS_URL")

    # Конфигурация детекции
    CONF_THRESHOLD: float = float(os.getenv("CONF_THRESHOLD", "0.25"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    IMG_SIZE: int = int(os.getenv("IMG_SIZE", "640"))

    # Производительность
    DEVICE: str = os.getenv("DEVICE", "auto")  # "cpu", "cuda", "auto"

    # Ограничения и таймауты
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))  # лимит входного файла
    REQUEST_TIMEOUT_S: float = float(os.getenv("REQUEST_TIMEOUT_S", "10"))  # таймаут инференса
    MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))  # лимит на количество пикселей (напр. 5000x5000)

    # Логи
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", "4"))
    WARMUP: int = int(os.getenv("WARMUP", "0"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()








