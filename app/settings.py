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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()








