## T-Bank Logo Detection API

REST API сервис для детекции логотипа Т-Банка на изображениях. Сервис использует обученную модель YOLO и возвращает координаты найденных логотипов.

### Возможности
- Эндпоинт `POST /detect` принимает изображение и возвращает список `bbox` по спецификации Pydantic
- Поддерживаемые форматы: JPEG, PNG, BMP, WEBP
- Время обработки: ориентировано < 10 сек/изображение на CPU, быстрее на GPU
- Эндпоинт `GET /health` для проверки работоспособности

### API контракт (Pydantic)
```python
class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]
```

Пример запроса:
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@demo.jpg"
```

### Запуск локально
1. Установить зависимости:
```bash
pip install -r requirements.txt
```
2. Убедитесь, что веса модели доступны локально:
- По умолчанию путь: `D:\\TrainModelTbank\\runs\\detect\\logo_detection3\\weights\\best.pt`
- Или задайте переменные окружения:
```bash
set WEIGHTS_PATH=D:\\path\\to\\best.pt
:: при наличии URL можно скачать при старте
set WEIGHTS_URL=https://.../best.pt
```
3. Запустить сервис:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
Сборка образа:
```bash
docker build -t tbank-logo-api .
```
Запуск (с локальными весами, проброс порта 8000):
```bash
docker run --rm -p 8000:8000 -e WEIGHTS_PATH=/weights/best.pt -v D:\\TrainModelTbank\\runs\\detect\\logo_detection3\\weights:/weights tbank-logo-api
```
Либо скачать веса по URL при старте (пример с Яндекс.Диском):
```bash
docker run --rm -p 8000:8000 -e WEIGHTS_URL="https://disk.yandex.ru/d/rZ0bPhrG0MZuFQ" tbank-logo-api
```

### Конфигурация (переменные окружения)
- `WEIGHTS_PATH` — путь к весам модели (.pt). По умолчанию: `weights/best.pt`
- `WEIGHTS_URL` — публичная ссылка (поддерживается Яндекс.Диск). При старте файл будет скачан в `WEIGHTS_PATH`
- `CONF_THRESHOLD` (default 0.25)
- `IOU_THRESHOLD` (default 0.45)
- `IMG_SIZE` (default 640)
- `DEVICE` — `cpu`, `cuda`, или `auto` (default `auto`)

### Валидация качества
Скрипт `validate.py` считает Precision/Recall/F1 при IoU=0.5 на предоставленных разметках (формат YOLO):
```bash
python validate.py --images_dir dataset/val/images --labels_dir dataset/val/labels --iou 0.5 --output validation_metrics.json
```
Загрузка тестового набора с Яндекс.Диска (пример):
```bash
# скачиваем test/images
python scripts/download_yadisk_public.py --public-link "https://disk.yandex.ru/d/8wH2owJfcRL6Gg" --remote-path "test/images" --dest dataset/test/images
# скачиваем test/labels
python scripts/download_yadisk_public.py --public-link "https://disk.yandex.ru/d/8wH2owJfcRL6Gg" --remote-path "test/labels" --dest dataset/test/labels
```
Вывод:
```json
{
  "images": 302,
  "tp": 123,
  "fp": 7,
  "fn": 9,
  "precision": 0.946,
  "recall": 0.932,
  "f1": 0.939,
  "iou_threshold": 0.5
}
```

### Подход к решению
- Использована модель YOLO (Ultralytics). Обучение произведено на кастомной разметке логотипа Т-Банка, включая негативные примеры.
- Проведены эксперименты по выбору `imgsz`, порога `conf`, `iou` NMS и балансировке классов (single class).
- Для инференса — фиксация детерминированных преобразований и ограничение времени обработки; допускается батч=1.

### Публикация весов и валидационного набора
- Рекомендуется загрузить `best.pt` и валидационную выборку в открытый доступ (GitHub Releases, облако) и указать ссылки здесь.

### Лицензия
MIT








