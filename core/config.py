from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "best.pt"
ALARM_PATH = ROOT_DIR / "assets" / "alarm.mp3"

PERSON_CLASS = "person"
HELMET_CLASS = "helmet"
VEST_CLASS = "safety-vest"
MODEL_CLASSES = [PERSON_CLASS, HELMET_CLASS, VEST_CLASS]

IMAGE_TYPES = ["jpg", "jpeg", "png"]
VIDEO_TYPES = ["mp4", "avi", "mov"]
MAX_UPLOAD_MB = 100

DEFAULT_CONFIDENCE = 0.4
DEFAULT_IMGSZ = 640
DEFAULT_CONFIRM_SECONDS = 2.0
DEFAULT_COOLDOWN_SECONDS = 10.0
DEFAULT_FRAME_STRIDE = 2
DEFAULT_ALARM_SOUND = True

HELMET_ZONE_RATIO = 0.33
VEST_ZONE_TOP = 0.2
VEST_ZONE_BOTTOM = 0.75

REQUIRED_PPE_OPTIONS = {
    "Helm & rompi": ("helmet", "vest"),
    "Helm saja": ("helmet",),
    "Rompi saja": ("vest",),
}

STATUS_COLORS = {
    "safe": (0, 200, 0),
    "unsafe": (0, 0, 230),
}

STATUS_LABELS = {
    "safe": "Aman",
    "unsafe": "Tidak Aman",
}

REASON_LABELS = {
    "no_helmet": "Tanpa helm",
    "no_vest": "Tanpa rompi",
    "no_both": "Tanpa helm & rompi",
}
