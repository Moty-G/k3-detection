from ultralytics import YOLO

from core import config


def load_model(model_path=config.MODEL_PATH):
    return YOLO(str(model_path))


def detect_frame(model, frame, conf, imgsz, track=False):
    class_ids = [i for i, name in model.names.items() if name in config.MODEL_CLASSES]
    options = dict(conf=conf, imgsz=imgsz, classes=class_ids, device="cpu", verbose=False)
    if track:
        result = model.track(frame, persist=True, **options)[0]
    else:
        result = model.predict(frame, **options)[0]

    detections = []
    boxes = result.boxes
    track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
    for box, cls, score, track_id in zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), track_ids):
        detections.append({
            "box": [int(v) for v in box],
            "class_name": model.names[cls],
            "confidence": float(score),
            "track_id": track_id,
        })
    return detections
