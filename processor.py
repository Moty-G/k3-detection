import tempfile

import cv2
import imageio
import numpy as np

import config
from detector import detect_frame
from rules import evaluate_ppe, update_temporal


def draw_workers(frame, workers):
    for worker in workers:
        x1, y1, x2, y2 = worker["box"]
        color = config.STATUS_COLORS[worker["status"]]
        label = config.STATUS_LABELS[worker["status"]]
        if worker["reason"]:
            label += f" - {config.REASON_LABELS[worker['reason']]}"
        if worker["track_id"] is not None:
            label = f"#{worker['track_id']} {label}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(y1 - text_h - 8, 0)), (x1 + text_w + 6, max(y1, text_h + 8)), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(y1 - 5, text_h + 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def build_summary(total_frames, workers_seen, safe_count, unsafe_count, violation_count):
    return {
        "total_frames": total_frames,
        "persons": workers_seen,
        "safe": safe_count,
        "unsafe": unsafe_count,
        "violations": violation_count,
    }


def to_violation(worker, video_time, frame_bgr):
    return {
        "video_time": round(video_time, 1),
        "track_id": worker["track_id"],
        "violation_type": worker["reason"],
        "confidence": round(worker["confidence"], 2),
        "evidence_image": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
    }


def process_image(image_bytes, model, settings):
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Berkas gambar rusak atau tidak dapat dibaca.")

    detections = detect_frame(model, frame, settings["confidence"], settings["imgsz"])
    workers = evaluate_ppe(detections, settings["required_ppe"])
    annotated = draw_workers(frame, workers)

    unsafe = [w for w in workers if w["status"] == "unsafe"]
    violations = [to_violation(w, 0.0, annotated) for w in unsafe]
    summary = build_summary(1, len(workers), len(workers) - len(unsafe), len(unsafe), len(violations))
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), summary, violations


def process_video(video_path, model, settings, on_progress=None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Berkas video rusak atau tidak dapat dibaca.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = settings["frame_stride"]
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(output_path, fps=fps / stride, codec="libx264", pixelformat="yuv420p", macro_block_size=1)

    if getattr(model.predictor, "trackers", None):
        del model.predictor.trackers
    state = {}
    seen_ids, unsafe_ids, violations = set(), set(), []
    frame_index = 0
    processed = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            video_time = frame_index / fps
            detections = detect_frame(model, frame, settings["confidence"], settings["imgsz"], track=True)
            workers = evaluate_ppe(detections, settings["required_ppe"])
            new_violations = update_temporal(workers, video_time, state, settings["confirm_seconds"], settings["cooldown_seconds"])

            annotated = draw_workers(frame, workers)
            for worker in new_violations:
                violations.append(to_violation(worker, video_time, annotated))
            for worker in workers:
                if worker["track_id"] is not None:
                    seen_ids.add(worker["track_id"])
                    if worker["status"] == "unsafe":
                        unsafe_ids.add(worker["track_id"])

            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            processed += 1
        frame_index += 1
        if on_progress:
            on_progress(frame_index, max(total_frames, frame_index))

    capture.release()
    writer.close()
    if processed == 0:
        raise ValueError("Video tidak berisi frame yang dapat diproses.")

    summary = build_summary(frame_index, len(seen_ids), len(seen_ids - unsafe_ids), len(unsafe_ids), len(violations))
    return output_path, summary, violations
