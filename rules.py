import config


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def is_inside(point, x1, y1, x2, y2):
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def has_helmet(person_box, helmets):
    x1, y1, x2, y2 = person_box
    zone_bottom = y1 + (y2 - y1) * config.HELMET_ZONE_RATIO
    return any(is_inside(box_center(h["box"]), x1, y1 - (y2 - y1) * 0.1, x2, zone_bottom) for h in helmets)


def has_vest(person_box, vests):
    x1, y1, x2, y2 = person_box
    top = y1 + (y2 - y1) * config.VEST_ZONE_TOP
    bottom = y1 + (y2 - y1) * config.VEST_ZONE_BOTTOM
    return any(is_inside(box_center(v["box"]), x1, top, x2, bottom) for v in vests)


def get_reason(missing_helmet, missing_vest):
    if missing_helmet and missing_vest:
        return "no_both"
    if missing_helmet:
        return "no_helmet"
    if missing_vest:
        return "no_vest"
    return None


def evaluate_ppe(detections, required_ppe):
    persons = [d for d in detections if d["class_name"] == config.PERSON_CLASS]
    helmets = [d for d in detections if d["class_name"] == config.HELMET_CLASS]
    vests = [d for d in detections if d["class_name"] == config.VEST_CLASS]

    workers = []
    for person in persons:
        missing_helmet = "helmet" in required_ppe and not has_helmet(person["box"], helmets)
        missing_vest = "vest" in required_ppe and not has_vest(person["box"], vests)
        reason = get_reason(missing_helmet, missing_vest)
        workers.append({
            "box": person["box"],
            "track_id": person["track_id"],
            "status": "unsafe" if reason else "safe",
            "reason": reason,
            "confidence": person["confidence"],
        })
    return workers


def update_temporal(workers, video_time, state, confirm_sec, cooldown_sec):
    new_violations = []
    for worker in workers:
        track_id = worker["track_id"]
        if track_id is None:
            continue
        info = state.setdefault(track_id, {"unsafe_since": None, "cooldown_until": -1.0})

        if worker["status"] != "unsafe":
            info["unsafe_since"] = None
            continue
        if info["unsafe_since"] is None:
            info["unsafe_since"] = video_time

        confirmed = video_time - info["unsafe_since"] >= confirm_sec
        if confirmed and video_time >= info["cooldown_until"]:
            info["cooldown_until"] = video_time + cooldown_sec
            new_violations.append(worker)
    return new_violations
