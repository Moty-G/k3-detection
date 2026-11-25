import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import config
from detector import load_model
from processor import process_image, process_video

st.set_page_config(page_title="Deteksi K3 Konstruksi", layout="wide")


@st.cache_resource
def get_model():
    return load_model(config.MODEL_PATH)


def read_settings():
    st.sidebar.header("Parameter Deteksi")
    ppe_label = st.sidebar.selectbox("APD wajib", list(config.REQUIRED_PPE_OPTIONS))
    return {
        "confidence": st.sidebar.slider("Confidence minimum", 0.1, 0.9, config.DEFAULT_CONFIDENCE, 0.05),
        "confirm_seconds": st.sidebar.number_input("Durasi konfirmasi (detik)", 0.0, 30.0, config.DEFAULT_CONFIRM_SECONDS, 0.5),
        "cooldown_seconds": st.sidebar.number_input("Cooldown (detik)", 0.0, 120.0, config.DEFAULT_COOLDOWN_SECONDS, 1.0),
        "frame_stride": st.sidebar.number_input("Frame stride", 1, 10, config.DEFAULT_FRAME_STRIDE),
        "imgsz": config.DEFAULT_IMGSZ,
        "required_ppe": config.REQUIRED_PPE_OPTIONS[ppe_label],
    }


def validate_upload(uploaded):
    extension = Path(uploaded.name).suffix.lower().lstrip(".")
    if extension not in config.IMAGE_TYPES + config.VIDEO_TYPES:
        return None, "Format berkas tidak didukung."
    if uploaded.size > config.MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"Ukuran berkas melebihi {config.MAX_UPLOAD_MB} MB."
    return ("image" if extension in config.IMAGE_TYPES else "video"), None


def run_video(uploaded, model, settings):
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(uploaded.getbuffer())
        input_path = temp_file.name
    progress = st.progress(0.0, text="Memproses video...")

    def on_progress(current, total):
        progress.progress(min(current / total, 1.0), text=f"Frame ke-{current} dari {total}")

    try:
        output_path, summary, violations = process_video(input_path, model, settings, on_progress)
    finally:
        os.remove(input_path)
    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()
    os.remove(output_path)
    return video_bytes, summary, violations


def show_alert(violations):
    if not violations:
        return
    st.error(f"PERINGATAN: {len(violations)} pelanggaran APD terkonfirmasi!")


def show_result(result):
    if result["kind"] == "image":
        st.image(result["media"], caption="Hasil deteksi", use_container_width=True)
        st.download_button("Unduh gambar hasil", result["download"], "hasil_deteksi.png", "image/png")
    else:
        st.video(result["media"])
        st.download_button("Unduh video hasil", result["media"], "hasil_deteksi.mp4", "video/mp4")

    summary = result["summary"]
    columns = st.columns(5)
    columns[0].metric("Total frame", summary["total_frames"])
    columns[1].metric("Orang terdeteksi", summary["persons"])
    columns[2].metric("Aman", summary["safe"])
    columns[3].metric("Tidak Aman", summary["unsafe"])
    columns[4].metric("Total pelanggaran", summary["violations"])

    if summary["persons"] == 0:
        st.info("Tidak ada orang terdeteksi.")


def show_violations(violations):
    if not violations:
        return
    st.subheader("Daftar & Bukti Pelanggaran")
    table = pd.DataFrame([{
        "Waktu video (detik)": v["video_time"],
        "ID pekerja": v["track_id"],
        "Jenis pelanggaran": config.REASON_LABELS[v["violation_type"]],
        "Confidence": v["confidence"],
    } for v in violations])
    st.dataframe(table, use_container_width=True, hide_index=True)

    columns = st.columns(3)
    for index, violation in enumerate(violations):
        caption = f"{violation['video_time']} s - {config.REASON_LABELS[violation['violation_type']]}"
        columns[index % 3].image(violation["evidence_image"], caption=caption, use_container_width=True)


st.title("Sistem Deteksi Keselamatan Kerja (K3) Konstruksi")
st.caption("Unggah gambar atau video untuk memeriksa kelengkapan helm dan rompi keselamatan pekerja.")

settings = read_settings()
uploaded = st.file_uploader("1. Unggah gambar atau video", type=config.IMAGE_TYPES + config.VIDEO_TYPES)

kind, error = (None, None)
if uploaded:
    kind, error = validate_upload(uploaded)
    if error:
        st.error(error)

if st.button("Proses", type="primary", disabled=kind is None):
    model = get_model()
    try:
        if kind == "image":
            image, summary, violations = process_image(uploaded.getvalue(), model, settings)

            download = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
            st.session_state["result"] = {"kind": kind, "media": image, "download": download, "summary": summary, "violations": violations}
        else:
            video_bytes, summary, violations = run_video(uploaded, model, settings)
            st.session_state["result"] = {"kind": kind, "media": video_bytes, "summary": summary, "violations": violations}
    except ValueError as exc:
        st.error(str(exc))
        st.session_state.pop("result", None)

result = st.session_state.get("result")
if result:
    show_alert(result["violations"])
    show_result(result)
    show_violations(result["violations"])
