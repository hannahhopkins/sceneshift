import os
import io
import math
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import streamlit as st

# Core CV + metrics
import cv2
from skimage.metrics import structural_similarity as ssim

try:
    import yt_dlp  # optional
    YTDLP_AVAILABLE = True
except Exception:
    YTDLP_AVAILABLE = False

st.set_page_config(page_title="Keyframe Extractor & Visual Change Explorer", layout="wide")
st.title("Keyframe Extractor & Visual Change Explorer")


@dataclass
class FrameRecord:
    index: int
    time_s: float
    image_rgb: np.ndarray


def bgr2rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb2gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def resize_max(img: np.ndarray, max_w: int = 960, max_h: int = 540) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def overlay_timestamp(img_rgb: np.ndarray, t: float) -> np.ndarray:
    img = img_rgb.copy()
    h, w = img.shape[:2]
    label = f"{t:.2f}s"
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w, h) / 600)
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x0, y0 = 10, 10
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x0 + pad, y0 + th + pad // 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------------
# NEW: Speed Optimization - Downscale frames for scoring only
# ------------------------------------------------------------------------
def downsnap(img, max_side=300):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------
# Video download via yt-dlp
# ------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _download_video_bytes(url: str, quality: str, no_audio: bool) -> Optional[bytes]:
    if not YTDLP_AVAILABLE:
        return None

    base_v = "bv*[vcodec~='^(avc1|h264|x264|vp9)']"
    if no_audio:
        fmt = f"{base_v}/bv*"
    else:
        fmt = f"{base_v}+ba/{base_v}/b"

    quality_caps = {"Low": 480, "Medium": 720, "Auto": 1080, "Lossless": 4320}
    height_cap = quality_caps.get(quality, 1080)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp_path = tmp.name

    try:
        ydl_opts_file = {
            'format': fmt,
            'outtmpl': tmp_path,
            'quiet': True,
            'merge_output_format': 'mp4',
            'retries': 3,
        }
        with yt_dlp.YoutubeDL(ydl_opts_file) as ydl:
            ydl.download([url])
        with open(tmp_path, 'rb') as f:
            return f.read()
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


def _write_temp_file(data: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def _transcode_to_h264(input_path: str) -> Optional[str]:
    try:
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', input_path,
            '-an',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            out_path
        ]
        subprocess.run(cmd, check=True)
        return out_path
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def decode_video_bytes(file_bytes: bytes, sample_fps: float = 2.0) -> Tuple[List[FrameRecord], float]:
    if not file_bytes:
        return [], 0.0

    in_path = _write_temp_file(file_bytes)

    def _decode(path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return [], 0.0
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(int(orig_fps / max(sample_fps, 0.1)), 1)

        frames = []
        idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frame_small = resize_max(frame_bgr)
                frames.append(FrameRecord(idx, idx / orig_fps, bgr2rgb(frame_small)))
            idx += 1
        cap.release()
        return frames, float(orig_fps)

    frames, fps = _decode(in_path)
    if len(frames) < 2:
        alt = _transcode_to_h264(in_path)
        if alt:
            frames, fps = _decode(alt)
            try: os.remove(alt)
            except: pass

    try: os.remove(in_path)
    except: pass

    return frames, fps


@st.cache_data(show_spinner=False)
def compute_consecutive_scores(frames: List[FrameRecord], metric: str) -> np.ndarray:
    if len(frames) < 2:
        return np.array([])

    scores = []
    for i in range(1, len(frames)):
        a = downsnap(frames[i - 1].image_rgb, max_side=300)
        b = downsnap(frames[i].image_rgb, max_side=300)

        if metric == "SSIM (1-SSIM)":
            val = 1 - float(ssim(rgb2gray(a), rgb2gray(b), data_range=255))
        elif metric == "Color Histogram (Bhattacharyya)":
            ha = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
            hb = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
            bins = (8, 8, 8)
            ranges = [0,180,0,256,0,256]
            ha_hist = cv2.calcHist([ha],[0,1,2],None,bins,ranges)
            hb_hist = cv2.calcHist([hb],[0,1,2],None,bins,ranges)
            cv2.normalize(ha_hist, ha_hist)
            cv2.normalize(hb_hist, hb_hist)
            val = float(cv2.compareHist(ha_hist, hb_hist, cv2.HISTCMP_BHATTACHARYYA))
        else:
            diff = (a.astype(np.float32) - b.astype(np.float32))
            val = float(np.mean(diff * diff)) / (255*255)

        scores.append(val)
    return np.array(scores, dtype=np.float32)


def _nms(scores, k, min_gap):
    order = list(np.argsort(scores)[::-1])
    sel = []
    for i in order:
        if len(sel) >= k:
            break
        if all(abs(i-s) >= min_gap for s in sel):
            sel.append(int(i))
    sel.sort()
    return sel


@st.cache_data(show_spinner=False)
def select_keyframes(frames, metric, k, min_gap_sec, sample_fps):
    scores = compute_consecutive_scores(frames, metric)
    min_gap = max(int(min_gap_sec * sample_fps), 1)
    idxs = _nms(scores, k, min_gap)
    chosen = [frames[i] for i in idxs]
    if frames and (not chosen or chosen[0].index != frames[0].index):
        chosen = [frames[0]] + chosen
    if frames and chosen[-1].index != frames[-1].index:
        chosen = chosen + [frames[-1]]
    return chosen[:k]


# UI ---------------------------------------------------------------------
with st.sidebar:
    source_mode = st.selectbox("Source", ["Upload file", "URL (yt-dlp)"])
    if source_mode == "Upload file":
        uploaded = st.file_uploader("Video file", type=["mp4","mov","avi","mkv","webm"])
        url = None
    else:
        url = st.text_input("Video URL")
        uploaded = None

    quality = st.selectbox("URL Download Quality", ["Auto","Medium","Low","Lossless"])
    no_audio = st.checkbox("No audio (faster)", True)
    sample_fps = st.slider("Sampling FPS", 0.5, 8.0, 2.0, 0.5)
    metric = st.selectbox("Change Metric", ["SSIM (1-SSIM)","Color Histogram (Bhattacharyya)","MSE"])
    k = st.slider("Keyframes", 5, 30, 12)
    min_gap_sec = st.slider("Minimum spacing (sec)", 0.0, 5.0, 0.5)


progress = st.empty()
progress_bar = st.progress(0)

if uploaded:
    video_bytes = uploaded.getvalue()
elif url:
    progress.write("Downloading…")
    video_bytes = _download_video_bytes(url, quality, no_audio)
    progress_bar.progress(100)
else:
    st.stop()

with st.spinner("Decoding video…"):
    frames, fps = decode_video_bytes(video_bytes, sample_fps)

if len(frames) < 2:
    st.error("Could not decode frames.")
    st.stop()

with st.spinner("Selecting keyframes…"):
    keyframes = select_keyframes(frames, metric, k, min_gap_sec, sample_fps)

st.subheader("Selected Keyframes")
cols = st.columns(min(len(keyframes), 6))
for i, fr in enumerate(keyframes):
    cols[i % len(cols)].image(overlay_timestamp(fr.image_rgb, fr.time_s), use_column_width=True)

st.subheader("Compare Frames")
if len(keyframes) >= 2:
    labels = [f"{fr.time_s:.2f}s (Frame {fr.index})" for fr in keyframes]
    iA = st.selectbox("Frame A", list(range(len(keyframes))), format_func=lambda i: labels[i])
    iB = st.selectbox("Frame B", list(range(len(keyframes))), format_func=lambda i: labels[i])
    A = keyframes[iA].image_rgb
    B = keyframes[iB].image_rgb
    h = min(A.shape[0],B.shape[0])
    w = min(A.shape[1],B.shape[1])
    A = cv2.resize(A,(w,h))
    B = cv2.resize(B,(w,h))
    st.image(A, caption="A", use_column_width=True)
    st.image(B, caption="B", use_column_width=True)
    alpha = st.slider("Blend", 0.0,1.0,0.5,0.05)
    mix = (A*alpha + B*(1-alpha)).astype(np.uint8)
    st.image(mix, caption=f"Blend {alpha:.2f}", use_column_width=True)
    score = ssim(rgb2gray(A), rgb2gray(B), data_range=255)
    st.markdown(f"SSIM: {score:.4f}")

