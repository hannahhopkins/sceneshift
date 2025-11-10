import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
import yt_dlp
import subprocess
from skimage.metrics import structural_similarity as ssim
import plotly.graph_objects as go

st.set_page_config(page_title="Keyframe Extractor & Visual Change Explorer", layout="wide")


# ------------------------------------------------------------
# Frame structure
# ------------------------------------------------------------
class FrameRecord:
    def __init__(self, index, time_s, bgr):
        self.index = index
        self.time_s = time_s
        self.bgr = bgr
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def resize_max(img, max_w=480):
    h, w = img.shape[:2]
    s = min(max_w / w, 1.0)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def rgb2gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def blend(a, b, alpha):
    return np.clip(a * alpha + b * (1 - alpha), 0, 255).astype(np.uint8)


def can_decode_video(file_bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file_bytes)
    tmp.flush()
    cap = cv2.VideoCapture(tmp.name)
    ok, _ = cap.read()
    cap.release()
    return ok


# ------------------------------------------------------------
# Cached downloader with automatic H.264 fallback + re-encode
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_video_cached(url, quality_format, no_audio):
    progress_bar = st.progress(0, text="Preparing download…")

    def hook(d):
        if d.get("status") == "downloading":
            t = d.get("total_bytes") or d.get("total_bytes_estimate")
            if t:
                frac = d.get("downloaded_bytes", 0) / t
                progress_bar.progress(frac, text=f"Downloading: {int(frac*100)}%")
        elif d.get("status") == "finished":
            progress_bar.progress(1.0, text="Processing video…")

    def yt_download(fmt):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            out_path = tmp.name
        ydl_opts = {
            "format": fmt,
            "outtmpl": out_path,
            "quiet": True,
            "progress_hooks": [hook],
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        with open(out_path, "rb") as f:
            return f.read(), out_path

    # Initial download attempt
    if no_audio:
        fmt = f"{quality_format}[ext=mp4]"
    else:
        fmt = f"{quality_format}+bestaudio[ext=mp4]/best[ext=mp4]"

    try:
        raw_bytes, local_path = yt_download(fmt)
    except Exception:
        raw_bytes, local_path = None, None

    # If decodable → return
    if raw_bytes and can_decode_video(raw_bytes):
        return raw_bytes

    # --------------------------------------------------------
    # Fallback: FORCE re-encode to H.264 + yuv420p (OpenCV-safe)
    # --------------------------------------------------------
    progress_bar.progress(0, text="Re-encoding to H.264 for compatibility…")

    reencoded_path = local_path + "_h264.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", local_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level:v", "4.1",
            "-c:a", "aac",
            "-movflags", "+faststart",
            reencoded_path
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    with open(reencoded_path, "rb") as f:
        return f.read()


def load_video_bytes(uploaded_file, url, quality_format, no_audio):
    if uploaded_file:
        return uploaded_file.getvalue()
    if url:
        return download_video_cached(url, quality_format, no_audio)
    return None


# ------------------------------------------------------------
# Decode + sample frames
# ------------------------------------------------------------
def decode_video(file_bytes, sample_fps):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file_bytes)
    tmp.flush()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        return [], 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) / fps

    # Adaptive sampling
    if duration > 120:
        sample_fps *= 0.25
    elif duration > 40:
        sample_fps *= 0.5

    stride = max(int(fps / sample_fps), 1)

    frames = []
    i = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if i % stride == 0:
            frames.append(FrameRecord(i, i / fps, resize_max(fr)))
        i += 1

    cap.release()
    return frames, fps, duration


# ------------------------------------------------------------
# Change scoring
# ------------------------------------------------------------
def compute_scores(frames, metric):
    scores = []
    for i in range(1, len(frames)):
        A = frames[i - 1].rgb
        B = frames[i].rgb

        if metric == "SSIM (1-SSIM)":
            val = 1 - ssim(rgb2gray(A), rgb2gray(B), data_range=255)
        elif metric == "Color Histogram (Bhattacharyya)":
            Ah = cv2.calcHist([cv2.cvtColor(A, cv2.COLOR_RGB2HSV)], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
            Bh = cv2.calcHist([cv2.cvtColor(B, cv2.COLOR_RGB2HSV)], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
            cv2.normalize(Ah, Ah)
            cv2.normalize(Bh, Bh)
            val = cv2.compareHist(Ah, Bh, cv2.HISTCMP_BHATTACHARYYA)
        else:
            diff = (A.astype(float) - B.astype(float))
            val = np.mean(diff * diff) / (255 * 255)

        scores.append(val)
    return np.array(scores)


# ------------------------------------------------------------
# Keyframe selection
# ------------------------------------------------------------
def pick_keyframes(frames, scores, k, min_gap_sec, sample_fps):
    gap = max(int(min_gap_sec * sample_fps), 1)
    order = np.argsort(scores)[::-1]
    picks = []
    for i in order:
        if len(picks) >= k:
            break
        if all(abs(i - p) >= gap for p in picks):
            picks.append(i)
    picks.sort()
    return [frames[i] for i in picks]


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    uploaded = st.file_uploader("Upload Video", type=["mp4","mov","avi","mkv"])
    url = st.text_input("Or paste a video link")

    quality_choice = st.selectbox("Video Quality", ["Low (≤480p)", "Medium (≤720p)", "High"])
    no_audio = st.checkbox("Do not download audio (faster)")

    if quality_choice == "Low (≤480p)":
        quality_format = "bv*[height<=480]"
    elif quality_choice == "Medium (≤720p)":
        quality_format = "bv*[height<=720]"
    else:
        quality_format = "bestvideo"

    sample_fps = st.slider("Sampling FPS", 1.0, 8.0, 2.0)
    metric = st.radio("Change Metric", ["SSIM (1-SSIM)", "Color Histogram (Bhattacharyya)", "MSE"])
    k = st.slider("Number of Keyframes", 5, 30, 12)
    min_gap_sec = st.slider("Minimum spacing between keyframes (seconds)", 0.0, 5.0, 0.5)


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
st.title("Keyframe Extractor & Visual Change Explorer")

file_bytes = load_video_bytes(uploaded, url, quality_format, no_audio)
if file_bytes is None:
    st.stop()

with st.expander("Preview"):
    st.video(file_bytes)

frames, fps, duration = decode_video(file_bytes, sample_fps)
if len(frames) < 2:
    st.error("Video could not be decoded even after fallback.")
    st.stop()

scores = compute_scores(frames, metric)
keyframes = pick_keyframes(frames, scores, k, min_gap_sec, sample_fps)


# ------------------------------------------------------------
# Timeline
# ------------------------------------------------------------
st.subheader("Visual Change Timeline")
times = [frames[i].time_s for i in range(1, len(frames))]
fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=scores, mode="lines"))
fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Change Score")
st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Keyframes
# ------------------------------------------------------------
st.subheader("Selected Keyframes")
cols = st.columns(min(len(keyframes), 6))
for i, fr in enumerate(keyframes):
    cols[i % len(cols)].image(fr.rgb, caption=f"{fr.time_s:.2f}s (Frame {fr.index})", use_column_width=True)


# ------------------------------------------------------------
# Compare Frames
# ------------------------------------------------------------
st.subheader("Compare Frames")
labels = [f"{fr.time_s:.2f}s (Frame {fr.index})" for fr in keyframes]
iA = st.selectbox("Frame A", range(len(keyframes)), format_func=lambda i: labels[i])
iB = st.selectbox("Frame B", range(len(keyframes)), format_func=lambda i: labels[i])

A = keyframes[iA].rgb
B = keyframes[iB].rgb
h = min(A.shape[0], B.shape[0])
w = min(A.shape[1], B.shape[1])
A = cv2.resize(A, (w,h))
B = cv2.resize(B, (w,h))

st.image(A, caption="Frame A", use_column_width=True)
st.image(B, caption="Frame B", use_column_width=True)

alpha = st.slider("Blend", 0.0, 1.0, 0.5)
st.image(blend(A,B,alpha), caption=f"Blend {alpha:.2f}", use_column_width=True)

score = ssim(rgb2gray(A), rgb2gray(B), data_range=255)
st.markdown(f"SSIM Score = {score:.4f}")
