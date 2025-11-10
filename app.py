import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
import yt_dlp
from skimage.metrics import structural_similarity as ssim
import plotly.graph_objects as go


st.set_page_config(page_title="Keyframe Extractor & Visual Change Explorer", layout="wide")


# -----------------------------
# Frame Record
# -----------------------------
class FrameRecord:
    def __init__(self, index, time_s, bgr):
        self.index = index
        self.time_s = time_s
        self.bgr = bgr
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# -----------------------------
# Utility Functions
# -----------------------------
def resize_max(img, max_w=480):
    h, w = img.shape[:2]
    scale = min(max_w / w, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def rgb2gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def blend(a, b, alpha):
    return np.clip(a * alpha + b * (1 - alpha), 0, 255).astype(np.uint8)


# -----------------------------
# Video Input (Upload OR URL + yt-dlp)
# -----------------------------
def load_video_bytes(uploaded_file, url):
    if uploaded_file:
        return uploaded_file.getvalue()

    if not url:
        return None

    # Try direct download first
    try:
        response = requests.get(url, timeout=10, stream=True)
        content_type = response.headers.get("Content-Type", "").lower()
        if "video" in content_type or url.lower().endswith((".mp4",".mov",".avi",".mkv",".webm")):
            return response.content
    except:
        pass

    # Fall back to yt-dlp with progress UI
    st.write("Downloading via yt-dlp...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_hook(d):
        if d.get("status") == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes", None) or d.get("total_bytes_estimate", None)
            if total:
                frac = min(downloaded / total, 1.0)
                progress_bar.progress(frac)
        elif d.get("status") == "finished":
            progress_bar.progress(1.0)
            status_text.text("Download complete. Processing...")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            out_path = tmp.name

        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": out_path,
            "quiet": True,
            "progress_hooks": [progress_hook]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        with open(out_path, "rb") as f:
            return f.read()

    except Exception as e:
        st.error(f"Video download failed: {e}")
        return None


# -----------------------------
# Decode + Sample Frames
# -----------------------------
def decode_video(file_bytes, sample_fps):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file_bytes)
    tmp.flush()
    tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Could not decode video.")
        return [], 0, 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) / orig_fps

    # Adaptive sampling for long videos
    if duration > 120:
        sample_fps *= 0.25
    elif duration > 40:
        sample_fps *= 0.5

    stride = max(int(orig_fps / sample_fps), 1)

    frames = []
    idx = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(FrameRecord(idx, idx / orig_fps, resize_max(fr)))
        idx += 1

    cap.release()
    return frames, orig_fps, duration


# -----------------------------
# Compute Change Between Frames
# -----------------------------
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
            cv2.normalize(Ah, Ah); cv2.normalize(Bh, Bh)
            val = cv2.compareHist(Ah, Bh, cv2.HISTCMP_BHATTACHARYYA)

        else:  # MSE
            diff = (A.astype(float) - B.astype(float))
            val = np.mean(diff * diff) / (255 * 255)

        scores.append(val)

    return np.array(scores)


# -----------------------------
# Keyframe Selection
# -----------------------------
def pick_keyframes(frames, scores, k, min_gap_sec, sample_fps):
    gap = max(int(min_gap_sec * sample_fps), 1)
    order = list(np.argsort(scores)[::-1])
    picks = []

    for i in order:
        if len(picks) >= k:
            break
        if all(abs(i - p) >= gap for p in picks):
            picks.append(i)

    picks.sort()
    return [frames[i] for i in picks]


# -----------------------------
# Sidebar UI
# -----------------------------
with st.sidebar:
    st.header("Video Input")
    uploaded = st.file_uploader("Upload Video File", type=["mp4","mov","avi","mkv"])
    url = st.text_input("Or paste a video link (YouTube, Vimeo, TikTok, etc.)")

    st.header("Keyframe Settings")
    sample_fps = st.slider("Sampling FPS", 1.0, 8.0, 2.0)
    metric = st.radio("Change Metric", ["SSIM (1-SSIM)", "Color Histogram (Bhattacharyya)", "MSE"])

    if metric == "SSIM (1-SSIM)":
        st.caption("Measures structural layout changes. Best for shot boundaries and reframing.")
    elif metric == "Color Histogram (Bhattacharyya)":
        st.caption("Compares overall color palette. Useful for lighting / tone / scene mood changes.")
    else:
        st.caption("Pixel-wise difference. Sensitive to noise; use when fine details matter.")

    k = st.slider("Number of Keyframes", 5, 30, 12)
    min_gap_sec = st.slider("Minimum Time Between Keyframes (sec)", 0.0, 5.0, 0.5)


st.title("Keyframe Extractor & Visual Change Explorer")

file_bytes = load_video_bytes(uploaded, url)
if not file_bytes:
    st.info("Upload a video or enter a link to begin.")
    st.stop()


# Video Preview
with st.expander("Preview Video"):
    st.video(file_bytes)


frames, orig_fps, duration = decode_video(file_bytes, sample_fps)
if len(frames) < 2:
    st.error("Not enough frames extracted.")
    st.stop()

scores = compute_scores(frames, metric)
keyframes = pick_keyframes(frames, scores, k, min_gap_sec, sample_fps)


# -----------------------------
# Interactive Timeline (Plotly)
# -----------------------------
st.subheader("Change Score Timeline (Interactive)")

if len(scores) > 0:
    times = [frames[i].time_s for i in range(1, len(frames))]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=times, y=scores, mode="lines", name="Change Score"))

    keyframe_times = [fr.time_s for fr in keyframes]
    fig.add_trace(go.Scatter(
        x=keyframe_times,
        y=[scores[min(i, len(scores)-1)] for i in [frames.index(f) for f in keyframes]],
        mode="markers",
        marker=dict(size=8),
        name="Keyframes"
    ))

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Change Score",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Not enough data to display timeline.")


# -----------------------------
# Display Keyframes
# -----------------------------
st.subheader(f"Selected Keyframes ({len(keyframes)})")
cols = st.columns(min(len(keyframes), 6))
for i, fr in enumerate(keyframes):
    cols[i % len(cols)].image(fr.rgb, caption=f"Time: {fr.time_s:.2f}s | Frame {fr.index}", use_column_width=True)


# -----------------------------
# Frame Comparison
# -----------------------------
st.markdown("---")
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

col1, col2 = st.columns(2)
col1.image(A, caption="Frame A", use_column_width=True)
col2.image(B, caption="Frame B", use_column_width=True)

alpha = st.slider("Crossfade Blend Amount", 0.0, 1.0, 0.5)
st.image(blend(A,B,alpha), caption=f"Blend: {alpha:.2f}", use_column_width=True)

ssim_score = ssim(rgb2gray(A), rgb2gray(B), data_range=255)

if ssim_score > 0.85:
    interpretation = "Frames are nearly identical"
elif ssim_score > 0.60:
    interpretation = "Moderate visual change"
elif ssim_score > 0.35:
    interpretation = "Meaningful change in visual content"
else:
    interpretation = "Major scene or shot transition"

st.markdown(f"SSIM Score = {ssim_score:.4f}")
st.markdown(f"Interpretation: {interpretation}")
