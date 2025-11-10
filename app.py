import streamlit as st
import cv2
import numpy as np
import tempfile
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Keyframe Extractor & Visual Change Explorer", layout="wide")


# -----------------------------
# Frame container
# -----------------------------
class FrameRecord:
    def __init__(self, index, time_s, bgr):
        self.index = index
        self.time_s = time_s
        self.bgr = bgr
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# -----------------------------
# Utility functions
# -----------------------------
def resize_max(img, max_w=480):
    h, w = img.shape[:2]
    scale = min(max_w / w, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def rgb2gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

def blend(a, b, alpha):
    return np.clip(a * alpha + b * (1 - alpha), 0, 255).astype(np.uint8)

def ssim_diff_map(a_rgb, b_rgb):
    g1, g2 = rgb2gray(a_rgb), rgb2gray(b_rgb)
    score, diff = ssim(g1, g2, data_range=255, full=True)
    diff = (1 - diff)
    diff = (diff * 255).astype("uint8")
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), float(score)


# -----------------------------
# Efficient decoding (optimized for long videos)
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
            fr_small = resize_max(fr, max_w=480)
            frames.append(FrameRecord(idx, idx / orig_fps, fr_small))
        idx += 1

    cap.release()
    return frames, orig_fps, duration


# -----------------------------
# Change scoring and keyframe selection
# -----------------------------
def compute_scores(frames, metric):
    scores = []
    for i in range(1, len(frames)):
        A = frames[i-1].rgb
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
            diff = (A.astype(float)-B.astype(float))
            val = np.mean(diff*diff)/(255*255)

        scores.append(val)
    return np.array(scores)

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
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"], key="video_uploader")
    sample_fps = st.slider("Sampling FPS", 1.0, 8.0, 2.0)
    metric = st.radio("Change Metric", ["SSIM (1-SSIM)", "Color Histogram (Bhattacharyya)", "MSE"])
    k = st.slider("Keyframes", 5, 30, 12)
    min_gap_sec = st.slider("Minimum Time Between Keyframes (sec)", 0.0, 5.0, 0.5)

st.title("Keyframe Extractor & Visual Change Explorer")

if not uploaded:
    st.stop()

# -----------------------------
# Process Video
# -----------------------------
frames, orig_fps, duration = decode_video(uploaded.getvalue(), sample_fps)

if len(frames) < 2:
    st.error("Not enough frames were decoded.")
    st.stop()

scores = compute_scores(frames, metric)
keyframes = pick_keyframes(frames, scores, k, min_gap_sec, sample_fps)

# -----------------------------
# Display Keyframes
# -----------------------------
st.subheader(f"Selected Keyframes ({len(keyframes)})")
cols = st.columns(min(len(keyframes), 6))
for i, fr in enumerate(keyframes):
    label = f"Time: {fr.time_s:.2f}s | Frame {fr.index}"
    cols[i % len(cols)].image(fr.rgb, caption=label, use_column_width=True)

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

heat, ssim_score = ssim_diff_map(A,B)
st.image(heat, caption=f"SSIM Score = {ssim_score:.4f}")

if ssim_score > 0.85:
    interpretation = "Frames are nearly identical"
elif ssim_score > 0.60:
    interpretation = "Moderate visual change"
elif ssim_score > 0.35:
    interpretation = "Meaningful change in visual content"
else:
    interpretation = "Major scene or shot transition"

st.markdown(f"Interpretation: {interpretation}")
