import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from typing import List, Tuple
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Keyframe Extractor & Visual Change Explorer", layout="wide")

# -----------------------------
# Frame container
# -----------------------------
class FrameRecord:
    def __init__(self, index: int, time_s: float, bgr: np.ndarray):
        self.index = index
        self.time_s = time_s
        self.bgr = bgr
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# -----------------------------
# Helpers
# -----------------------------
def resize_max(img, max_w=960, max_h=540):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def rgb2gray(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

def blend(a, b, alpha):
    return np.clip(a*alpha + b*(1-alpha), 0, 255).astype(np.uint8)

def ssim_diff_map(a_rgb, b_rgb):
    g1, g2 = rgb2gray(a_rgb), rgb2gray(b_rgb)
    score, diff = ssim(g1, g2, data_range=255, full=True)
    diff = (1 - diff)
    diff = (diff * 255).astype("uint8")
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), float(score)


# -----------------------------
# Video decoding (≠ get_cache_path — FIXED)
# -----------------------------
@st.cache_data(show_spinner=False)
def decode_video(file_bytes: bytes, sample_fps: float):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file_bytes)
    tmp.flush()
    tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Could not decode video. Check codec availability.")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(int(orig_fps / sample_fps), 1)

    frames = []
    idx = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            fr = resize_max(fr)
            t = idx / orig_fps
            frames.append(FrameRecord(idx, t, fr))
        idx += 1

    cap.release()
    return frames


# -----------------------------
# Keyframe scoring
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_scores(frames: List[FrameRecord], metric: str):
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
        else:  # MSE
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
# UI
# -----------------------------
st.title("🎬 Keyframe Extractor & Visual Change Explorer")

uploaded = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])
if not uploaded:
    st.stop()

sample_fps = st.slider("Sampling FPS", 1.0, 8.0, 2.0)
metric = st.selectbox("Change metric", ["SSIM (1-SSIM)", "Color Histogram (Bhattacharyya)", "MSE"])
k = st.slider("Number of keyframes", 5, 30, 12)
min_gap_sec = st.slider("Minimum gap between selected frames (sec)", 0.0, 5.0, 0.5)


frames = decode_video(uploaded.getvalue(), sample_fps)
scores = compute_scores(frames, metric)
keyframes = pick_keyframes(frames, scores, k, min_gap_sec, sample_fps)

st.subheader("Selected Keyframes")
cols = st.columns(min(len(keyframes), 6))
for i, fr in enumerate(keyframes):
    cols[i % len(cols)].image(fr.rgb, caption=f"t={fr.time_s:.2f}s (#{fr.index})", use_column_width=True)


# -----------------------------
# Frame comparison + SSIM heatmap + NARRATIVE
# -----------------------------
st.markdown("---")
st.subheader("🔍 Compare Frames")

names = [f"t={fr.time_s:.2f}s (#{fr.index})" for fr in keyframes]
iA = st.selectbox("Frame A", range(len(keyframes)), format_func=lambda i: names[i])
iB = st.selectbox("Frame B", range(len(keyframes)), format_func=lambda i: names[i])

A = keyframes[iA].rgb
B = keyframes[iB].rgb
h = min(A.shape[0], B.shape[0])
w = min(A.shape[1], B.shape[1])
A = cv2.resize(A, (w,h))
B = cv2.resize(B, (w,h))

ca, cb = st.columns(2)
ca.image(A, caption="A", use_column_width=True)
cb.image(B, caption="B", use_column_width=True)

alpha = st.slider("Blend Frames", 0.0, 1.0, 0.5)
st.image(blend(A,B,alpha), use_column_width=True, caption=f"Blend α={alpha:.2f}")

heat, ssim_score = ssim_diff_map(A,B)
st.image(heat, caption=f"SSIM Score = {ssim_score:.4f} (higher = more similar)")

# Narrative explanation
if ssim_score > 0.85:
    narrative = "These frames are nearly identical — this is the same moment, visually and structurally."
elif ssim_score > 0.6:
    narrative = "There is moderate change — likely motion, camera movement, or subtle compositional shift."
elif ssim_score > 0.35:
    narrative = "There is a meaningful visual change — likely a new pose, camera angle, or scene emphasis."
else:
    narrative = "This is a **major scene change** — likely a new shot or location."

st.markdown(f"**Interpretation:** {narrative}")
