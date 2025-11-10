import tempfile
import io
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import streamlit as st

# Core CV + metrics
import cv2
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# Helpers & data structures
# -----------------------------
@dataclass
class FrameRecord:
    index: int
    time_s: float
    image_bgr: np.ndarray  # stored as BGR (OpenCV)
    image_rgb: np.ndarray  # cached RGB for display


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


# -----------------------------
# Video decoding
# -----------------------------
@st.cache_data(show_spinner=False)
def decode_video(file_bytes: bytes, sample_fps: float = 2.0, max_dim: Tuple[int, int] = (960, 540)) -> Tuple[List[FrameRecord], float]:
    """Decode the uploaded video using OpenCV and sample frames at ~sample_fps."""
    
    # Write bytes to a temporary file for cv2.VideoCapture
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video. Ensure FFmpeg/codec support is available.")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(orig_fps / max(sample_fps, 0.1))), 1)

    frames: List[FrameRecord] = []
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if idx % stride == 0:
            frame_bgr_small = resize_max(frame_bgr, max_w=max_dim[0], max_h=max_dim[1])
            frame_rgb_small = bgr2rgb(frame_bgr_small)
            t = idx / max(orig_fps, 1e-6)
            frames.append(FrameRecord(index=idx, time_s=t, image_bgr=frame_bgr_small, image_rgb=frame_rgb_small))

        idx += 1

    cap.release()
    return frames, float(orig_fps)



# -----------------------------
# Difference metrics
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_consecutive_scores(frames: List[FrameRecord], metric: str) -> np.ndarray:
    """Compute a score of change between consecutive sampled frames.
    Supported metrics: 'SSIM (1-SSIM)', 'Color Histogram (Bhattacharyya)', 'MSE'.
    Higher = more change.
    """
    if len(frames) < 2:
        return np.array([])

    scores = []
    for i in range(1, len(frames)):
        a = frames[i - 1].image_rgb
        b = frames[i].image_rgb

        if metric == 'SSIM (1-SSIM)':
            g1, g2 = rgb2gray(a), rgb2gray(b)
            val = 1.0 - ssim(g1, g2, data_range=255)
        elif metric == 'Color Histogram (Bhattacharyya)':
            # HSV 3D histogram comparison
            ha = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
            hb = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
            bins = (8, 8, 8)
            ranges = [0, 180, 0, 256, 0, 256]
            ha_hist = cv2.calcHist([ha], [0, 1, 2], None, bins, ranges)
            hb_hist = cv2.calcHist([hb], [0, 1, 2], None, bins, ranges)
            cv2.normalize(ha_hist, ha_hist)
            cv2.normalize(hb_hist, hb_hist)
            val = cv2.compareHist(ha_hist, hb_hist, cv2.HISTCMP_BHATTACHARYYA)
        elif metric == 'MSE':
            diff = (a.astype(np.float32) - b.astype(np.float32))
            val = float(np.mean(diff * diff)) / (255.0 * 255.0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        scores.append(float(val))

    return np.array(scores)


def non_maximum_suppression(scores: np.ndarray, desired_k: int, min_gap: int) -> List[int]:
    """Greedy pick of top indices with a minimum gap constraint.
    Returns selected indices referring to the *second* frame of each pair (i.e., frames[i]).
    """
    if len(scores) == 0:
        return []

    order = list(np.argsort(scores)[::-1])  # descending by score
    selected: List[int] = []

    for idx in order:
        if len(selected) >= desired_k:
            break
        # Enforce gap in index-space
        ok = True
        for s in selected:
            if abs(idx - s) < min_gap:
                ok = False
                break
        if ok:
            selected.append(int(idx))

    selected.sort()
    return selected


@st.cache_data(show_spinner=False)
def select_keyframes(frames: List[FrameRecord], metric: str, k: int, min_gap_seconds: float, sample_fps: float) -> List[FrameRecord]:
    if len(frames) == 0:
        return []

    scores = compute_consecutive_scores(frames, metric)
    # Convert min gap in seconds to sampled-frame units. Each sampled frame ~ 1/sample_fps seconds.
    min_gap = max(int(round(min_gap_seconds * sample_fps)), 1)
    idxs = non_maximum_suppression(scores, desired_k=k, min_gap=min_gap)

    # Map score index to frame index; score i compares frames[i-1] & frames[i], and we return frames[i].
    chosen = [frames[i] for i in idxs]

    # Always consider including the first / last sampled frame for coverage
    if frames and (len(chosen) == 0 or chosen[0].index != frames[0].index):
        chosen = [frames[0]] + chosen
    if frames and (len(chosen) == 0 or chosen[-1].index != frames[-1].index):
        chosen = chosen + [frames[-1]]

    # Trim or pad to k if needed (best-effort)
    if len(chosen) > k:
        # Keep the highest-score subset (guaranteeing first/last where possible)
        # Compute contribution scores for chosen (use original scores, map by sampled index) 
        def score_for_frame(fr: FrameRecord):
            # score index equals frame position i in sampled sequence
            pos = next((i for i, f in enumerate(frames) if f.index == fr.index), None)
            if pos is None or pos == 0 or pos - 1 >= len(scores):
                return 0.0
            return float(scores[pos - 1])
        chosen_sorted = sorted(chosen, key=score_for_frame, reverse=True)
        chosen = sorted(chosen_sorted[:k], key=lambda fr: fr.index)
    elif len(chosen) < k:
        # Fill from remaining frames at regular intervals
        missing = k - len(chosen)
        remaining = [f for f in frames if f.index not in {c.index for c in chosen}]
        if remaining:
            step = max(len(remaining) // (missing + 1), 1)
            fillers = remaining[::step][:missing]
            chosen = sorted(chosen + fillers, key=lambda fr: fr.index)

    return chosen


# -----------------------------
# Visualization utilities
# -----------------------------
def ssim_diff_map(imgA_rgb: np.ndarray, imgB_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    g1 = rgb2gray(imgA_rgb)
    g2 = rgb2gray(imgB_rgb)
    score, diff = ssim(g1, g2, data_range=255, full=True)
    # diff is similarity in [0,1]; we convert to difference heatmap
    diff_inv = (1.0 - diff)
    heat = (np.clip(diff_inv, 0.0, 1.0) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return heat_color, float(score)


def blend_images(imgA_rgb: np.ndarray, imgB_rgb: np.ndarray, alpha: float) -> np.ndarray:
    a = imgA_rgb.astype(np.float32)
    b = imgB_rgb.astype(np.float32)
    mixed = a * alpha + b * (1.0 - alpha)
    return np.clip(mixed, 0, 255).astype(np.uint8)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Keyframe Extractor & Visual Change Explorer", layout="wide")

st.title("🎬 Keyframe Extractor & Visual Change Explorer")

st.markdown(
    """
Upload a short video and extract the **most significant moments of visual change**.
Then compare frames side‑by‑side or inspect an **SSIM difference heatmap**.

**How it works** (quick options):
- **Metric for change**
  - *SSIM (1‑SSIM)* – Measures structural change. Great for scene/layout shifts.
  - *Color Histogram* – Captures large color palette shifts.
  - *MSE* – Simple pixel difference; sensitive to noise/camera shake.
- **SSIM Diff Map** – A color‑coded heatmap where warmer colors highlight stronger change.
    """
)

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "m4v", "avi", "mkv", "webm"]) 

    sample_fps = st.slider("Sampling rate (frames/sec)", 0.5, 8.0, 2.0, 0.5, help="Higher = more precise but slower")

    metric = st.selectbox(
        "Change metric",
        ["SSIM (1-SSIM)", "Color Histogram (Bhattacharyya)", "MSE"],
        index=0,
        help=(
            "SSIM emphasizes structural changes; Histogram captures color palette shifts; "
            "MSE is raw pixel difference."
        ),
    )

    k = st.slider("How many keyframes?", 5, 30, 12, 1, help="10–15 is a good starting range")

    min_gap_sec = st.slider(
        "Minimum gap between picks (sec)", 0.0, 5.0, 0.5, 0.5,
        help="Avoids picking near-duplicate frames"
    )

    show_ssim_toggle = st.checkbox("Enable SSIM difference heatmap viewer", value=True)

    st.markdown("""
**Tips**
- If extraction seems too similar, increase the *gap* or try *Histogram*.
- If reflections/noise dominate, prefer *SSIM*.
    """)

if uploaded is None:
    st.info("Upload a video to begin.")
    st.stop()

# Decode & sample frames
with st.spinner("Decoding video & sampling frames..."):
    frames, orig_fps = decode_video(uploaded.getvalue(), sample_fps=sample_fps)

if not frames:
    st.error("No frames decoded. Try a different file.")
    st.stop()

# Select keyframes
with st.spinner("Scoring changes & selecting keyframes..."):
    keyframes = select_keyframes(frames, metric=metric, k=k, min_gap_seconds=min_gap_sec, sample_fps=sample_fps)

st.subheader("Selected Keyframes")

# Small gallery of thumbnails with timestamps
thumb_cols = st.columns(min(6, max(2, len(keyframes))))
for i, fr in enumerate(keyframes):
    with thumb_cols[i % len(thumb_cols)]:
        st.image(fr.image_rgb, caption=f"t={fr.time_s:.2f}s (#{fr.index})", use_column_width=True)

# -----------------------------
# Interactive comparison
# -----------------------------
st.markdown("---")

st.subheader("🔍 Compare Frames")

if len(keyframes) < 2:
    st.info("Need at least two keyframes to compare.")
else:
    labels = [f"t={fr.time_s:.2f}s (#{fr.index})" for fr in keyframes]
    c1, c2 = st.columns(2)
    with c1:
        iA = st.selectbox("Left frame", list(range(len(keyframes))), format_func=lambda i: labels[i], index=0)
    with c2:
        iB = st.selectbox("Right frame", list(range(len(keyframes))), format_func=lambda i: labels[i], index=min(1, len(keyframes)-1))

    A = keyframes[iA].image_rgb
    B = keyframes[iB].image_rgb

    # Make the images the same size for fair comparison
    h = min(A.shape[0], B.shape[0])
    w = min(A.shape[1], B.shape[1])
    A_res = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    B_res = cv2.resize(B, (w, h), interpolation=cv2.INTER_AREA)

    colL, colR = st.columns(2)
    with colL:
        st.caption("Left (A)")
        st.image(A_res, use_column_width=True)
    with colR:
        st.caption("Right (B)")
        st.image(B_res, use_column_width=True)

    st.markdown("**Swipe-style blend** (drag the slider to crossfade)")
    alpha = st.slider("Blend toward A", 0.0, 1.0, 0.5, 0.05)
    blend = blend_images(A_res, B_res, alpha)
    st.image(blend, use_column_width=True, caption=f"Blend α={alpha:.2f}")

    if show_ssim_toggle:
        st.markdown("**SSIM Difference Heatmap** (warm colors = larger change)")
        heat, ssim_score = ssim_diff_map(A_res, B_res)
        st.image(heat, use_column_width=True, caption=f"SSIM score = {ssim_score:.4f} (higher = more similar)")

# -----------------------------
# Explanations (short, per-toggle)
# -----------------------------
with st.expander("What do these options mean?", expanded=False):
    st.markdown(
        """
**SSIM (1‑SSIM)** – SSIM compares structure/texture; we take *(1‑SSIM)* so higher means more change.
Good for detecting edits, cuts, or layout shifts while ignoring mild lighting noise.

**Color Histogram (Bhattacharyya)** – Compares the distribution of colors. Higher values indicate a
larger shift in overall palette (e.g., moving from a blue scene to a warm indoor shot).

**MSE** – Mean Squared Error of pixels. Straightforward but can be overly sensitive to small motion or noise.

**SSIM Difference Heatmap** – Visualizes *where* changes occur; hotter colors highlight stronger differences.
        """
    )

st.markdown("---")
small = "**Note:** Frame decoding uses OpenCV, which on most platforms leverages FFmpeg codecs under the hood."
st.caption(small)
