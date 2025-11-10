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

# =============================
# Config & Page
# =============================
st.set_page_config(page_title="Keyframe Extractor & Visual Change Explorer", layout="wide")
st.title("Keyframe Extractor & Visual Change Explorer")

# =============================
# Data structures & helpers
# =============================
@dataclass
class FrameRecord:
    index: int
    time_s: float
    image_rgb: np.ndarray  # RGB, already resized for display/processing


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
    # simple background box
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


# =============================
# Video IO (upload or URL via yt-dlp)
# =============================
@st.cache_data(show_spinner=False)
def _download_video_bytes(url: str, quality: str, no_audio: bool) -> Optional[bytes]:
    """Download a single MP4/H264-ish stream via yt-dlp with format fallbacks.
    Returns bytes or None on error. Cached by URL+options.
    """
    if not YTDLP_AVAILABLE:
        return None

    # Prefer H.264/avc1 or VP9 to avoid AV1 hardware issues; allow medium/low for speed.
    # Build format string
    # Base video selector prioritizing non-AV1:
    base_v = "bv*[vcodec~='^(avc1|h264|x264|vp9)']"
    if no_audio:
        fmt = f"{base_v}/bv*"
    else:
        fmt = f"{base_v}+ba/{base_v}/b"

    # Quality hint via format sorting
    # 'quality' affects height cap
    height_cap = {
        "Low": 480,
        "Medium": 720,
        "Auto": 1080,
        "Lossless": 4320,  # not truly lossless but do not downcap
    }.get(quality, 1080)

    ydl_opts = {
        'format': fmt,
        'outtmpl': '-',
        'quiet': True,
        'noprogress': True,
        'merge_output_format': 'mp4',
        'retries': 3,
    }

    # We'll actually download to a temp file to avoid large memory spikes
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp_path = tmp.name
    try:
        def _progress(d):
            # We do not update Streamlit here (cache func), just ignore
            pass

        # Use direct file output via 'outtmpl'
        ydl_opts_file = ydl_opts.copy()
        ydl_opts_file['outtmpl'] = tmp_path
        # Use height cap by postprocessor args? Easiest: rely on extractor sorting + recode fallback.
        with yt_dlp.YoutubeDL(ydl_opts_file) as ydl:
            ydl.download([url])
        with open(tmp_path, 'rb') as f:
            return f.read()
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _write_temp_file(data: bytes, suffix: str = '.mp4') -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def _transcode_to_h264(input_path: str) -> Optional[str]:
    """Best-effort ffmpeg fallback to H.264 if OpenCV struggles (e.g., AV1). Returns new path or None."""
    try:
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', input_path,
            '-an',  # drop audio for speed/stability
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            out_path
        ]
        subprocess.run(cmd, check=True)
        return out_path
    except Exception:
        return None


# =============================
# Decoding & sampling
# =============================
@st.cache_data(show_spinner=False)
def decode_video_bytes(file_bytes: bytes, sample_fps: float = 2.0, max_dim: Tuple[int, int] = (960, 540)) -> Tuple[List[FrameRecord], float]:
    """Decode with OpenCV (FFmpeg under the hood). Returns sampled FrameRecord list and original FPS."""
    if not file_bytes:
        return [], 0.0
    in_path = _write_temp_file(file_bytes)

    def _decode(path: str) -> Tuple[List[FrameRecord], float]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return [], 0.0
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames == 0:
            cap.release()
            return [], float(orig_fps)

        # Compute stride for sampling to limit memory
        stride = max(int(round(orig_fps / max(sample_fps, 0.1))), 1)

        frames: List[FrameRecord] = []
        idx = 0
        out_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frame_small = resize_max(frame_bgr, max_w=max_dim[0], max_h=max_dim[1])
                frame_rgb = bgr2rgb(frame_small)
                t = idx / max(orig_fps, 1e-6)
                frames.append(FrameRecord(index=idx, time_s=t, image_rgb=frame_rgb))
                out_idx += 1
            idx += 1
        cap.release()
        return frames, float(orig_fps)

    frames, fps = _decode(in_path)
    if len(frames) < 2:
        # Try ffmpeg transcode fallback (e.g., AV1 -> H.264)
        alt = _transcode_to_h264(in_path)
        if alt:
            frames, fps = _decode(alt)
            try:
                os.remove(alt)
            except Exception:
                pass
    try:
        os.remove(in_path)
    except Exception:
        pass
    return frames, fps


# =============================
# Change metrics (cache-safe)
# =============================
@st.cache_data(show_spinner=False)
def compute_consecutive_scores(_frames: List[FrameRecord], metric: str) -> np.ndarray:
    """Higher score = more change between consecutive sampled frames."""
    frames = _frames  # alias
    if len(frames) < 2:
        return np.array([])

    scores = []
    for i in range(1, len(frames)):
        a = frames[i - 1].image_rgb
        b = frames[i].image_rgb
        if metric == 'SSIM (1-SSIM)':
            g1, g2 = rgb2gray(a), rgb2gray(b)
            val = 1.0 - float(ssim(g1, g2, data_range=255))
        elif metric == 'Color Histogram (Bhattacharyya)':
            ha = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
            hb = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
            bins = (8, 8, 8)
            ranges = [0, 180, 0, 256, 0, 256]
            ha_hist = cv2.calcHist([ha], [0, 1, 2], None, bins, ranges)
            hb_hist = cv2.calcHist([hb], [0, 1, 2], None, bins, ranges)
            cv2.normalize(ha_hist, ha_hist)
            cv2.normalize(hb_hist, hb_hist)
            val = float(cv2.compareHist(ha_hist, hb_hist, cv2.HISTCMP_BHATTACHARYYA))
        elif metric == 'MSE':
            diff = (a.astype(np.float32) - b.astype(np.float32))
            val = float(np.mean(diff * diff)) / (255.0 * 255.0)
        else:
            val = 0.0
        scores.append(val)
    return np.array(scores, dtype=np.float32)


def _nms(scores: np.ndarray, desired_k: int, min_gap: int) -> List[int]:
    if len(scores) == 0:
        return []
    order = list(np.argsort(scores)[::-1])
    selected: List[int] = []
    for idx in order:
        if len(selected) >= desired_k:
            break
        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(int(idx))
    selected.sort()
    return selected


@st.cache_data(show_spinner=False)
def select_keyframes(_frames: List[FrameRecord], metric: str, k: int, min_gap_seconds: float, sample_fps: float) -> List[FrameRecord]:
    frames = _frames
    if len(frames) == 0:
        return []
    scores = compute_consecutive_scores(frames, metric)
    min_gap = max(int(round(min_gap_seconds * sample_fps)), 1)
    idxs = _nms(scores, desired_k=k, min_gap=min_gap)
    chosen = [frames[i] for i in idxs]
    if frames and (len(chosen) == 0 or chosen[0].index != frames[0].index):
        chosen = [frames[0]] + chosen
    if frames and (len(chosen) == 0 or chosen[-1].index != frames[-1].index):
        chosen = chosen + [frames[-1]]
    if len(chosen) > k:
        def score_for(fr: FrameRecord):
            pos = next((i for i, f in enumerate(frames) if f.index == fr.index), None)
            if pos is None or pos == 0 or pos - 1 >= len(scores):
                return -1.0
            return float(scores[pos - 1])
        chosen = sorted(sorted(chosen, key=score_for, reverse=True)[:k], key=lambda fr: fr.index)
    elif len(chosen) < k:
        missing = k - len(chosen)
        remaining = [f for f in frames if f.index not in {c.index for c in chosen}]
        if remaining:
            step = max(len(remaining) // (missing + 1), 1)
            fillers = remaining[::step][:missing]
            chosen = sorted(chosen + fillers, key=lambda fr: fr.index)
    return chosen


# =============================
# Sidebar UI
# =============================
with st.sidebar:
    st.header("Inputs")
    source_type = st.selectbox("Video source", ["Upload file", "URL (yt-dlp)"])

    if source_type == "Upload file":
        uploaded = st.file_uploader("Video file", type=["mp4", "mov", "m4v", "avi", "mkv", "webm"], key="uploader_primary")
        url = None
    else:
        url = st.text_input("Paste a video URL (YouTube, etc.)")
        if not YTDLP_AVAILABLE:
            st.info("yt-dlp is not installed in this runtime. URL downloads will be disabled.")
        uploaded = None

    quality = st.selectbox("Download quality (URL only)", ["Auto", "Medium", "Low", "Lossless"], index=0)
    no_audio = st.checkbox("Download without audio (faster, smaller)", value=True)

    sample_fps = st.slider("Sampling rate (frames/sec)", 0.5, 8.0, 2.0, 0.5, help="Higher = more precise but slower")
    metric = st.selectbox(
        "Change metric",
        ["SSIM (1-SSIM)", "Color Histogram (Bhattacharyya)", "MSE"],
        index=0,
    )
    k = st.slider("How many keyframes?", 5, 30, 12, 1)
    min_gap_sec = st.slider("Minimum gap between picks (sec)", 0.0, 5.0, 0.5, 0.5)

    st.markdown("---")
    st.subheader("How to choose a metric")
    st.markdown(
        """
**SSIM (1-SSIM)** — Emphasizes *structural* differences (edges, textures, shapes). It tends to ignore small lighting flicker and camera noise, so it’s great for detecting cuts, new shots, objects entering/leaving, or composition changes. A higher score here means a bigger change.

**Color Histogram (Bhattacharyya)** — Focuses on overall *color palette* shifts. It will highlight transitions like day→night, indoor→outdoor, or strong grading changes, even if the layout barely changes. Useful when color cues matter more than geometry.

**MSE (pixel error)** — Straight pixel-wise difference. Very sensitive to any motion, noise, or compression artifacts. It can catch subtle changes but also overreact to handheld shake; use when you want maximum sensitivity and are okay with noise.
        """
    )


# =============================
# Acquire video bytes
# =============================
video_bytes: Optional[bytes] = None
progress_txt = st.empty()
progress_bar = st.progress(0)

if 'last_download_key' not in st.session_state:
    st.session_state.last_download_key = None

if uploaded is None and not url:
    progress_txt.write("Awaiting input…")
    progress_bar.progress(0)
    st.stop()

if uploaded is not None:
    progress_txt.write("Reading uploaded file… 100%")
    video_bytes = uploaded.getvalue()
    progress_bar.progress(100)
else:
    if not YTDLP_AVAILABLE:
        st.error("URL mode requires yt-dlp. Install it in your environment.")
        st.stop()
    cache_key = (url or "", quality, no_audio)
    if st.session_state.last_download_key != cache_key:
        progress_txt.write("Downloading with yt-dlp… 0%")
        # Use a simple staged progress since yt-dlp progress hooks are inside cache
        video_bytes = _download_video_bytes(url, quality, no_audio)
        if video_bytes is None:
            st.error("Download failed. Try a different URL or lower quality.")
            st.stop()
        st.session_state.last_download_key = cache_key
        progress_txt.write("Downloading with yt-dlp… 100%")
        progress_bar.progress(100)
    else:
        progress_txt.write("Using cached download… 100%")
        progress_bar.progress(100)
        video_bytes = _download_video_bytes(url, quality, no_audio)

# =============================
# Decode & sample
# =============================
with st.spinner("Decoding and sampling frames…"):
    frames, orig_fps = decode_video_bytes(video_bytes, sample_fps=sample_fps)

if len(frames) < 2:
    st.error("Not enough frames could be decoded. If the source uses AV1, try URL mode at Low/Medium quality or let the app transcode via ffmpeg.")
    st.stop()

# =============================
# Select keyframes
# =============================
with st.spinner("Scoring changes and selecting keyframes…"):
    keyframes = select_keyframes(frames, metric=metric, k=k, min_gap_seconds=min_gap_sec, sample_fps=sample_fps)

# =============================
# Timeline / preview
# =============================
st.subheader("Selected Keyframes")

thumb_cols = st.columns(min(6, max(2, len(keyframes))))
for i, fr in enumerate(keyframes):
    with thumb_cols[i % len(thumb_cols)]:
        stamped = overlay_timestamp(fr.image_rgb, fr.time_s)
        st.image(stamped, caption=f"t={fr.time_s:.2f}s (#{fr.index})", use_column_width=True)

st.markdown("---")

st.subheader("Compare Frames")
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

    # Align sizes
    h = min(A.shape[0], B.shape[0])
    w = min(A.shape[1], B.shape[1])
    A_res = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    B_res = cv2.resize(B, (w, h), interpolation=cv2.INTER_AREA)

    colL, colR = st.columns(2)
    with colL:
        st.caption("Left (A)")
        st.image(overlay_timestamp(A_res, keyframes[iA].time_s), use_column_width=True)
    with colR:
        st.caption("Right (B)")
        st.image(overlay_timestamp(B_res, keyframes[iB].time_s), use_column_width=True)

    st.markdown("Crossfade blend")
    alpha = st.slider("Blend toward A", 0.0, 1.0, 0.5, 0.05)
    def _blend(a, b, alpha):
        af = a.astype(np.float32)
        bf = b.astype(np.float32)
        mix = af * alpha + bf * (1.0 - alpha)
        return np.clip(mix, 0, 255).astype(np.uint8)
    st.image(_blend(A_res, B_res, alpha), use_column_width=True, caption=f"Blend alpha={alpha:.2f}")

    # SSIM score only (heatmap removed by request)
    ssim_score = float(ssim(rgb2gray(A_res), rgb2gray(B_res), data_range=255))
    if ssim_score > 0.85:
        narrative = "Frames are nearly identical (minimal structural change)."
    elif ssim_score > 0.6:
        narrative = "Moderate structural change—likely motion or composition tweaks."
    elif ssim_score > 0.35:
        narrative = "Clear visual change—objects/scene layout likely shifted."
    else:
        narrative = "Major scene/shot transition with strong structural differences."
    st.markdown(f"SSIM score: **{ssim_score:.4f}** — {narrative}")

st.markdown("---")

st.caption(
    "Note: Decoding uses OpenCV/FFmpeg. AV1 or exotic codecs may require the built-in ffmpeg fallback (transcodes to H.264)."
)
