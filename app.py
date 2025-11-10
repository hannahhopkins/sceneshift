```python
import streamlit as st
import ffmpeg
import numpy as np
import cv2
from PIL import Image
import tempfile
import math
import shutil

# ------------------------------------------------------------
# Optional GPU Support for Frame Difference Computation
# ------------------------------------------------------------
def compute_diff(prev, curr):
    """Compute mean absolute difference; GPU if available."""
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        prev_gpu = cv2.cuda_GpuMat()
        curr_gpu = cv2.cuda_GpuMat()
        prev_gpu.upload(prev)
        curr_gpu.upload(curr)
        diff_gpu = cv2.cuda.absdiff(prev_gpu, curr_gpu)
        diff = cv2.cuda.mean(diff_gpu)[0]
        return diff
    else:
        return np.mean(np.abs(curr.astype(float) - prev.astype(float)))

# ------------------------------------------------------------
# Extract frames with ffmpeg (fast & no OpenCV dependency for decode)
# ------------------------------------------------------------
def extract_frames_ffmpeg(video_path, fps=4):
    """
    Decode frames uniformly using ffmpeg at ~fps.
    Returns list of real-color frames (H, W, 3).
    """
    temp_dir = tempfile.mkdtemp()
    frame_pattern = f"{temp_dir}/frame_%05d.png"

    (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=fps)
        .output(frame_pattern, format="image2", vframes=None, loglevel="error")
        .run()
    )

    # Load frames into memory (in correct order)
    frames = []
    import os
    for fname in sorted(os.listdir(temp_dir)):
        frame = cv2.imread(f"{temp_dir}/{fname}")
        if frame is not None:
            frames.append(frame)

    shutil.rmtree(temp_dir)
    return frames

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("Keyframe Extractor: Most Significant Visual Change")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
num_frames = st.slider("How many significant frames to return?", 6, 30, 12)
fps_sampling = st.slider("Sampling Rate (FPS for analysis)", 1, 12, 4)

if uploaded_video is not None:
    # Store upload to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(uploaded_video.read())
    tmp_file.close()

    st.write("Extracting frames…")
    frames = extract_frames_ffmpeg(tmp_file.name, fps=fps_sampling)

    if len(frames) < 2:
        st.error("Not enough extractable frames.")
        st.stop()

    st.write(f"Analyzing {len(frames)} frames for visual change…")

    # Preprocess frames for comparison
    small_frames = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 64))
        for f in frames
    ]

    # Calculate visual difference scores
    diffs = []
    for i in range(1, len(small_frames)):
        score = compute_diff(small_frames[i-1], small_frames[i])
        diffs.append((score, i, frames[i]))

    # Sort by change intensity
    diffs.sort(reverse=True, key=lambda x: x[0])
    selected = diffs[:num_frames]
    selected.sort(key=lambda x: x[1])  # Chronological

    st.subheader("Keyframes:")

    # Display in grid layout
    cols = st.columns(4)
    for idx, (score, frame_index, frame) in enumerate(selected):
        col = cols[idx % 4]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        col.image(Image.fromarray(rgb), caption=f"Frame {frame_index} | Score {score:.2f}")

    st.success("Done.")
