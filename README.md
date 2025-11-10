# Keyframe Extractor & Visual Change Explorer

Upload a short video, pick a change metric (SSIM / Color Histogram / MSE),
and extract the most significant moments of visual change. Then compare frames
side-by-side, crossfade between them, or view an SSIM difference heatmap.

## Run locally
python -m pip install -r requirements.txt
streamlit run app.py

## Deploy to Streamlit Cloud
- Push to GitHub with app.py at repo root (or set working dir in the app config).
- Set Python version to 3.10–3.12. No extra system packages required.
