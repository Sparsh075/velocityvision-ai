# VelocityVision (High-Accuracy Upgrade)

VelocityVision is now a robust sports analytics backend with homography-based metric calculations, ByteTrack-style tracking, stabilized possession/pass analytics, and next-pass prediction.

## New Backend Modules

- `backend/calibrate.py`
- `backend/detection.py`
- `backend/tracking.py`
- `backend/ball_tracking.py`
- `backend/team_id.py`
- `backend/filters.py`
- `backend/analytics_speed.py`
- `backend/analytics_pass.py`
- `backend/prediction.py`
- `backend/metrics.py`
- `backend/video_renderer.py`
- `backend/main.py`

## Key Accuracy Improvements

- Homography image->pitch mapping (105m x 68m) for metric speed/pass analytics
- ByteTrack-inspired tracking engine with high/low confidence association
- Smoothed metric trajectories + outlier-capped speed estimation
- Temporal ball linking and short-gap interpolation
- Possession stability window and pass cooldown logic
- Team-ID via jersey-color K-Means for cleaner pass/team metrics
- Optional next-pass prediction with interpretable geometry features

## Install

```bash
cd sports-analytics
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

### Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
python -m streamlit run frontend/app.py
```

## Homography Calibration (Manual)

You can provide manual calibration points through a JSON file and set:

```bash
# PowerShell
$env:VELOCITYVISION_CALIBRATION_JSON="C:\path\to\calibration.json"
```

JSON format:

```json
{
  "img_pts": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "field_pts": [[0,0], [105,0], [105,68], [0,68]]
}
```

If not provided, fallback homography is used and metrics are marked `metrics_approximate: true`.

## Example Config

See: `backend/config.example.json`

## Artifacts

Generated in `outputs/`:

- `{job_id}_annotated.mp4`
- `{job_id}_metrics.json`
- `{job_id}_pass_network.png`
- `{job_id}_heatmap.png`
- `{job_id}_possession_timeline.png`
- `{job_id}_player_rankings.csv`
