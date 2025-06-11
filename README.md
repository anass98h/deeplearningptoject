# Real-time Pose Estimation and Analysis System

This project showcases a production‑ready pipeline for extracting, analysing and scoring human motion using state‑of‑the‑art deep learning models.  The backend coordinates multiple neural networks written in TensorFlow and scikit‑learn, while a modern Next.js frontend streams pose data directly from the browser.

At its core the system performs real‑time pose capture, converts 2D keypoints to 3D skeletons and evaluates exercise quality with a CNN‑LSTM scoring network.  The processing stages are designed for research and extend easily to new movements or additional quality checks.

## Pipeline Overview

```
Video ▶ MoveNet ▶ Ugly‑2D Gate ▶ 2D→Kinect Conversion ▶ Depth Prediction ▶
Frame Trimming ▶ Bad‑3D Gate ▶ Exercise Scoring
```

Each step uses a dedicated model trained on curated motion‑capture data.  Detailed descriptions of the algorithms and diagnostics can be found in [backend/documentation.md](backend/documenation.md).

## Key Highlights

- **Live browser capture** via TensorFlow.js PoseNet
- **Hybrid CNN analysis** for video quality checking
- **Kinect‑style 3D skeletons** predicted from single‑view footage
- **Automatic exercise segmentation** using recurrent networks
- **Granular scoring** with a deep CNN‑LSTM model

## Repository Layout

```
backend/        # FastAPI applications and API documentation
frontend/       # Main Next.js interface for realtime capture
frontend_alt/   # Experimental alternative interface
ML/             # Training notebooks and data utilities
models/         # Example models and scalers
saved_models/   # Additional trained model checkpoints
```

## Quick Start

### Backend (Python 3.10+)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload            # full pipeline API
# or
uvicorn app.main2:app --reload           # lightweight model serving API
```

Run tests with:

```bash
pytest
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 to see pose predictions streamed from your webcam.  The `frontend_alt/` directory contains an experimental interface built on the same API.

## Deep Learning Highlights

- **Pose Conversion** – A pair of dense networks translate MoveNet output to Kinect‑style coordinates and add depth via a regression model.
- **Quality Gates** – CNN‑based detectors provide early stopping with detailed diagnostics for low‑confidence or poorly executed movements.
- **Exercise Scoring** – A convolutional‑recurrent network analyses the full pose sequence and returns a continuous score from 0 (perfect) to 4 (poor).

The models are saved under `models/` and loaded automatically by the backend's `ModelLoader` utility.

## License

This project is provided as‑is for research and educational use.
