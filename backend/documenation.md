# Video Processing Pipeline Documentation

## Overview

This is a comprehensive video processing pipeline that analyzes exercise videos to extract pose data, assess video quality, evaluate exercise form, and provide detailed scoring. The pipeline includes multiple quality gates and produces 3D skeleton data suitable for visualization.

## Pipeline Architecture

```
Video Upload → MoveNet Extraction → Ugly 2D Check → Kinect 2D Conversion → Kinect 3D Prediction → Frame Trimming → Bad 3D Check → Exercise Scoring
                                         ↓                                                                              ↓
                                    STOP if ugly                                                                 STOP if bad form
                                    (return error)                                                               (return error)
```

## Pipeline Stages

### 1. Video Upload & MoveNet Extraction

- **Input**: Video file (various formats supported)
- **Process**: Extract 2D pose keypoints using Google's MoveNet model
- **Output**: CSV with 13 keypoints (x, y coordinates) + FrameNo
- **File**: `movenet_extraction.py`

**Keypoints extracted:**

- head, left_shoulder, right_shoulder, left_elbow, right_elbow
- left_hand, right_hand, left_hip, right_hip, left_knee, right_knee
- left_foot, right_foot

### 2. Ugly 2D Quality Check ⚠️

- **Input**: Original video + MoveNet pose data
- **Process**: Hybrid model analyzes video frames + pose data for quality assessment
- **Model**: `trained_hybrid_model.keras` (expects 40 features)
- **Output**: Goodness score + Confidence score
- **Action**: **STOPS pipeline** if quality below thresholds
- **File**: `ugly_2d_detector.py`

**Quality thresholds:**

- Goodness threshold: 0.5 (minimum acceptable)
- Confidence threshold: 0.3 (minimum acceptable)

**Note**: If MoveNet data lacks confidence scores (26 features), the system automatically re-extracts with confidence scores (40 features) for this step, then strips them for pipeline compatibility.

### 3. Kinect 2D Format Conversion

- **Input**: MoveNet pose data (26 features)
- **Process**: Neural network converts MoveNet format to Kinect 2D format
- **Model**: `xy_to_xy_best_2.keras` + scalers
- **Output**: Kinect-compatible 2D pose data
- **File**: `posenet_to_kinect2d.py`

### 4. Kinect 3D Depth Prediction

- **Input**: Kinect 2D pose data
- **Process**: Add z-coordinates using depth prediction model
- **Model**: `kinect_depth_model.keras`
- **Output**: Full 3D pose data (39 features: 13 joints × 3 coordinates)
- **File**: `kinect2d_to_kinect3d.py`

### 5. Frame Trimming (Exercise Segmentation)

- **Input**: Kinect 3D pose data
- **Process**: Identify the relevant exercise segment from the full video
- **Model**: `kinect_cutting_model.keras` + scaler
- **Output**: Trimmed 3D pose data containing only the exercise movement
- **File**: `frame_trimmer.py`

### 6. Bad 3D Exercise Form Check ⚠️

- **Input**: Trimmed 3D pose data
- **Process**: Analyze exercise form quality using dense neural network
- **Model**: `kinect_dense_good_vs_bad_model.keras` + scaler
- **Expected Input**: (1, 10, 39) - 10 frames with 39 features
- **Output**: Quality score (1.0 = good, 0.0 = bad)
- **Action**: **STOPS pipeline** if form is poor (score < threshold)
- **File**: `bad_3d_detector.py`

### 7. Exercise Scoring

- **Input**: Trimmed 3D pose data
- **Process**: CNN-LSTM model provides detailed exercise scoring
- **Model**: `scorer_model.keras` + `scorer_scaler.pkl`
- **Expected Input**: (1, 48, features) - 48 uniformly sampled frames
- **Output**: Score 0.0-4.0 (0.0 = perfect, 4.0 = worst)
- **File**: `exercise_scorer.py`

## Score Interpretations

### Exercise Quality Scoring (0.0 - 4.0)

- **0.0-0.5**: "Excellent form" (nearly perfect)
- **0.5-1.0**: "Very good form"
- **1.0-1.5**: "Good form"
- **1.5-2.0**: "Fair form"
- **2.0-2.5**: "Poor form"
- **2.5-3.0**: "Very poor form"
- **3.0-4.0**: "Extremely poor form"

## API Endpoints

### Video Processing

```http
POST /process-video
Content-Type: multipart/form-data

# Upload video file for processing
# Returns job_id for tracking
```

### Status Tracking

```http
GET /job-status/{job_id}

# Returns current processing status and all quality scores
```

### Data Access

```http
# Individual data stages
GET /video-data/{job_id}/original     # MoveNet pose data
GET /video-data/{job_id}/kinect2d     # Kinect 2D format
GET /video-data/{job_id}/kinect3d     # With depth predictions
GET /video-data/{job_id}/trimmed      # Trimmed exercise segment

# Complete results (recommended)
GET /video-data/{job_id}/final        # All scores + 3D skeleton data
```

## Response Formats

### Job Status Response

```json
{
  "id": "job_uuid",
  "filename": "exercise_video.mp4",
  "status": "completed|processing|failed|uploaded",
  "message": "Processing completed successfully - Score: 0.8/4.0 (Very good form)",
  "created_at": 1640995200.0,
  "updated_at": 1640995300.0,
  "files": {
    "video": "/path/to/video",
    "original": "/path/to/movenet.csv",
    "kinect2d": "/path/to/kinect2d.csv",
    "kinect3d": "/path/to/kinect3d.csv",
    "trimmed": "/path/to/trimmed.csv"
  },
  "quality_scores": {
    "ugly_2d_goodness": 0.75,
    "ugly_2d_confidence": 0.82,
    "bad_3d_exercise_score": 0.23,
    "final_exercise_score": 0.8,
    "score_interpretation": "Very good form"
  }
}
```

### Final Results Response

```json
{
  "job_id": "job_uuid",
  "filename": "exercise_video.mp4",
  "status": "completed",
  "message": "Processing completed successfully - Score: 0.8/4.0 (Very good form)",
  "created_at": 1640995200.0,
  "updated_at": 1640995300.0,
  "quality_scores": {
    "ugly_2d_goodness": 0.75,
    "ugly_2d_confidence": 0.82,
    "bad_3d_exercise_score": 0.23,
    "final_exercise_score": 0.8,
    "score_interpretation": "Very good form"
  },
  "skeleton_data": "FrameNo,head_x,head_y,head_z,left_shoulder_x,...\n0,245,120,850,200,180,900,...",
  "data_format": "kinect_3d_trimmed"
}
```

## Required Models

All models should be placed in the `models/` directory:

### Ugly 2D Detection

- `trained_hybrid_model.keras` - Hybrid CNN model for video quality assessment

### Pose Conversion Models

- `xy_to_xy_best_2.keras` - MoveNet to Kinect 2D conversion
- `X_scaler_best.pkl` - Input scaler for conversion
- `y_scaler_best.pkl` - Output scaler for conversion

### Depth Prediction

- `kinect_depth_model.keras` - 2D to 3D depth prediction model

### Frame Trimming

- `kinect_cutting_model.keras` or `kinect_cutting_model_with_distances.keras`
- `kinect_cutting_scaler.pkl` or `kinect_cutting_scaler_with_distances.pkl`

### Exercise Quality Assessment

- `kinect_dense_good_vs_bad_model.keras` - Exercise form classifier
- `kinect_dense_good_vs_bad_scaler.pkl` - Preprocessing scaler

### Exercise Scoring

- `scorer_model.keras` - CNN-LSTM scoring model
- `scorer_scaler.pkl` - Feature scaler
- `score_minmaxscaler.pkl` - Target score scaler

## Error Handling

### Quality Gates

The pipeline includes two quality gates that can stop processing:

1. **Ugly 2D Check**: Stops if video quality is too poor for reliable processing
2. **Bad 3D Check**: Stops if exercise form is detected as poor quality

### Error Types

- `UglyDetectionError`: Video quality too poor
- `BadExerciseError`: Exercise form too poor
- `FileNotFoundError`: Missing model files
- Standard HTTP errors (404, 400, 500)

### Graceful Fallbacks

- If quality check models fail, pipeline continues with warnings
- If scoring fails, exercise still marked as successful if it passes quality checks
- Missing intermediate files fall back to previous stage data

## File Structure

```
project/
├── main.py                          # Server entry point
├── video_processing_route.py        # Main FastAPI routes
├── movenet_extraction.py           # MoveNet pose extraction
├── ugly_2d_detector.py            # Video quality assessment
├── posenet_to_kinect2d.py         # Format conversion
├── kinect2d_to_kinect3d.py        # Depth prediction
├── frame_trimmer.py               # Exercise segmentation
├── bad_3d_detector.py             # Exercise form quality check
├── exercise_scorer.py             # Final scoring
├── model_loader.py                # Model management utilities
├── models/                        # All trained models
│   ├── trained_hybrid_model.keras
│   ├── xy_to_xy_best_2.keras
│   ├── kinect_depth_model.keras
│   ├── kinect_cutting_model.keras
│   ├── kinect_dense_good_vs_bad_model.keras
│   └── scorer_model.keras
└── outputs/                       # Processing outputs
    └── {job_id}/                  # Per-job directories
```

## Usage Examples

### Basic Video Processing

```python
import requests

# Upload video
files = {'file': open('exercise_video.mp4', 'rb')}
response = requests.post('http://localhost:8000/process-video', files=files)
job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:8000/job-status/{job_id}')
print(status.json())

# Get final results
results = requests.get(f'http://localhost:8000/video-data/{job_id}/final')
final_data = results.json()
print(f"Score: {final_data['quality_scores']['final_exercise_score']}")
print(f"Interpretation: {final_data['quality_scores']['score_interpretation']}")
```

### Processing Workflow

1. Upload video via `/process-video`
2. Poll `/job-status/{job_id}` until status is "completed" or "failed"
3. Retrieve results via `/video-data/{job_id}/final`
4. Parse skeleton data for 3D visualization
5. Display scores and interpretations to user

## Performance Considerations

- **CPU vs GPU**: Models can run on CPU (forced) or GPU
- **Processing Time**: Varies by video length and complexity
- **Memory Usage**: Models loaded once and reused
- **Concurrent Processing**: Background tasks handle multiple videos simultaneously
- **Storage**: Temporary files cleaned up after processing

## Dependencies

- **FastAPI**: Web framework
- **TensorFlow**: Deep learning models
- **OpenCV**: Video processing
- **NumPy/Pandas**: Data manipulation
- **scikit-learn**: Preprocessing utilities
- **Kaggle Hub**: MoveNet model downloads

## Configuration

Key parameters can be adjusted in each module:

- Frame sampling rates
- Model confidence thresholds
- Quality gate thresholds
- Sequence lengths for different models
- CPU/GPU usage settings
