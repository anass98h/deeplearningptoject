# Video Processing Pipeline Documentation

## Overview

This is a comprehensive video processing pipeline that analyzes exercise videos to extract pose data, assess video quality, evaluate exercise form, and provide detailed scoring. The pipeline includes multiple quality gates with advanced diagnostics and produces 3D skeleton data suitable for visualization.

## Pipeline Architecture

```
Video Upload → MoveNet Extraction → Ugly 2D Check → Kinect 2D Conversion → Kinect 3D Prediction → Frame Trimming → Bad 3D Check → Exercise Scoring
                                         ↓                                                                              ↓
                                    STOP if ugly                                                                 STOP if bad form
                                   (detailed analysis)                                                          (return error)
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

### 2. Ugly 2D Quality Check ⚠️ (Enhanced with Advanced Analysis)

- **Input**: Original video + MoveNet pose data
- **Process**: Hybrid model analyzes video frames + pose data for quality assessment
- **Model**: `trained_hybrid_model.keras` (expects 40 features with confidence scores)
- **Output**: Goodness score + Confidence score + **Detailed diagnostics**
- **Action**: **STOPS pipeline** if quality below thresholds
- **File**: `ugly_2d_detector.py`

**Quality thresholds:**

- Goodness threshold: 0.5 (minimum acceptable)
- Confidence threshold: 0.3 (minimum acceptable)

**Advanced Diagnostics Features:**

- **Confidence Analysis**: Identifies pose detection quality issues across 4 body blocks
- **Motion Jitter Analysis**: Detects unstable/shaky pose tracking
- **Timeline Analysis**: Shows exactly when and where problems occur
- **Block-level Diagnostics**: Pinpoints which body parts have tracking issues

**Note**: If MoveNet data lacks confidence scores (26 features), the system automatically re-extracts with confidence scores (40 features) for quality assessment, then strips them for pipeline compatibility.

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

## Advanced Quality Analysis

When videos fail the Ugly 2D quality check, the system provides comprehensive diagnostic information:

### Body Block Analysis

The system analyzes pose quality across 4 body regions:

- **top_left**: left_shoulder, left_elbow, left_hand
- **bottom_left**: left_hip, left_knee, left_foot
- **top_right**: right_shoulder, right_elbow, right_hand
- **bottom_right**: right_hip, right_knee, right_foot

### Confidence Analysis

- **Window-based assessment**: Analyzes pose detection confidence in 5-frame windows
- **Block averaging**: Identifies which body parts have poor detection quality
- **Statistical summary**: Provides min/max/average confidence scores
- **Problem identification**: Lists worst confidence windows with precise timing

### Motion Jitter Analysis

- **Frame-to-frame displacement**: Measures pose tracking stability
- **Euclidean distance calculation**: Quantifies movement between consecutive frames
- **Jitter detection**: Identifies extremely unstable tracking periods
- **Movement statistics**: Provides displacement distribution analysis

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
GET /video-data/{job_id}/kinect3d     # With depth predictions (full sequence)
GET /video-data/{job_id}/trimmed      # Trimmed exercise segment only
GET /video-data/{job_id}/untrimmed    # Full 3D sequence (alias for kinect3d)

# Complete results (recommended)
GET /video-data/{job_id}/final        # All scores + both trimmed & untrimmed data
```

## Response Formats

### Successful Job Response

```json
{
  "id": "job_uuid",
  "filename": "exercise_video.mp4",
  "status": "completed",
  "message": "Processing completed successfully - Score: 0.8/4.0 (Very good form)",
  "created_at": 1748603695.521178,
  "updated_at": 1748603712.371018,
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
    "score_interpretation": "Very good form",
    "advanced_analysis": {
      "summary": {
        "total_frames": 120,
        "windows_analyzed": 24,
        "has_confidence_data": true,
        "confidence_stats": { "min": 0.65, "max": 0.95, "avg": 0.82 },
        "motion_stats": { "min": 15.2, "max": 156.8, "avg": 45.3 }
      },
      "worst_confidence_windows": [],
      "most_jittery_windows": []
    }
  }
}
```

### Failed Job Response (with Detailed Diagnostics)

```json
{
  "id": "job_uuid",
  "filename": "B21.avi",
  "status": "failed",
  "message": "Video quality check failed: Model confidence too low for reliable processing. Confidence score: 0.289 (minimum required: 0.3)",
  "created_at": 1748603695.521178,
  "updated_at": 1748603712.371018,
  "files": {
    "video": "/path/to/video",
    "original": "/path/to/movenet.csv",
    "kinect2d": null,
    "kinect3d": null,
    "trimmed": null
  },
  "quality_scores": {
    "ugly_2d_goodness": 1.167,
    "ugly_2d_confidence": 0.289,
    "bad_3d_exercise_score": null,
    "final_exercise_score": null,
    "score_interpretation": null,
    "advanced_analysis": {
      "summary": {
        "total_frames": 262,
        "windows_analyzed": 52,
        "has_confidence_data": true,
        "confidence_stats": {
          "min": 0.117,
          "max": 0.45,
          "avg": 0.27
        },
        "motion_stats": {
          "min": 10.0,
          "max": 8924.9,
          "avg": 1429.6
        }
      },
      "worst_confidence_windows": [
        {
          "window": 39,
          "frames": "196-200",
          "time": "00:00:13-00:00:13",
          "worst_block": "bottom_right",
          "confidence": 0.117
        }
      ],
      "most_jittery_windows": [
        {
          "window": 17,
          "frames": "86-90",
          "time": "00:00:06-00:00:06",
          "worst_block": "top_left",
          "displacement": 8924.9
        }
      ]
    }
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
  "created_at": 1748603695.521178,
  "updated_at": 1748603712.371018,
  "quality_scores": {
    "ugly_2d_goodness": 0.75,
    "ugly_2d_confidence": 0.82,
    "bad_3d_exercise_score": 0.23,
    "final_exercise_score": 0.8,
    "score_interpretation": "Very good form",
    "advanced_analysis": { ... }
  },
  "skeleton_data": {
    "trimmed": "FrameNo,head_x,head_y,head_z,left_shoulder_x,...\n0,245,120,850,200,180,900,...",
    "untrimmed": "FrameNo,head_x,head_y,head_z,left_shoulder_x,...\n0,240,115,845,195,175,895,..."
  },
  "data_formats": {
    "trimmed": "kinect_3d_trimmed_exercise_segment",
    "untrimmed": "kinect_3d_full_sequence"
  }
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

## Error Handling & Quality Gates

### Quality Gates

The pipeline includes two quality gates that can stop processing:

1. **Ugly 2D Check**: Stops if video quality is too poor for reliable processing

   - Provides detailed diagnostics explaining why the video failed
   - Includes confidence analysis, motion jitter analysis, and problem timeline
   - Offers specific recommendations for improvement

2. **Bad 3D Check**: Stops if exercise form is detected as poor quality
   - Returns error message about exercise form issues
   - Includes exercise quality score for context

### Diagnostic Information for Failed Videos

When a video fails quality checks, users receive:

- **Specific problem areas**: Which body parts have tracking issues
- **Timeline information**: When problems occur (precise frame ranges and timestamps)
- **Statistical analysis**: Confidence and motion distribution data
- **Actionable recommendations**: How to improve video quality

### Example Diagnostic Insights

From a failed video analysis:

- **Problem**: "Very low pose detection confidence (avg 0.27, need 0.3+)"
- **Location**: "Problems concentrated in right leg area (bottom_right block)"
- **Timing**: "Most issues occur around 5-15 seconds into video"
- **Severity**: "Extremely unstable tracking (displacement up to 8924 units)"
- **Recommendation**: "Improve lighting and camera stability"

### Error Types

- `UglyDetectionError`: Video quality too poor (with detailed analysis)
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
├── ugly_2d_detector.py            # Video quality assessment + advanced analysis
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
result = status.json()

if result['status'] == 'completed':
    print(f"Score: {result['quality_scores']['final_exercise_score']}")
    print(f"Interpretation: {result['quality_scores']['score_interpretation']}")
elif result['status'] == 'failed':
    print(f"Failed: {result['message']}")
    # Check advanced analysis for detailed diagnostics
    analysis = result['quality_scores']['advanced_analysis']
    if analysis.get('worst_confidence_windows'):
        print("Confidence issues detected:")
        for window in analysis['worst_confidence_windows'][:3]:
            print(f"  Frame {window['frames']}: {window['confidence']} confidence")
    if analysis.get('most_jittery_windows'):
        print("Motion stability issues detected:")
        for window in analysis['most_jittery_windows'][:3]:
            print(f"  Frame {window['frames']}: {window['displacement']} displacement")

# Get final results with both trimmed and untrimmed skeleton data
results = requests.get(f'http://localhost:8000/video-data/{job_id}/final')
final_data = results.json()

# Access both datasets
trimmed_csv = final_data['skeleton_data']['trimmed']     # Exercise segment only
untrimmed_csv = final_data['skeleton_data']['untrimmed'] # Full video sequence

# Use trimmed data for exercise analysis and scoring visualization
# Use untrimmed data for full video playback or timeline analysis
```

### Handling Failed Videos

```python
def process_video_with_diagnostics(video_path):
    # Upload and process video
    files = {'file': open(video_path, 'rb')}
    response = requests.post('http://localhost:8000/process-video', files=files)
    job_id = response.json()['job_id']

    # Poll for completion
    while True:
        status = requests.get(f'http://localhost:8000/job-status/{job_id}')
        result = status.json()

        if result['status'] in ['completed', 'failed']:
            break
        time.sleep(2)

    if result['status'] == 'failed':
        # Extract diagnostic information
        analysis = result['quality_scores']['advanced_analysis']

        print("Video processing failed with detailed diagnostics:")
        print(f"Reason: {result['message']}")

        if analysis.get('summary'):
            summary = analysis['summary']
            print(f"Frames analyzed: {summary['total_frames']}")

            if summary.get('confidence_stats'):
                stats = summary['confidence_stats']
                print(f"Confidence: avg={stats['avg']}, min={stats['min']}, max={stats['max']}")

            if summary.get('motion_stats'):
                stats = summary['motion_stats']
                print(f"Motion: avg={stats['avg']}, max={stats['max']}")

        # Show specific problem windows
        if analysis.get('worst_confidence_windows'):
            print("\nWorst confidence periods:")
            for window in analysis['worst_confidence_windows'][:3]:
                print(f"  {window['time']}: {window['confidence']} ({window['worst_block']})")

        if analysis.get('most_jittery_windows'):
            print("\nMost unstable periods:")
            for window in analysis['most_jittery_windows'][:3]:
                print(f"  {window['time']}: {window['displacement']} displacement ({window['worst_block']})")

        return None
    else:
        return result
```

### Processing Workflow

1. Upload video via `/process-video`
2. Poll `/job-status/{job_id}` until status is "completed" or "failed"
3. If failed, extract diagnostic information from `advanced_analysis`
4. If successful, retrieve results via `/video-data/{job_id}/final`
5. Parse both trimmed and untrimmed skeleton data for visualization:
   - **Trimmed data**: Use for exercise scoring analysis and form feedback
   - **Untrimmed data**: Use for full video timeline and context visualization
6. Display scores and interpretations to user

## Performance Considerations

- **CPU vs GPU**: Models can run on CPU (forced) or GPU
- **Processing Time**: Varies by video length and complexity
- **Memory Usage**: Models loaded once and reused
- **Concurrent Processing**: Background tasks handle multiple videos simultaneously
- **Storage**: Temporary files cleaned up after processing
- **Analysis Overhead**: Advanced diagnostics add minimal processing time

## Benefits of Advanced Analysis

### For Users

- **Understand failures**: Know exactly why their video was rejected
- **Targeted improvements**: Specific recommendations (lighting, stability, etc.)
- **Problem timing**: Know which parts of the video had issues
- **Quality metrics**: Quantitative feedback on video suitability

### For Developers

- **Debugging**: Detailed logs and metrics for troubleshooting
- **Quality metrics**: Track video quality trends over time
- **User feedback**: Help users improve their video capture technique
- **System monitoring**: Identify common quality issues

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
- Analysis window sizes and parameters
