# Real-time Pose Estimation and Analysis System

This project implements a web-based system that captures real-time human pose data using PoseNet, processes it through a backend system, and stores the pose data for analysis. The system consists of a frontend web interface and a backend processing server. Additionally, it includes components for training and experimenting with deep learning models.

## Project Overview

The system consists of three main components:

### Frontend
- Web interface that accesses the user's camera
- Real-time pose estimation using PoseNet
- Visualization of detected poses
- Data transmission to backend

### Backend
- Data processing and storage
- CSV/JSON file generation
- API endpoints for data handling
- Pose analysis capabilities
- Model inference using trained models

### Deep Learning Training
- Model training and experimentation
- Data preprocessing and augmentation
- Model evaluation and validation
- Hyperparameter tuning
- Model deployment to backend

## Technologies Used

- **Frontend**:
  - NextJs
  - TensorFlow.js
  - PoseNet model
  - Webcam API

- **Backend**:
  - Python
  - TensorFlow
  - Keras
  - FastAPI/Flask (for API endpoints)
  - Pandas (for data handling)

- **Deep Learning**:
  - TensorFlow
  - Keras
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib (for visualization)
  - Jupyter Notebook (for experimentation)

## Setup Instructions


## Project Structure

```
project/
├── backend/
├── frontend/
├── dl/
├── data/              # Shared data directory
└── tests/             # Test files
```
