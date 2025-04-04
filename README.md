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
  - HTML5, CSS3, JavaScript
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

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install backend and deep learning dependencies:
```bash
pip install tensorflow
pip install keras
pip install numpy
pip install pandas
pip install fastapi  # or flask
pip install uvicorn  # if using FastAPI
pip install python-multipart
pip install scikit-learn
pip install matplotlib
pip install jupyter
pip install notebook
```

3. Install frontend dependencies:
```bash
# Navigate to frontend directory
cd frontend
npm install
npm install @tensorflow/tfjs
npm install @tensorflow-models/posenet
```

## Project Structure

```
project/
├── backend/
│   ├── api/           # API endpoints
│   ├── models/        # Deployed models for inference
│   │   ├── pose_estimation/  # Pose estimation models
│   │   └── pose_classification/  # Pose classification models
│   ├── data/          # Data storage
│   ├── utils/         # Utility functions
│   └── main.py        # Backend server
├── frontend/
│   ├── public/        # Static files
│   ├── src/           # Source code
│   │   ├── components/  # React components
│   │   ├── utils/      # Utility functions
│   │   └── App.js      # Main application
│   └── package.json   # Frontend dependencies
├── deep_learning/
│   ├── notebooks/     # Jupyter notebooks for experimentation
│   ├── training/      # Training scripts
│   │   ├── models/    # Model definitions
│   │   └── configs/   # Training configurations
│   ├── data/          # Training datasets
│   │   ├── raw/       # Raw collected data
│   │   ├── processed/ # Processed training data
│   │   └── augmented/ # Augmented data
│   ├── preprocessing/ # Data preprocessing scripts
│   ├── evaluation/    # Model evaluation scripts
│   └── utils/         # Utility functions
├── data/              # Shared data directory
└── tests/             # Test files
```

## Usage

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload  # if using FastAPI
# or
python main.py  # if using Flask
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Access the application:
- Open your browser and navigate to `http://localhost:3000`
- Allow camera access when prompted
- The system will start capturing and analyzing poses
- Data will be automatically saved to the backend

## Deep Learning Development and Deployment

The `deep_learning` directory is used for model development and training, while trained models are deployed to the `backend/models` directory for inference.

1. **Data Collection and Processing**:
   - Use the web interface to collect pose data
   - Data is stored in the `data/raw` directory
   - Preprocess data using scripts in `preprocessing/`

2. **Model Development**:
   - Create and experiment with models in Jupyter notebooks
   - Implement custom models in `training/models/`
   - Use `training/` scripts for batch training

3. **Model Evaluation**:
   - Evaluate models using scripts in `evaluation/`
   - Compare different architectures and hyperparameters
   - Generate performance metrics and visualizations

4. **Model Deployment**:
   - After training and evaluation, deploy the best model to `backend/models/`
   - Update the backend to use the new model
   - Test the deployed model with the web interface

5. **Example Workflow**:
```bash
# Start Jupyter notebook for experimentation
cd deep_learning
jupyter notebook

# Train a model
python training/train_model.py --config configs/training_config.json

# Evaluate model performance
python evaluation/evaluate_model.py --model_path training/models/saved_model

# Deploy the model to backend
python utils/deploy_model.py --source training/models/saved_model --target ../backend/models/pose_estimation/
```

## Data Storage

The system stores pose data in two formats:
- CSV files for tabular data
- JSON files for structured pose data

Data is organized by:
- Timestamp
- User ID (if implemented)
- Pose keypoints
- Confidence scores

## License


## Contributing

 