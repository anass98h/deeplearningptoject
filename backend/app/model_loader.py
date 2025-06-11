from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple, Union
from pathlib import Path
import logging
import joblib
import json
import os
import sys
import re

# Add tensorflow import for Keras models
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Keras models will not be loadable.")
    
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',  'ML')))

@dataclass
class ModelInfo:
    name: str
    version: str
    path: str
    model_type: str  # "regression", "classifier", "depth_estimator"
    framework: str   # "sklearn", "keras", etc.

class ModelLoader:
    def __init__(
        self, 
        regression_dir: str = "ML/regression_models",
        classifier_dir: str = "ML/classifier_models",
        depth_dir: str = "ML/depth_models"
    ):
        self.regression_dir = Path(regression_dir)
        self.classifier_dir = Path(classifier_dir)
        self.depth_dir = Path(depth_dir)
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        
        # Create directories if they don't exist
        self.regression_dir.mkdir(exist_ok=True)
        self.classifier_dir.mkdir(exist_ok=True)
        self.depth_dir.mkdir(exist_ok=True)

    def load_models(self) -> None:
        """Load all models from all directories."""
        try:
            # Clear existing models
            self.models = {}
            self.model_info = {}
            
            # Load regression models
            self._load_models_from_dir(self.regression_dir, "regression")
            
            # Load classifier models
            self._load_models_from_dir(self.classifier_dir, "classifier")
            
            # Load depth estimation models
            self._load_models_from_dir(self.depth_dir, "depth_estimator")
            
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def _load_models_from_dir(self, directory: Path, model_type: str) -> None:
        """Load models from a specific directory."""
        # Look for both sklearn models (.pkl) and keras models (.h5, .keras)
        sklearn_files = list(directory.glob("*.pkl"))
        keras_files = list(directory.glob("*.h5")) + list(directory.glob("*.keras"))
        
        if not sklearn_files and not keras_files:
            logging.warning(f"No model files found in {directory}")
            return

        # Load scikit-learn models
        for model_path in sklearn_files:
            try:
                self._load_sklearn_model(model_path, model_type)
            except Exception as e:
                logging.error(f"Failed to load sklearn model {model_path}: {str(e)}")
        
        # Load Keras models if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            for model_path in keras_files:
                try:
                    self._load_keras_model(model_path, model_type)
                except Exception as e:
                    logging.error(f"Failed to load Keras model {model_path}: {str(e)}")
        elif keras_files:
            logging.warning(f"Found {len(keras_files)} Keras models but TensorFlow is not available")

    def _load_sklearn_model(self, model_path: Path, model_type: str) -> None:
        """Load a scikit-learn model."""
        model_name = model_path.stem
        model_data = joblib.load(model_path)

        if isinstance(model_data, tuple) and len(model_data) == 2:
            model, version = model_data
        else:
            model = model_data
            version = self._extract_version_from_filename(model_path.stem)

        # Store model and its info
        self.models[model_name] = model
        self.model_info[model_name] = ModelInfo(
            name=model_name,
            version=version,
            path=str(model_path),
            model_type=model_type,
            framework="sklearn"
        )
        logging.info(f"Loaded sklearn {model_type} model: {model_name} (version {version})")

    def _load_keras_model(self, model_path: Path, model_type: str) -> None:
        """Load a Keras model."""
        model_name = model_path.stem
        
        # Look for metadata file
        metadata_path = model_path.with_suffix('.json')
        version = "1.0.0"  # Default version
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    version = metadata.get('version', version)
            except Exception as e:
                logging.warning(f"Failed to load metadata for {model_path}: {str(e)}")
                version = self._extract_version_from_filename(model_path.stem)
        else:
            version = self._extract_version_from_filename(model_path.stem)
        
        # Load Keras model
        model = tf.keras.models.load_model(str(model_path))
        
        # Store model and its info
        self.models[model_name] = model
        self.model_info[model_name] = ModelInfo(
            name=model_name,
            version=version,
            path=str(model_path),
            model_type=model_type,
            framework="keras"
        )
        logging.info(f"Loaded Keras {model_type} model: {model_name} (version {version})")

    def _extract_version_from_filename(self, filename: str) -> str:
        """Extract version number from filename."""
        try:
            # Try to find version patterns like _v1.0.0 or -v1.0.0
            version_match = re.search(r'[_-]v(\d+(\.\d+)*)', filename)
            if version_match:
                return version_match.group(1)
            return "1.0.0"
        except Exception:
            return "1.0.0"

    def save_model(self, model: Any, name: str, version: str, model_type: str) -> None:
        """Save model along with its version information."""
        try:
            # Determine directory based on model type
            if model_type == "classifier":
                directory = self.classifier_dir
            elif model_type == "regression":
                directory = self.regression_dir
            elif model_type == "depth_estimator":
                directory = self.depth_dir
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Determine if it's a Keras model
            is_keras_model = TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model)
            
            if is_keras_model:
                # Save Keras model
                model_path = directory / f"{name}_v{version}.h5"
                model.save(str(model_path))
                
                # Save metadata
                metadata = {
                    'version': version,
                    'model_type': model_type,
                    'framework': 'keras'
                }
                with open(model_path.with_suffix('.json'), 'w') as f:
                    json.dump(metadata, f)
                
                logging.info(f"Saved Keras {model_type} model {name} version {version}")
            else:
                # Save scikit-learn model
                model_path = directory / f"{name}_v{version}.pkl"
                joblib.dump((model, version), model_path)
                logging.info(f"Saved sklearn {model_type} model {name} version {version}")
                
        except Exception as e:
            logging.error(f"Failed to save model {name}: {str(e)}")
            raise

    def get_model(self, name: str) -> Optional[Any]:
        return self.models.get(name)

    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        return self.model_info.get(name)

    def list_models(self, model_type: Optional[str] = None, framework: Optional[str] = None) -> List[ModelInfo]:
        """List models, optionally filtered by type and/or framework."""
        models = list(self.model_info.values())
        
        if model_type:
            models = [info for info in models if info.model_type == model_type]
            
        if framework:
            models = [info for info in models if info.framework == framework]
            
        return models