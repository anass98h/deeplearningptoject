#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

def predict_csv(input_csv, output_csv, model_path, x_scaler_path, y_scaler_path):
    # 1) Load input data
    df = pd.read_csv(input_csv)
    if "FrameNo" not in df.columns:
        raise ValueError("Input CSV must contain a 'FrameNo' column")
    frame_no = df["FrameNo"].values

    # 2) Extract feature columns
    feature_cols = [c for c in df.columns if c != "FrameNo"]
    X = df[feature_cols].to_numpy(dtype=float)

    # 3) Load scalers and model
    X_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    model    = keras.models.load_model(model_path)

    # 4) Scale, predict, and inverse‐scale
    X_scaled     = X_scaler.transform(X)
    y_pred_scaled = model.predict(X_scaled)
    y_pred        = y_scaler.inverse_transform(y_pred_scaled)

    # 5) Build and save output
    df_out = pd.DataFrame(y_pred, columns=feature_cols)
    df_out.insert(0, "FrameNo", frame_no)
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Predictions written to {output_csv}")

if __name__ == "__main__":
    # Prompt the user for all file paths
    input_csv     = "A1.csv"
    output_csv    = "predictions.csv"
    model_path    = "xy_to_xy_best.keras"
    x_scaler_path = "X_scaler.pkl"
    y_scaler_path = "y_scaler.pkl "

    predict_csv(input_csv, output_csv, model_path, x_scaler_path, y_scaler_path)
