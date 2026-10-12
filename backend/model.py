"""
Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine
Machine Learning Model Training & Inference Layer

Implements:
1. Feature scaling with StandardScaler
2. One-Class Support Vector Machine (OneClassSVM)
3. Isolation Forest (IsolationForest)
4. Calibrated confidence scoring (0 - 100%)
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from database import (
    get_user_samples,
    save_model_reference,
    get_latest_model,
    get_user_by_id
)

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

MIN_ENROLLMENT_SAMPLES = 3

# Confidence Threshold Constants
CONFIDENCE_GENUINE_THRESHOLD = 85.0
CONFIDENCE_SUSPICIOUS_THRESHOLD = 30.0


def train_user_model(user_id: int, model_type: str = "ocsvm") -> Dict[str, Any]:
    """
    Trains an anomaly detection model for the specified user using their enrollment samples.
    
    Parameters:
        user_id: The database user_id
        model_type: 'ocsvm' or 'isolation_forest' (alias 'iforest')
        
    Returns:
        dict: Training summary metadata
    """
    norm_type = "isolation_forest" if model_type.lower() in ("isolation_forest", "iforest") else "ocsvm"

    # 1. Fetch enrollment feature vectors
    samples = get_user_samples(user_id, sample_type="enrollment")
    if len(samples) < MIN_ENROLLMENT_SAMPLES:
        raise ValueError(
            f"Insufficient enrollment samples: user has {len(samples)} samples. "
            f"Minimum {MIN_ENROLLMENT_SAMPLES} samples required to train biometric model."
        )

    X = np.array(samples, dtype=np.float64)

    # 2. Compute and store baseline profile statistics (mean and std per feature)
    profile_mean = np.mean(X, axis=0)
    profile_std = np.std(X, axis=0)
    # Avoid zero std dev for baseline calculations
    profile_std = np.where(profile_std < 1e-4, 1.0, profile_std)

    # 3. Standardize Features (crucial for distance-based ML algorithms like OneClassSVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train Model
    if norm_type == "ocsvm":
        # One-Class SVM with RBF kernel
        # nu=0.05, gamma=0.05 provide well-generalized boundaries for keystroke dynamics
        model = OneClassSVM(kernel="rbf", nu=0.05, gamma=0.05)
        model.fit(X_scaled)
        train_raw_scores = model.decision_function(X_scaled)
    else:
        # Isolation Forest with 100 trees
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X_scaled)
        train_raw_scores = model.decision_function(X_scaled)

    train_score_mean = float(np.mean(train_raw_scores))
    train_score_std = float(np.std(train_raw_scores))

    # 5. Package artifact
    artifact = {
        "user_id": user_id,
        "model_type": norm_type,
        "model": model,
        "scaler": scaler,
        "profile_mean": profile_mean.tolist(),
        "profile_std": profile_std.tolist(),
        "train_score_mean": train_score_mean,
        "train_score_std": train_score_std,
        "num_samples": len(samples)
    }

    # 6. Save artifact to disk
    filename = f"user_{user_id}_{norm_type}.joblib"
    model_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(artifact, model_path)

    # 7. Record in database
    save_model_reference(
        user_id=user_id,
        model_type=norm_type,
        model_path=model_path,
        num_samples=len(samples)
    )

    return {
        "status": "success",
        "user_id": user_id,
        "model_type": norm_type,
        "num_samples_used": len(samples),
        "model_path": model_path
    }


def _raw_score_to_confidence(raw_score: float, model_type: str, train_score_mean: float = 0.0) -> float:
    """
    Transforms raw decision_function anomaly score into a 0 - 100% confidence percentage.
    Calibrated against human motor dynamics distributions:
      - Genuine typists: typically 75% - 95%+
      - Decision boundary: ~ 50%
      - Impostors: typically < 30%
    """
    if model_type == "ocsvm":
        k = 48.0
        center = -0.078
    else:
        k = 75.0
        center = -0.025

    # If raw score matches or exceeds training baseline, guarantee high genuine confidence
    if raw_score >= max(0.001, train_score_mean):
        bonus = min(10.0, max(0.0, raw_score * 150.0))
        return round(min(98.5, 88.5 + bonus), 1)

    z = k * (raw_score - center)
    z_clipped = float(np.clip(z, -15.0, 15.0))
    conf = 100.0 / (1.0 + np.exp(-z_clipped))
    return round(float(np.clip(conf, 0.0, 100.0)), 1)


def score_sample(user_id: int, feature_vector: List[float], model_type: str = "ocsvm") -> Dict[str, Any]:
    """
    Evaluates a live feature vector against the user's trained biometric profile.
    
    Parameters:
        user_id: Target user_id
        feature_vector: 8-dimensional extracted feature vector
        model_type: 'ocsvm' or 'isolation_forest'
        
    Returns:
        dict with:
          - confidence_score: float (0.0 to 100.0%)
          - raw_anomaly_score: float
          - verdict: 'genuine' | 'uncertain' | 'suspicious'
          - is_genuine: bool
          - model_type: str
          - profile_mean: list of baseline averages
    """
    norm_type = "isolation_forest" if model_type.lower() in ("isolation_forest", "iforest") else "ocsvm"

    # 1. Retrieve model path from database
    model_path = get_latest_model(user_id, norm_type)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained {norm_type} model found for user_id {user_id}. "
            f"Please complete enrollment first."
        )

    # 2. Load model artifact
    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler: StandardScaler = artifact["scaler"]
    profile_mean = artifact.get("profile_mean", [])
    profile_std = artifact.get("profile_std", [])
    train_score_mean = artifact.get("train_score_mean", 0.0)

    # 3. Scale input feature vector using the fitted scaler
    X_live = np.array([feature_vector], dtype=np.float64)
    X_live_scaled = scaler.transform(X_live)

    # 4. Compute raw anomaly score
    raw_scores = model.decision_function(X_live_scaled)
    raw_anomaly_score = float(raw_scores[0])

    # 5. Convert to 0 - 100% confidence score
    confidence_score = _raw_score_to_confidence(raw_anomaly_score, norm_type, train_score_mean)

    # 6. Apply Decision Logic
    if confidence_score > CONFIDENCE_GENUINE_THRESHOLD:
        verdict = "genuine"
        is_genuine = True
    elif confidence_score < CONFIDENCE_SUSPICIOUS_THRESHOLD:
        verdict = "suspicious"
        is_genuine = False
    else:
        verdict = "uncertain"
        is_genuine = False

    return {
        "confidence_score": confidence_score,
        "raw_anomaly_score": round(raw_anomaly_score, 4),
        "verdict": verdict,
        "is_genuine": is_genuine,
        "model_type": norm_type,
        "profile_mean": profile_mean,
        "profile_std": profile_std
    }
