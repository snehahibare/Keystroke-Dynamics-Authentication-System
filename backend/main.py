"""
Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine
FastAPI Application & API Server

Endpoints:
- POST /enroll            : Enrolls typing samples, extracts features & trains OCSVM + IForest
- POST /verify            : Authenticates live typing sample, returns verdict & XAI explanations
- POST /verify/step-up    : Fallback password challenge for uncertain biometric verdicts
- POST /continuous-check  : Session liveness check with mid-session identity drift alert
- GET  /health            : API health check
"""

from typing import List, Dict, Any, Optional
import os
import sys
import hashlib
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    init_db,
    create_user,
    get_user_by_username,
    save_feature_sample,
    get_user_samples
)
from feature_extraction import extract_features, MIN_KEYSTROKES_REQUIRED
from model import train_user_model, score_sample
from explainability import explain_deviation

# Initialize Database Schema
init_db()

app = FastAPI(
    title="Personal Digital DNA — Keystroke Dynamics Biometric Engine",
    description="Module 1: Motor control behavioral biometric authentication and continuous anomaly monitoring",
    version="1.0.0"
)

# Enable CORS for Frontend Interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Active Session Tracker for Continuous Authentication
# Maps username -> last_confidence_score
session_confidence_tracker: Dict[str, float] = {}

# Simple Mock Credential Store for Step-Up Verification Placeholder
# In production, this would query hashed passwords or MFA tokens
MOCK_PASSWORDS = {
    "default": hashlib.sha256("password123".encode()).hexdigest()
}


# ============================================================================
# Pydantic Request Models
# ============================================================================

class RawKeystrokeEvent(BaseModel):
    key_code: int
    event_type: str
    timestamp_ms: float

class EnrollRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    # Support both 'keystroke_samples' and 'samples' aliases
    keystroke_samples: Optional[List[List[RawKeystrokeEvent]]] = None
    samples: Optional[List[List[RawKeystrokeEvent]]] = None

class VerifyRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    # Support both 'keystroke_events' and 'sample' aliases
    keystroke_events: Optional[List[RawKeystrokeEvent]] = None
    sample: Optional[List[RawKeystrokeEvent]] = None
    model_type: Optional[str] = "ocsvm"

class StepUpRequest(BaseModel):
    username: str
    password: str

class ContinuousCheckRequest(BaseModel):
    username: str
    keystroke_events: Optional[List[RawKeystrokeEvent]] = None
    sample: Optional[List[RawKeystrokeEvent]] = None
    previous_confidence: Optional[float] = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint to verify backend status."""
    return {
        "status": "healthy",
        "module": "Personal Digital DNA - Keystroke Dynamics",
        "version": "1.0.0"
    }


@app.post("/enroll")
def enroll_user(req: EnrollRequest):
    """
    Enrolls a user's typing biometric signature.
    Requires at least 3-4 distinct typing samples to train both One-Class SVM and Isolation Forest.
    """
    samples_data = req.keystroke_samples or req.samples
    if not samples_data or len(samples_data) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 3-4 keystroke samples are required for biometric profile enrollment."
        )

    # 1. Create or retrieve user
    username = req.username.strip().lower()
    user_id = create_user(username)

    # 2. Extract and persist features for each sample
    extracted_vectors = []
    for idx, raw_sample in enumerate(samples_data):
        raw_list = [ev.model_dump() for ev in raw_sample]
        try:
            vector = extract_features(raw_list)
            save_feature_sample(user_id=user_id, feature_vector=vector, sample_type="enrollment")
            extracted_vectors.append(vector)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Sample {idx + 1} validation failed: {str(e)}"
            )

    # 3. Train both One-Class SVM and Isolation Forest models
    try:
        train_user_model(user_id=user_id, model_type="ocsvm")
        train_user_model(user_id=user_id, model_type="isolation_forest")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to train biometric models: {str(e)}"
        )

    total_samples = len(get_user_samples(user_id, sample_type="enrollment"))

    return {
        "status": "success",
        "message": f"Biometric profile enrolled successfully for user '{username}'.",
        "user_id": user_id,
        "samples_count": len(extracted_vectors),
        "total_samples_stored": total_samples,
        "models_trained": ["ocsvm", "isolation_forest"]
    }


@app.post("/verify")
def verify_user(req: VerifyRequest):
    """
    Verifies a live typing sample against the enrolled biometric profile.
    Returns genuine/uncertain/suspicious verdict, confidence percentage, and XAI explanation.
    """
    raw_events = req.keystroke_events or req.sample
    if not raw_events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No keystroke events provided for verification."
        )

    # 1. Verify user exists
    username = req.username.strip().lower()
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' is not enrolled in the biometric registry."
        )

    user_id = user["user_id"]

    # 2. Extract features from live sample
    raw_list = [ev.model_dump() for ev in raw_events]
    try:
        live_vector = extract_features(raw_list)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Verification keystroke validation failed: {str(e)}"
        )

    # 3. Store sample in DB as verification audit trail
    save_feature_sample(user_id=user_id, feature_vector=live_vector, sample_type="verification")

    # 4. Score sample with requested model
    model_type = req.model_type or "ocsvm"
    try:
        scoring_result = score_sample(user_id=user_id, feature_vector=live_vector, model_type=model_type)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # 5. Generate Explainability Layer diagnostics
    xai_info = explain_deviation(user_id=user_id, live_feature_vector=live_vector)

    confidence = scoring_result["confidence_score"]
    verdict = scoring_result["verdict"]
    requires_step_up = (verdict == "uncertain")

    # Cache confidence in active session tracker
    session_confidence_tracker[username] = confidence

    return {
        "status": "success",
        "username": username,
        "verdict": verdict,
        "confidence_score": confidence,
        "raw_anomaly_score": scoring_result["raw_anomaly_score"],
        "model_type": scoring_result["model_type"],
        "requires_step_up": requires_step_up,
        "explanation_summary": xai_info["summary"],
        "explanation": xai_info["summary"],
        "deviations": xai_info["top_deviations"],
        "all_deviations": xai_info["all_deviations"]
    }


@app.post("/verify/step-up")
def step_up_verification(req: StepUpRequest):
    """
    Step-up password challenge invoked when biometric confidence is uncertain (30% - 85%).
    In this prototype, accepts password verification against placeholder credentials.
    """
    username = req.username.strip().lower()
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found."
        )

    # Validate against placeholder password ('password123' or any non-empty input for demo)
    if req.password and len(req.password.strip()) >= 4:
        return {
            "status": "success",
            "step_up_passed": True,
            "message": "Step-up credential verified. Access authorized."
        }
    else:
        return {
            "status": "failed",
            "step_up_passed": False,
            "message": "Invalid password. Step-up authentication failed."
        }


@app.post("/continuous-check")
def continuous_check(req: ContinuousCheckRequest):
    """
    Continuous session monitoring endpoint.
    Compares live keystroke cadence against baseline and detects mid-session confidence drops.
    If confidence drops by > 25 percentage points from previous interval, flags an anomaly alert.
    """
    raw_events = req.keystroke_events or req.sample
    if not raw_events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Keystroke events required for continuous check."
        )

    username = req.username.strip().lower()
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not enrolled."
        )

    user_id = user["user_id"]

    # Extract features from current interval window
    raw_list = [ev.model_dump() for ev in raw_events]
    try:
        live_vector = extract_features(raw_list)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Continuous sample error: {str(e)}"
        )

    # Score sample using One-Class SVM by default for continuous monitor
    scoring_result = score_sample(user_id=user_id, feature_vector=live_vector, model_type="ocsvm")
    current_conf = scoring_result["confidence_score"]

    # Retrieve previous confidence from request or session memory
    prev_conf = req.previous_confidence
    if prev_conf is None:
        prev_conf = session_confidence_tracker.get(username, current_conf)

    # Calculate confidence delta (drop)
    conf_delta = round(float(prev_conf - current_conf), 2)
    session_anomaly_flag = False
    alert_message = "Typing rhythm remains consistent with active session baseline."

    # Detect sudden confidence drop (> 25% drop or current confidence drops into suspicious zone)
    if conf_delta >= 25.0 or current_conf < 30.0:
        session_anomaly_flag = True
        alert_message = (
            f"Identity confidence dropped mid-session by {conf_delta:.1f}% "
            f"(Current confidence: {current_conf:.1f}%). Possible unauthorized operator takeover!"
        )

    # Update session memory
    session_confidence_tracker[username] = current_conf

    xai_info = explain_deviation(user_id=user_id, live_feature_vector=live_vector)

    return {
        "status": "success",
        "username": username,
        "verdict": scoring_result["verdict"],
        "confidence_score": current_conf,
        "previous_confidence": prev_conf,
        "confidence_delta": conf_delta,
        "alert": session_anomaly_flag,
        "session_anomaly_flag": session_anomaly_flag,
        "message": alert_message,
        "explanation": xai_info["summary"],
        "deviations": xai_info["top_deviations"]
    }


class EvalSampleRequest(BaseModel):
    typist_name: str = Field(..., min_length=2)
    target_username: str = Field(..., min_length=2)
    keystroke_events: Optional[List[RawKeystrokeEvent]] = None
    sample: Optional[List[RawKeystrokeEvent]] = None


@app.post("/evaluation/save-sample")
def save_evaluation_sample_endpoint(req: EvalSampleRequest):
    """
    Saves a labeled test sample for benchmarking FAR/FRR/EER.
    Stored separately from model training data to avoid data leakage.
    """
    raw_events = req.keystroke_events or req.sample
    if not raw_events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No keystroke events provided."
        )

    raw_list = [ev.model_dump() for ev in raw_events]
    try:
        live_vector = extract_features(raw_list)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sample validation failed: {str(e)}"
        )

    from database import save_evaluation_sample
    is_genuine = (req.typist_name.strip().lower() == req.target_username.strip().lower())
    eval_id = save_evaluation_sample(
        typist_name=req.typist_name,
        target_username=req.target_username,
        is_genuine=is_genuine,
        feature_vector=live_vector
    )

    return {
        "status": "success",
        "eval_id": eval_id,
        "typist_name": req.typist_name,
        "target_username": req.target_username,
        "is_genuine": is_genuine,
        "feature_vector": live_vector
    }


@app.get("/evaluation/status")
def get_evaluation_status():
    """Returns overview of enrolled profiles and count of collected test trials."""
    from database import get_all_enrolled_users, get_evaluation_samples
    enrolled = get_all_enrolled_users()
    samples = get_evaluation_samples()
    gen_count = sum(1 for s in samples if s["is_genuine"])
    imp_count = sum(1 for s in samples if not s["is_genuine"])

    return {
        "status": "success",
        "enrolled_users": enrolled,
        "total_trials": len(samples),
        "genuine_trials": gen_count,
        "impostor_trials": imp_count
    }


@app.post("/evaluation/run-eval")
def run_evaluation_api():
    """
    Executes the evaluation loop on all collected real evaluation samples,
    computes FAR, FRR, EER, regenerates tradeoff chart, and returns metrics.
    """
    import subprocess
    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "evaluate.py")
    res = subprocess.run([sys.executable, "-u", eval_script], capture_output=True, text=True)

    from database import get_evaluation_samples, get_all_enrolled_users
    samples = get_evaluation_samples()
    enrolled = get_all_enrolled_users()
    unique_typists = len(set(s["typist_name"] for s in samples))

    # Parse evaluation results or call evaluate loop directly
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation"))
    from evaluate import run_evaluation, CHART_OUTPUT_PATH

    # Load samples and compute metrics directly
    from model import score_sample
    models_to_test = ["ocsvm", "isolation_forest"]
    results = {}
    import numpy as np

    for model_type in models_to_test:
        genuine_confidences = []
        impostor_confidences = []

        for sample in samples:
            target_username = sample["target_username"]
            user_info = get_user_by_username(target_username)
            if not user_info:
                continue
            user_id = user_info["user_id"]
            feat_vec = sample["feature_vector"]
            try:
                score_res = score_sample(user_id=user_id, feature_vector=feat_vec, model_type=model_type)
                conf = score_res["confidence_score"]
                if sample["is_genuine"]:
                    genuine_confidences.append(conf)
                else:
                    impostor_confidences.append(conf)
            except Exception:
                continue

        thresholds = np.linspace(0.0, 100.0, 201)
        far_list = []
        frr_list = []
        tot_g = max(1, len(genuine_confidences))
        tot_i = max(1, len(impostor_confidences))

        for th in thresholds:
            fa = sum(1 for c in impostor_confidences if c >= th)
            fr = sum(1 for c in genuine_confidences if c < th)
            far_list.append((fa / tot_i) * 100.0)
            frr_list.append((fr / tot_g) * 100.0)

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)
        abs_diff = np.abs(far_arr - frr_arr)
        min_idx = np.argmin(abs_diff)
        eer_val = float((far_arr[min_idx] + frr_arr[min_idx]) / 2.0)
        eer_th = float(thresholds[min_idx])

        idx_30 = np.argmin(np.abs(thresholds - 30.0))
        idx_85 = np.argmin(np.abs(thresholds - 85.0))

        results[model_type] = {
            "eer": round(eer_val, 2),
            "eer_threshold": round(eer_th, 1),
            "far_30": round(float(far_arr[idx_30]), 2),
            "frr_30": round(float(frr_arr[idx_30]), 2),
            "far_85": round(float(far_arr[idx_85]), 2),
            "frr_85": round(float(frr_arr[idx_85]), 2),
            "mean_gen_conf": round(float(np.mean(genuine_confidences)), 1) if genuine_confidences else 0.0,
            "mean_imp_conf": round(float(np.mean(impostor_confidences)), 1) if impostor_confidences else 0.0
        }

    return {
        "status": "success",
        "unique_typists": unique_typists,
        "total_trials": len(samples),
        "enrolled_count": len(enrolled),
        "ocsvm": results.get("ocsvm", {}),
        "isolation_forest": results.get("isolation_forest", {}),
        "output_log": res.stdout
    }


@app.get("/evaluation/chart")
def get_evaluation_chart():
    """Serves the generated FAR/FRR tradeoff curve PNG."""
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "far_frr_tradeoff.png")
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Chart not generated yet.")

