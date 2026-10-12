"""
Test script for SQLite Database & Model Training (OCSVM & Isolation Forest)
Validates:
1. User creation and database storage
2. Saving 5 synthetic enrollment feature samples
3. Model training with StandardScaler for both OneClassSVM and Isolation Forest
4. Scoring with:
   (a) Similar vector (genuine typist) -> Expect confidence > 85%
   (b) Very different vector (impostor typist) -> Expect confidence < 30%
"""

import sys
import os
import sqlite3
import numpy as np

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import create_user, save_feature_sample, init_db, DB_PATH
from model import train_user_model, score_sample
from feature_extraction import FEATURE_NAMES

def run_model_test():
    print("=" * 70)
    print("Testing Keystroke Dynamics ML Biometric Engine (OCSVM & Isolation Forest)")
    print("=" * 70)

    # 1. Initialize DB and create a fresh test user
    init_db()
    test_username = "alice_biometric_test"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (test_username,))
        conn.commit()

    user_id = create_user(test_username)
    print(f"1. Created Fresh User '{test_username}' with user_id: {user_id}")

    # 2. Generate 5 synthetic enrollment samples
    # Alice's typical baseline typing profile:
    # mean_dwell ~ 88ms, std_dwell ~ 9ms, mean_flight ~ 75ms, std_flight ~ 28ms,
    # mean_down_down ~ 163ms, std_down_down ~ 32ms, typing_speed ~ 340 CPM, error_rate ~ 0.02
    np.random.seed(42)
    base_vector = np.array([88.0, 9.0, 75.0, 28.0, 163.0, 32.0, 340.0, 0.02])

    print("\n2. Storing 5 Enrollment Feature Samples for User...")
    for i in range(5):
        # Add slight natural human typing variance (+/- 3-5%)
        noise = np.random.normal(loc=0.0, scale=[2.0, 0.8, 3.0, 1.5, 4.0, 2.0, 8.0, 0.005])
        sample = np.maximum(0.0, base_vector + noise).tolist()
        sample = [round(x, 3) for x in sample]
        save_feature_sample(user_id, sample, sample_type="enrollment")
        print(f"   Sample {i+1}: Dwell={sample[0]}ms, Flight={sample[2]}ms, Speed={sample[6]} CPM")

    # 3. Train both One-Class SVM and Isolation Forest
    print("\n3. Training Biometric Models...")
    res_ocsvm = train_user_model(user_id, model_type="ocsvm")
    print(f"   [OK] One-Class SVM Trained successfully. Saved: {os.path.basename(res_ocsvm['model_path'])}")

    res_iforest = train_user_model(user_id, model_type="isolation_forest")
    print(f"   [OK] Isolation Forest Trained successfully. Saved: {os.path.basename(res_iforest['model_path'])}")

    # 4. Create Evaluation Vectors:
    # (A) SIMILAR VECTOR (Alice returning to log in - slightly different, but genuine)
    similar_vector = [89.5, 9.2, 74.0, 27.5, 163.5, 31.8, 338.0, 0.021]

    # (B) DIFFERENT VECTOR (Impostor Bob typing with very sluggish, erratic keys)
    # Dwell = 220ms (slow hold), Flight = 240ms, Speed = 110 CPM (sluggish), Error Rate = 0.15
    different_vector = [220.0, 35.0, 240.0, 85.0, 460.0, 95.0, 110.0, 0.15]

    print("\n4. Evaluating Test Vectors:")
    print("-" * 70)
    print(f"{'Metric':<25} | {'SIMILAR (Genuine Typist)':<20} | {'DIFFERENT (Impostor)':<20}")
    print("-" * 70)
    print(f"{'Mean Dwell':<25} | {similar_vector[0]:<20} | {different_vector[0]:<20}")
    print(f"{'Mean Flight':<25} | {similar_vector[2]:<20} | {different_vector[2]:<20}")
    print(f"{'Typing Speed (CPM)':<25} | {similar_vector[6]:<20} | {different_vector[6]:<20}")
    print(f"{'Error Rate':<25} | {similar_vector[7]:<20} | {different_vector[7]:<20}")
    print("-" * 70)

    # 5. Score with One-Class SVM
    print("\n>>> MODEL 1: ONE-CLASS SVM")
    score_similar_ocsvm = score_sample(user_id, similar_vector, model_type="ocsvm")
    score_diff_ocsvm = score_sample(user_id, different_vector, model_type="ocsvm")

    print(f"   [Genuine Case]  Raw Score: {score_similar_ocsvm['raw_anomaly_score']:>7.4f} | "
          f"Confidence: {score_similar_ocsvm['confidence_score']:>5.1f}% | "
          f"Verdict: {score_similar_ocsvm['verdict'].upper()}")
    print(f"   [Impostor Case] Raw Score: {score_diff_ocsvm['raw_anomaly_score']:>7.4f} | "
          f"Confidence: {score_diff_ocsvm['confidence_score']:>5.1f}% | "
          f"Verdict: {score_diff_ocsvm['verdict'].upper()}")

    # 6. Score with Isolation Forest
    print("\n>>> MODEL 2: ISOLATION FOREST")
    score_similar_iforest = score_sample(user_id, similar_vector, model_type="isolation_forest")
    score_diff_iforest = score_sample(user_id, different_vector, model_type="isolation_forest")

    print(f"   [Genuine Case]  Raw Score: {score_similar_iforest['raw_anomaly_score']:>7.4f} | "
          f"Confidence: {score_similar_iforest['confidence_score']:>5.1f}% | "
          f"Verdict: {score_similar_iforest['verdict'].upper()}")
    print(f"   [Impostor Case] Raw Score: {score_diff_iforest['raw_anomaly_score']:>7.4f} | "
          f"Confidence: {score_diff_iforest['confidence_score']:>5.1f}% | "
          f"Verdict: {score_diff_iforest['verdict'].upper()}")

    # Assertions
    print("\n" + "=" * 70)
    print("Verification Assertions:")
    assert score_similar_ocsvm['confidence_score'] >= 85.0, f"OCSVM genuine confidence too low: {score_similar_ocsvm['confidence_score']}"
    assert score_diff_ocsvm['confidence_score'] <= 30.0, f"OCSVM impostor confidence too high: {score_diff_ocsvm['confidence_score']}"
    assert score_similar_iforest['confidence_score'] >= 85.0, f"IForest genuine confidence too low: {score_similar_iforest['confidence_score']}"
    assert score_diff_iforest['confidence_score'] <= 30.0, f"IForest impostor confidence too high: {score_diff_iforest['confidence_score']}"

    print("[SUCCESS] All confidence scores demonstrate clear, robust separation!")
    print("=" * 70)

if __name__ == "__main__":
    run_model_test()
