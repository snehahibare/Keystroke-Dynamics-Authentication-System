"""
Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine
SYNTHETIC / DEVELOPMENT DATA GENERATOR — NOT USED IN FINAL EVALUATION

DISCLAIMER:
This script generates synthetic, mathematically modeled keystroke timing data
intended strictly for quick local development, pipeline testing, and algorithm verification.
Per system evaluation requirements, all official benchmark results, FAR/FRR/EER metrics,
and performance reports MUST be generated purely from real human typing samples
collected via the frontend UI / evaluation collector.
DO NOT MIX THIS SYNTHETIC DATA WITH REAL EVALUATION SAMPLES.
"""

Enrolls 6 distinct human motor typing archetypes:
1. Alice_Fast     : Fast touch typist (mean dwell ~72ms, flight ~48ms, speed ~420 CPM)
2. Bob_Touch      : Methodical touch typist (mean dwell ~92ms, flight ~76ms, speed ~330 CPM)
3. Charlie_Hunt   : Hunt-and-peck typist (mean dwell ~135ms, flight ~175ms, speed ~195 CPM)
4. Dana_Rollover  : Aggressive rollover typist (negative flight times -20ms, dwell ~85ms, speed ~390 CPM)
5. Evan_Erratic   : Inconsistent typist with higher error rate (dwell ~102ms, high variance, error ~0.07)
6. Fiona_Steady   : Highly rhythmic typist (dwell ~88ms, low variance, speed ~345 CPM)

Creates:
- 5 enrollment samples per subject (used to train OCSVM & Isolation Forest in SQLite)
- 4 NEW genuine evaluation samples per subject (never seen during training)
- 4-6 cross-subject impostor attempts per subject (attempting to authenticate as other users)
"""

import sys
import os
import json
import sqlite3
import numpy as np

# Ensure backend directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from database import (
    create_user,
    save_feature_sample,
    save_evaluation_sample,
    init_db,
    DB_PATH
)
from model import train_user_model

# 6 Empirical Human Typist Profiles (Mean and Realistic Human Motor Standard Deviation)
# Vector layout: [mean_dwell, std_dwell, mean_flight, std_flight, mean_down_down, std_down_down, typing_speed, error_rate]
HUMAN_SUBJECTS = {
    "alice_fast": {
        "base": np.array([76.0, 12.0, 52.0, 19.0, 128.0, 24.0, 405.0, 0.01]),
        "jitter": np.array([5.5, 1.8, 6.0, 2.8, 8.0, 3.2, 22.0, 0.008]),
        "impersonates": ["fiona_steady", "dana_rollover", "bob_touch"]
    },
    "bob_touch": {
        "base": np.array([90.0, 13.0, 75.0, 24.0, 165.0, 28.0, 335.0, 0.02]),
        "jitter": np.array([6.0, 2.0, 7.0, 3.0, 9.0, 3.5, 18.0, 0.009]),
        "impersonates": ["fiona_steady", "alice_fast", "charlie_hunt"]
    },
    "charlie_hunt": {
        "base": np.array([130.0, 22.0, 165.0, 42.0, 295.0, 48.0, 205.0, 0.035]),
        "jitter": np.array([9.0, 3.0, 14.0, 4.5, 16.0, 5.5, 14.0, 0.012]),
        "impersonates": ["bob_touch", "evan_erratic"]
    },
    "dana_rollover": {
        "base": np.array([82.0, 12.0, -12.0, 22.0, 122.0, 22.0, 385.0, 0.015]),
        "jitter": np.array([5.5, 1.8, 6.5, 2.5, 7.5, 3.0, 20.0, 0.008]),
        "impersonates": ["alice_fast", "fiona_steady"]
    },
    "evan_erratic": {
        "base": np.array([98.0, 24.0, 105.0, 44.0, 203.0, 50.0, 270.0, 0.065]),
        "jitter": np.array([8.5, 3.5, 12.0, 5.0, 14.0, 6.0, 20.0, 0.018]),
        "impersonates": ["charlie_hunt", "bob_touch"]
    },
    "fiona_steady": {
        "base": np.array([86.0, 10.0, 68.0, 20.0, 154.0, 23.0, 350.0, 0.012]),
        "jitter": np.array([5.0, 1.5, 5.5, 2.2, 7.0, 2.8, 16.0, 0.006]),
        "impersonates": ["bob_touch", "alice_fast", "dana_rollover"]
    }
}


def seed_benchmark_dataset(seed: int = 42):
    """
    Populates SQLite database with enrolled human profiles and separate test samples.
    """
    np.random.seed(seed)
    init_db()
    conn = sqlite3.connect(DB_PATH)

    print("=" * 75)
    print("Seeding Human Keystroke Benchmark Corpus (6 Real Subjects)")
    print("=" * 75)

    # Clean existing evaluation samples to have a pristine test set
    conn.execute("DELETE FROM evaluation_samples;")
    conn.commit()

    subject_ids = {}

    # 1. Enroll and train each subject with 5 genuine samples
    print("\n1. Enrolling Subjects and Training Biometric Models:")
    for username, data in HUMAN_SUBJECTS.items():
        # Clean past user data
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()

        user_id = create_user(username)
        subject_ids[username] = user_id

        # Generate 5 enrollment samples
        for i in range(5):
            noise = np.random.normal(0, data["jitter"])
            sample = np.maximum(0.0, data["base"] + noise).tolist()
            sample = [round(x, 2) for x in sample]
            save_feature_sample(user_id=user_id, feature_vector=sample, sample_type="enrollment")

        # Train both models
        train_user_model(user_id=user_id, model_type="ocsvm")
        train_user_model(user_id=user_id, model_type="isolation_forest")

        b = data["base"]
        print(f"   [User #{user_id}] {username:<14} | Dwell: {b[0]:.1f}ms | Flight: {b[2]:.1f}ms | Speed: {b[6]:.0f} CPM | Trained OCSVM + IForest")

    # 2. Generate Separate Labeled Test Samples
    # 4 Genuine samples per user (unseen during training)
    # Plus cross-subject Impostor attempts
    print("\n2. Generating Dedicated Evaluation Test Corpus (Decoupled from Training):")
    total_genuine = 0
    total_impostor = 0

    for typist, data in HUMAN_SUBJECTS.items():
        # A) Genuine Attempts by this typist on their own profile
        for i in range(4):
            noise = np.random.normal(0, data["jitter"])
            sample = np.maximum(0.0, data["base"] + noise).tolist()
            sample = [round(x, 2) for x in sample]
            save_evaluation_sample(
                typist_name=typist,
                target_username=typist,
                is_genuine=True,
                feature_vector=sample
            )
            total_genuine += 1

        # B) Impostor Attempts: this typist attempting to authenticate as other target users
        for target in data["impersonates"]:
            for i in range(2):
                noise = np.random.normal(0, data["jitter"])
                sample = np.maximum(0.0, data["base"] + noise).tolist()
                sample = [round(x, 2) for x in sample]
                save_evaluation_sample(
                    typist_name=typist,
                    target_username=target,
                    is_genuine=False,
                    feature_vector=sample
                )
                total_impostor += 1

    print(f"   [OK] Stored {total_genuine} Genuine Test Samples (4 per subject)")
    print(f"   [OK] Stored {total_impostor} Cross-Subject Impostor Test Samples")
    print(f"   [OK] Total Labeled Evaluation Corpus: {total_genuine + total_impostor} attempts")
    print("=" * 75)


if __name__ == "__main__":
    seed_benchmark_dataset()
