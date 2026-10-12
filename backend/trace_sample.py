"""
Diagnostic and Trace Tool for Keystroke Dynamics Pipeline
Traces the full data path:
1. Frontend capture (script.js performance.now() microsecond timestamps)
2. API Payload transmission
3. Feature extraction (dwell, flight, down-down deltas)
4. Stored feature vector in SQLite

Explains and demonstrates:
- Why automated browser synthetic keystrokes produce ~11ms intervals (CDP dispatch speed)
- How human keystrokes produce authentic ~85-95ms dwell and ~160ms down-down
- Verifies the fix for directional wording ("slower" vs "faster") and std dev floors
"""

import sys
import os
import json
import sqlite3
import numpy as np

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extraction import extract_features, extract_features_dict, FEATURE_NAMES
from explainability import explain_deviation
from database import create_user, save_feature_sample, init_db, DB_PATH
from model import train_user_model, score_sample

def trace_browser_vs_human():
    print("=" * 80)
    print("STAGE 1: TRACING THE ROOT CAUSE OF 11ms BASELINE VALUES")
    print("=" * 80)

    # 1. Inspect what was captured in SQLite for user 'alex_dna'
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE username = 'alex_dna'").fetchone()

    if user:
        samples = conn.execute(
            "SELECT sample_id, feature_vector FROM feature_samples WHERE user_id = ? AND sample_type = 'enrollment'",
            (user['user_id'],)
        ).fetchall()
        print(f"\n[A] Stored SQLite Feature Vectors for 'alex_dna' (user_id={user['user_id']}):")
        for s in samples:
            vec = json.loads(s['feature_vector'])
            print(f"    Sample #{s['sample_id']}: Mean Dwell={vec[0]:.2f}ms | Mean Down-Down={vec[4]:.2f}ms | Typing Speed={vec[6]:.1f} CPM")

    # 2. Trace the Pipeline for Browser Subagent Automated Events
    print("\n[B] Step-by-Step Pipeline Trace for Browser Automation Keystrokes:")
    print("    In browser_subagent, Chrome DevTools Protocol (CDP) `keyboard.type()` sends synthetic events.")
    print("    Let's trace 3 consecutive keys typed by the automated browser:")

    # Simulated raw events emitted by browser CDP automation
    cdp_raw_events = [
        {"key_code": 84, "event_type": "keydown", "timestamp_ms": 1000.12},
        {"key_code": 84, "event_type": "keyup",   "timestamp_ms": 1011.45}, # Dwell = 1011.45 - 1000.12 = 11.33ms
        {"key_code": 72, "event_type": "keydown", "timestamp_ms": 1017.30}, # Flight = 1017.30 - 1011.45 = 5.85ms, DD = 17.18ms
        {"key_code": 72, "event_type": "keyup",   "timestamp_ms": 1028.60}, # Dwell = 1028.60 - 1017.30 = 11.30ms
        {"key_code": 69, "event_type": "keydown", "timestamp_ms": 1034.40}, # Flight = 1034.40 - 1028.60 = 5.80ms, DD = 17.10ms
        {"key_code": 69, "event_type": "keyup",   "timestamp_ms": 1045.85}  # Dwell = 1045.85 - 1034.40 = 11.45ms
    ]

    print("    Stage 1: Frontend script.js capture:")
    for ev in cdp_raw_events:
        print(f"      - Event: {ev['event_type']:<7} | KeyCode: {ev['key_code']} | timestamp_ms: {ev['timestamp_ms']:.2f}")

    print("    Stage 2: Feature Extraction intermediate deltas:")
    for i in range(0, len(cdp_raw_events), 2):
        kd = cdp_raw_events[i]
        ku = cdp_raw_events[i+1]
        dwell = ku["timestamp_ms"] - kd["timestamp_ms"]
        print(f"      * Key {kd['key_code']}: KeyUp ({ku['timestamp_ms']}) - KeyDown ({kd['timestamp_ms']}) = DWELL: {dwell:.2f}ms")

    for i in range(0, len(cdp_raw_events) - 2, 2):
        curr_kd = cdp_raw_events[i]
        curr_ku = cdp_raw_events[i+1]
        next_kd = cdp_raw_events[i+2]
        flight = next_kd["timestamp_ms"] - curr_ku["timestamp_ms"]
        dd = next_kd["timestamp_ms"] - curr_kd["timestamp_ms"]
        print(f"      * Inter-key: Next_KeyDown ({next_kd['timestamp_ms']}) - Prev_KeyUp ({curr_ku['timestamp_ms']}) = FLIGHT: {flight:.2f}ms | DOWN-DOWN: {dd:.2f}ms")

    # 3. Contrast with Authentic Human Keystroke Timing
    print("\n[C] Comparison with Physical Human Keystroke Timings:")
    print("    A human typist experiences mechanical key travel and neurological motor latency:")
    print("    - Human keypress hold (dwell): ~75ms to 110ms")
    print("    - Human finger travel between keys (flight): ~50ms to 120ms")
    print("    - Human down-down interval: ~140ms to 240ms")
    print("    - Resulting human typing speed: ~250 to 450 CPM (50-90 Words Per Minute)")

    print("\n" + "=" * 80)
    print("STAGE 2: TESTING HUMAN BASELINE WITH CORRECTED EXPLAINABILITY & STD FLOORS")
    print("=" * 80)

    # Enroll a realistic human user 'david_human'
    human_username = "david_human"
    user_id = create_user(human_username)
    
    # Clean previous records if re-running
    conn.execute("DELETE FROM feature_samples WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM models WHERE user_id = ?", (user_id,))
    conn.commit()

    # Create 5 human-consistent enrollment samples:
    # Baseline: Dwell ~88ms, Flight ~74ms, Down-Down ~162ms, Speed ~340 CPM, Error ~0.02
    np.random.seed(101)
    base_human_vector = np.array([88.5, 9.2, 74.0, 27.5, 162.5, 31.8, 340.0, 0.02])

    print(f"1. Enrolling Realistic Human User '{human_username}' (user_id={user_id}):")
    for i in range(5):
        noise = np.random.normal(0, [2.0, 0.8, 3.0, 1.5, 4.0, 2.0, 8.0, 0.005])
        sample = np.maximum(0.0, base_human_vector + noise).tolist()
        sample = [round(x, 2) for x in sample]
        save_feature_sample(user_id, sample, sample_type="enrollment")
        print(f"   Enrollment Sample #{i+1}: Dwell={sample[0]}ms, Flight={sample[2]}ms, Down-Down={sample[4]}ms, Speed={sample[6]} CPM")

    # Train models
    train_user_model(user_id, "ocsvm")
    train_user_model(user_id, "isolation_forest")
    print("   [OK] Trained One-Class SVM and Isolation Forest on human baseline.")

    # 2. Test the EXACT Impostor Vector from User Prompt:
    # "dwell 250ms, flight 300ms, down-down 550ms"
    # Impostor: Dwell=250ms, Flight=300ms, Down-Down=550ms, Speed=110 CPM, Error=0.12
    impostor_vector = [250.0, 35.0, 300.0, 85.0, 550.0, 95.0, 110.0, 0.12]

    print("\n2. Evaluating Impostor Test Sample against Human Baseline:")
    print(f"   Impostor Live Vector: Dwell={impostor_vector[0]}ms, Flight={impostor_vector[2]}ms, Down-Down={impostor_vector[4]}ms, Speed={impostor_vector[6]} CPM")

    scoring = score_sample(user_id, impostor_vector, model_type="ocsvm")
    xai = explain_deviation(user_id, impostor_vector)

    print("\n3. Biometric Scoring Result:")
    print(f"   Verdict:          {scoring['verdict'].upper()}")
    print(f"   Confidence Score: {scoring['confidence_score']}%")
    print(f"   Raw Score:        {scoring['raw_anomaly_score']}")

    print("\n4. Explainability Diagnostics (Wording & Std Dev Floors Audit):")
    print(f"   Summary Sentence:")
    print(f"   \"{xai['summary']}\"")
    print("\n   Top Feature Deviations:")
    print(f"   {'-'*75}")
    print(f"   {'Feature':<26} | {'Baseline':<10} | {'Live':<10} | {'z-score':<8} | {'Description'}")
    print(f"   {'-'*75}")
    for dev in xai["top_deviations"]:
        print(f"   {dev['feature_name']:<26} | {dev['baseline_val']:<10.1f} | {dev['live_val']:<10.1f} | {dev['z_score']:<8.2f} | {dev['description']}")
    print(f"   {'-'*75}")

    # Verify Assertions
    print("\n5. Verification Checks:")
    # Check 1: z-score should be bounded and not absurdly large (e.g. |z| <= 15)
    for dev in xai["top_deviations"]:
        assert abs(dev["z_score"]) <= 15.0, f"z-score too high: {dev['z_score']}"
    print("   [PASS] All z-scores are bounded and mathematically sound (no astronomical values).")

    # Check 2: Directional wording must say "slower/longer" for high dwell, flight, down-down
    for dev in xai["top_deviations"]:
        if dev["feature_key"] in ("mean_dwell", "mean_flight", "mean_down_down"):
            assert "slower" in dev["description"] or "longer" in dev["description"], \
                f"Incorrect wording: {dev['description']}"
            assert "faster" not in dev["description"], f"Error: Higher interval labeled faster: {dev['description']}"
    print("   [PASS] High time intervals are correctly labeled 'slower / longer' (NOT 'faster').")

    # Check 3: Typing speed should say "slower" since 110 CPM < 340 CPM
    speed_dev = [d for d in xai["all_deviations"] if d["feature_key"] == "typing_speed"][0]
    assert "slower" in speed_dev["description"], f"Typing speed wording error: {speed_dev['description']}"
    print("   [PASS] Lower typing speed correctly labeled 'slower'.")

    print("\n" + "=" * 80)
    print("ALL AUDIT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    trace_browser_vs_human()
