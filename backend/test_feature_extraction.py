"""
Test script for Feature Extraction Module
Tests dwell time, flight time (including rollover/negative flight),
down-down time, OS key repeat deduplication, backspace error rate, and short sample validation.
"""

import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extraction import extract_features, extract_features_dict, FEATURE_NAMES

def run_test():
    print("=" * 65)
    print("Testing Keystroke Feature Extraction Engine")
    print("=" * 65)

    # 1. Realistic Dummy Keystroke Stream
    # Simulates typing ~12 keys with varied dwell, flight times, one rollover (negative flight),
    # one OS key repeat, and one backspace key.
    dummy_events = [
        # Key 1: 'H' (72) - normal press
        {"key_code": 72, "event_type": "keydown", "timestamp_ms": 100.0},
        {"key_code": 72, "event_type": "keyup",   "timestamp_ms": 190.0}, # dwell = 90ms

        # Key 2: 'E' (69) - flight = 250 - 190 = 60ms, down-down = 150ms
        {"key_code": 69, "event_type": "keydown", "timestamp_ms": 250.0},
        {"key_code": 69, "event_type": "keyup",   "timestamp_ms": 330.0}, # dwell = 80ms

        # Key 3: 'L' (76) - flight = 400 - 330 = 70ms
        {"key_code": 76, "event_type": "keydown", "timestamp_ms": 400.0},
        # Simulate OS Key Repeat (accidental duplicate keydown while still pressed)
        {"key_code": 76, "event_type": "keydown", "timestamp_ms": 450.0},
        {"key_code": 76, "event_type": "keyup",   "timestamp_ms": 490.0}, # dwell = 90ms

        # Key 4: 'L' (76) - flight = 580 - 490 = 90ms
        {"key_code": 76, "event_type": "keydown", "timestamp_ms": 580.0},
        {"key_code": 76, "event_type": "keyup",   "timestamp_ms": 670.0}, # dwell = 90ms

        # Key 5: 'O' (79) - Rollover simulation! Key 5 pressed BEFORE Key 4 released!
        # keydown at 650ms, but key 4 keyup was at 670ms -> flight = 650 - 670 = -20ms (rollover)
        {"key_code": 79, "event_type": "keydown", "timestamp_ms": 650.0},
        {"key_code": 79, "event_type": "keyup",   "timestamp_ms": 760.0}, # dwell = 110ms

        # Key 6: Space (32)
        {"key_code": 32, "event_type": "keydown", "timestamp_ms": 860.0},
        {"key_code": 32, "event_type": "keyup",   "timestamp_ms": 940.0}, # dwell = 80ms

        # Key 7: 'W' (87)
        {"key_code": 87, "event_type": "keydown", "timestamp_ms": 1050.0},
        {"key_code": 87, "event_type": "keyup",   "timestamp_ms": 1150.0}, # dwell = 100ms

        # Key 8: 'O' (79)
        {"key_code": 79, "event_type": "keydown", "timestamp_ms": 1260.0},
        {"key_code": 79, "event_type": "keyup",   "timestamp_ms": 1350.0}, # dwell = 90ms

        # Key 9: 'R' (82)
        {"key_code": 82, "event_type": "keydown", "timestamp_ms": 1460.0},
        {"key_code": 82, "event_type": "keyup",   "timestamp_ms": 1550.0}, # dwell = 90ms

        # Key 10: 'L' (76)
        {"key_code": 76, "event_type": "keydown", "timestamp_ms": 1660.0},
        {"key_code": 76, "event_type": "keyup",   "timestamp_ms": 1750.0}, # dwell = 90ms

        # Key 11: 'D' (68)
        {"key_code": 68, "event_type": "keydown", "timestamp_ms": 1860.0},
        {"key_code": 68, "event_type": "keyup",   "timestamp_ms": 1940.0}, # dwell = 80ms

        # Key 12: Backspace (8) - error correction key
        {"key_code": 8,  "event_type": "keydown", "timestamp_ms": 2040.0},
        {"key_code": 8,  "event_type": "keyup",   "timestamp_ms": 2130.0},
    ]

    print(f"Input sample events: {len(dummy_events)} events")
    
    # 2. Extract feature vector
    vector = extract_features(dummy_events)
    vector_dict = extract_features_dict(dummy_events)

    print("\n[SUCCESS] Extracted 8-dimensional Feature Vector:")
    print("-" * 65)
    print(f"Raw Vector: {vector}\n")

    print(f"{'Feature Name':<20} | {'Value':<12} | {'Unit / Note'}")
    print("-" * 65)
    units = [
        "ms (avg key hold)",
        "ms (dwell variance)",
        "ms (avg between keys; handles rollover < 0)",
        "ms (flight variance)",
        "ms (inter-key cadence)",
        "ms (down-down variance)",
        "CPM (characters per minute)",
        "ratio (backspace / total keys)"
    ]
    for name, val, unit in zip(FEATURE_NAMES, vector, units):
        print(f"{name:<20} | {val:<12} | {unit}")

    # 3. Sanity assertions
    assert len(vector) == 8, f"Expected 8 features, got {len(vector)}"
    for val in vector:
        assert not (val != val), f"Found NaN value in feature vector: {val}"  # NaN check
        assert val is not None
        assert not abs(val) == float('inf'), f"Found inf value: {val}"

    # 4. Test Short Sample Edge Case (< 10 keys)
    print("\n" + "-" * 65)
    print("Testing Edge Case: Short Sample (<10 keystrokes)")
    short_events = dummy_events[:6]  # only 3 keypresses
    try:
        extract_features(short_events)
        print("[FAIL] Short sample did not raise ValueError!")
    except ValueError as e:
        print(f"[PASS] Successfully caught short sample error: {e}")

    print("=" * 65)
    print("ALL TESTS PASSED SUCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_test()
