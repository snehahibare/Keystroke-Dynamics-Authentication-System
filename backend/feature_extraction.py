"""
Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine
Feature Extraction Module

Converts raw timing events [{key_code, event_type, timestamp_ms}, ...]
into a fixed-size, standardized statistical feature vector:
[mean_dwell, std_dwell, mean_flight, std_flight, mean_down_down, std_down_down, typing_speed, error_rate]
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Standardized Feature Schema
FEATURE_NAMES = [
    "mean_dwell",
    "std_dwell",
    "mean_flight",
    "std_flight",
    "mean_down_down",
    "std_down_down",
    "typing_speed",
    "error_rate"
]

# Standard key codes for correction keys
BACKSPACE_KEY_CODE = 8
DELETE_KEY_CODE = 46
EDIT_KEY_CODES = {BACKSPACE_KEY_CODE, DELETE_KEY_CODE}

# Threshold constraints
MIN_KEYSTROKES_REQUIRED = 10
MAX_REASONABLE_PAUSE_MS = 10000.0  # 10s maximum pause threshold to clip anomalous idle gaps


def extract_features(raw_events: List[Dict[str, Any]]) -> List[float]:
    """
    Extracts a fixed-size 8-dimensional numeric feature vector from raw keystroke events.
    
    Parameters:
        raw_events: List of dicts with 'key_code' (int), 'event_type' ('keydown'|'keyup'), 
                    and 'timestamp_ms' (float).
                    
    Returns:
        List[float]: Fixed 8-element feature vector:
        [mean_dwell, std_dwell, mean_flight, std_flight, mean_down_down, std_down_down, 
         typing_speed, error_rate]
         
    Raises:
        ValueError: If sample contains fewer than MIN_KEYSTROKES_REQUIRED keypresses.
    """
    if not raw_events or not isinstance(raw_events, list):
        raise ValueError("Keystroke events must be a non-empty list of event objects.")

    # 1. Sort chronologically by timestamp_ms to guarantee sequential integrity
    sorted_events = sorted(raw_events, key=lambda e: e.get("timestamp_ms", 0.0))

    # 2. Separate and deduplicate keydown / keyup events (handling OS key-repeat)
    active_keydowns: Dict[int, float] = {}
    completed_presses: List[Dict[str, Any]] = []  # Ordered list of {key_code, down_time, up_time}
    total_raw_keydowns = 0
    edit_key_count = 0

    for event in sorted_events:
        key_code = int(event.get("key_code", 0))
        event_type = event.get("event_type", "").lower()
        timestamp = float(event.get("timestamp_ms", 0.0))

        if event_type == "keydown":
            # Track correction keys for error rate
            if key_code in EDIT_KEY_CODES:
                edit_key_count += 1

            # Handle OS Key-Repeat: if key is already pressed down without keyup, ignore repeat
            if key_code in active_keydowns:
                continue

            active_keydowns[key_code] = timestamp
            total_raw_keydowns += 1

            # Placeholder in sequence; up_time will be filled on keyup
            completed_presses.append({
                "key_code": key_code,
                "down_time": timestamp,
                "up_time": None
            })

        elif event_type == "keyup":
            if key_code in active_keydowns:
                down_time = active_keydowns.pop(key_code)
                # Find the corresponding press record
                for press in reversed(completed_presses):
                    if press["key_code"] == key_code and press["up_time"] is None:
                        press["up_time"] = timestamp
                        break

    # 3. Validation: Verify minimum sample length
    if total_raw_keydowns < MIN_KEYSTROKES_REQUIRED:
        raise ValueError(
            f"Sample too short: contains {total_raw_keydowns} valid keystrokes. "
            f"At least {MIN_KEYSTROKES_REQUIRED} keystrokes required for biometric extraction."
        )

    # 4. Compute Dwell Times (keyup - keydown)
    # Exclude backspace/delete keys from dwell stats to preserve motor rhythm
    dwell_times: List[float] = []
    for press in completed_presses:
        if press["up_time"] is not None and press["key_code"] not in EDIT_KEY_CODES:
            dwell = press["up_time"] - press["down_time"]
            if 10.0 <= dwell <= 3000.0:  # Plausible human dwell range (10ms to 3s)
                dwell_times.append(dwell)

    # 5. Compute Flight Times & Down-Down Times between consecutive keypresses
    # Filter out edit keys from timing sequence to analyze regular typing rhythm
    content_presses = [p for p in completed_presses if p["key_code"] not in EDIT_KEY_CODES]

    flight_times: List[float] = []
    down_down_times: List[float] = []

    for i in range(len(content_presses) - 1):
        curr_press = content_presses[i]
        next_press = content_presses[i + 1]

        # Down-Down Time: next_keydown - current_keydown
        dd_time = next_press["down_time"] - curr_press["down_time"]
        if 0.0 < dd_time <= MAX_REASONABLE_PAUSE_MS:
            down_down_times.append(dd_time)

        # Flight Time: next_keydown - current_keyup
        # Note: Can be NEGATIVE when keys overlap (key rollover), which is a key biometric trait!
        if curr_press["up_time"] is not None:
            flight_time = next_press["down_time"] - curr_press["up_time"]
            # Plausible range: -600ms (fast rollover) up to 5000ms pause
            if -600.0 <= flight_time <= MAX_REASONABLE_PAUSE_MS:
                flight_times.append(flight_time)

    # 6. Overall Typing Speed (Characters/Keystrokes Per Minute)
    first_time = sorted_events[0].get("timestamp_ms", 0.0)
    last_time = sorted_events[-1].get("timestamp_ms", 0.0)
    duration_ms = max(1.0, last_time - first_time)
    duration_minutes = (duration_ms / 1000.0) / 60.0
    typing_speed = float(total_raw_keydowns / duration_minutes) if duration_minutes > 0 else 0.0

    # 7. Error Rate (Backspace / Delete frequency)
    error_rate = float(edit_key_count / total_raw_keydowns) if total_raw_keydowns > 0 else 0.0

    # 8. Compute Mean & Standard Deviation with Robust Fallbacks
    def safe_stats(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.array(values, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr)) if len(values) > 1 else 0.0
        # Prevent NaN or inf
        if np.isnan(mean_val) or np.isinf(mean_val):
            mean_val = 0.0
        if np.isnan(std_val) or np.isinf(std_val):
            std_val = 0.0
        return round(mean_val, 3), round(std_val, 3)

    mean_dwell, std_dwell = safe_stats(dwell_times)
    mean_flight, std_flight = safe_stats(flight_times)
    mean_dd, std_dd = safe_stats(down_down_times)

    feature_vector = [
        mean_dwell,
        std_dwell,
        mean_flight,
        std_flight,
        mean_dd,
        std_dd,
        round(typing_speed, 2),
        round(error_rate, 4)
    ]

    return feature_vector


def extract_features_dict(raw_events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Convenience function returning a keyed dictionary of extracted features.
    """
    vec = extract_features(raw_events)
    return dict(zip(FEATURE_NAMES, vec))
