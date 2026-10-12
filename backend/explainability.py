"""
Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine
Explainability (XAI) Diagnostics Layer

Analyzes why a live keystroke sample scored genuine, uncertain, or suspicious
by computing feature-level z-score deviations against the enrolled baseline profile
and generating human-readable diagnostic explanations.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from database import get_user_samples
from feature_extraction import FEATURE_NAMES

# Friendly human-readable names
FEATURE_DISPLAY_NAMES = {
    "mean_dwell": "key-hold duration",
    "std_dwell": "key-hold consistency",
    "mean_flight": "gap between keystrokes",
    "std_flight": "flight time variance",
    "mean_down_down": "inter-key rhythm cadence",
    "std_down_down": "typing rhythm consistency",
    "typing_speed": "typing speed",
    "error_rate": "correction / backspace frequency"
}

FEATURE_UNITS = {
    "mean_dwell": "ms",
    "std_dwell": "ms",
    "mean_flight": "ms",
    "std_flight": "ms",
    "mean_down_down": "ms",
    "std_down_down": "ms",
    "typing_speed": "CPM",
    "error_rate": "ratio"
}

# Domain-aware standard deviation floors based on physical human variability
# Prevents micro-variances (e.g. 0.2ms std) from producing astronomical z-scores
FEATURE_STD_FLOORS = {
    "mean_dwell": 15.0,       # Min 15ms variability in human keypress hold
    "std_dwell": 5.0,         # Min 5ms variability in hold variance
    "mean_flight": 20.0,      # Min 20ms variability in finger travel gap
    "std_flight": 10.0,       # Min 10ms variability in flight variance
    "mean_down_down": 25.0,   # Min 25ms variability in inter-key cadence
    "std_down_down": 10.0,    # Min 10ms variability in cadence variance
    "typing_speed": 25.0,     # Min 25 CPM variability in typing speed
    "error_rate": 0.05        # Min 0.05 floor for error rate
}


def explain_deviation(user_id: int, live_feature_vector: List[float]) -> Dict[str, Any]:
    """
    Compares live feature vector against the user's enrollment baseline.
    
    Parameters:
        user_id: Database user_id
        live_feature_vector: 8-dimensional numeric list
        
    Returns:
        Dict containing:
            - summary: Human-readable sentence explaining top deviations
            - top_deviations: List of top 2-3 feature deviations with metrics
            - all_deviations: Full feature breakdown dictionary
    """
    # 1. Fetch enrollment feature vectors
    enrollment_samples = get_user_samples(user_id, sample_type="enrollment")
    if not enrollment_samples:
        return {
            "summary": "No baseline enrollment profile available for comparison.",
            "top_deviations": [],
            "all_deviations": []
        }

    X_enroll = np.array(enrollment_samples, dtype=np.float64)
    live_vec = np.array(live_feature_vector, dtype=np.float64)

    # 2. Compute baseline statistics (mean and std) per feature
    profile_mean = np.mean(X_enroll, axis=0)
    profile_std = np.std(X_enroll, axis=0)

    # 3. Compute z-score and percentage deviation for each feature with domain-aware floors
    deviations = []
    for i, name in enumerate(FEATURE_NAMES):
        base_m = float(profile_mean[i])
        raw_s = float(profile_std[i])
        live_v = float(live_vec[i])

        # Apply domain-aware minimum standard deviation floor
        floor_s = FEATURE_STD_FLOORS.get(name, 10.0)
        effective_std = max(raw_s, floor_s)

        # Compute z-score and bound to realistic range [-15.0, 15.0]
        raw_z = (live_v - base_m) / effective_std
        z_score = float(np.clip(raw_z, -15.0, 15.0))
        abs_z = abs(z_score)

        pct_diff = 0.0
        if abs(base_m) > 1e-4:
            pct_diff = ((live_v - base_m) / abs(base_m)) * 100.0

        unit = FEATURE_UNITS.get(name, "")
        display_name = FEATURE_DISPLAY_NAMES.get(name, name)

        # Precise Directional Wording Audit:
        # - For time intervals (dwell, flight, down_down):
        #   Higher value = SLOWER / LONGER. Lower value = FASTER / SHORTER.
        # - For typing_speed (CPM):
        #   Higher value = FASTER. Lower value = SLOWER.
        # - For error_rate:
        #   Higher value = HIGHER ERROR RATE. Lower value = LOWER ERROR RATE.
        # - For variances (std_*):
        #   Higher value = MORE ERRATIC / LESS CONSISTENT. Lower value = MORE CONSISTENT.
        if name in ("mean_dwell", "mean_flight", "mean_down_down"):
            if pct_diff > 0:
                dir_label = "slower / longer"
                summary_phrase = f"{display_name} is {abs(pct_diff):.1f}% longer/slower than baseline"
            else:
                dir_label = "faster / shorter"
                summary_phrase = f"{display_name} is {abs(pct_diff):.1f}% shorter/faster than baseline"
        elif name == "typing_speed":
            if pct_diff > 0:
                dir_label = "faster"
                summary_phrase = f"typing speed is {abs(pct_diff):.1f}% faster than usual"
            else:
                dir_label = "slower"
                summary_phrase = f"typing speed is {abs(pct_diff):.1f}% slower than usual"
        elif name == "error_rate":
            if pct_diff > 0:
                dir_label = "higher"
                summary_phrase = f"correction/backspace frequency increased by {abs(pct_diff):.1f}%"
            else:
                dir_label = "lower"
                summary_phrase = f"correction/backspace frequency decreased by {abs(pct_diff):.1f}%"
        else: # std_dwell, std_flight, std_down_down
            if pct_diff > 0:
                dir_label = "more erratic / irregular"
                summary_phrase = f"{display_name} is {abs(pct_diff):.1f}% more erratic than baseline"
            else:
                dir_label = "more consistent / steady"
                summary_phrase = f"{display_name} is {abs(pct_diff):.1f}% more consistent than baseline"

        description = (
            f"{abs(pct_diff):.1f}% {dir_label} than baseline "
            f"({live_v:.1f}{unit} vs normal {base_m:.1f}{unit})"
        )

        deviations.append({
            "feature_key": name,
            "feature_name": display_name,
            "baseline_val": round(base_m, 2),
            "live_val": round(live_v, 2),
            "unit": unit,
            "z_score": round(z_score, 2),
            "abs_z": round(abs_z, 2),
            "percentage_diff": round(pct_diff, 1),
            "description": description,
            "summary_phrase": summary_phrase
        })

    # 4. Sort by absolute z-score descending (largest deviation first)
    deviations_sorted = sorted(deviations, key=lambda d: d["abs_z"], reverse=True)
    top_3 = deviations_sorted[:3]

    # 5. Generate human-readable summary sentence
    significant_devs = [d for d in top_3 if d["abs_z"] > 1.2 or abs(d["percentage_diff"]) > 25.0]

    if not significant_devs:
        summary_sentence = (
            "Typing dynamics align tightly with your enrolled biometric baseline profile; "
            "motor rhythm and keystroke velocity are within normal physiological bounds."
        )
    else:
        phrases = [dev["summary_phrase"] for dev in significant_devs[:2]]
        summary_sentence = "Noticeable divergence detected: " + " and ".join(phrases) + "."

    return {
        "summary": summary_sentence,
        "top_deviations": top_3,
        "all_deviations": deviations_sorted
    }
