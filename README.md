# Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine

An AI-driven behavioral biometric authentication and continuous anomaly detection module that verifies user identity by typing cadence and rhythm rather than typed content.

---

## 1. Zero-Knowledge Privacy Architecture
- **No Character Surveillance:** Only physical key codes, event types (`keydown` / `keyup`), and high-precision microsecond timestamps are recorded.
- **Zero Content Storage:** Raw text strings and characters typed are never stored, transmitted, or inspected.

---

## 2. Statistical Feature Vector (8-Dimensional)
Each keystroke sample produces a fixed-size numeric feature vector:
1. `mean_dwell`: Average key-hold duration ($T_{keyup} - T_{keydown}$) in milliseconds.
2. `std_dwell`: Standard deviation of key-hold duration.
3. `mean_flight`: Average flight time ($T_{keydown, i+1} - T_{keyup, i}$) in milliseconds. Negative values are preserved to model fast rollover typing!
4. `std_flight`: Standard deviation of flight time.
5. `mean_down_down`: Average time interval between consecutive keydown events.
6. `std_down_down`: Standard deviation of down-down interval cadence.
7. `typing_speed`: Overall typing velocity in Characters/Keystrokes Per Minute (CPM).
8. `error_rate`: Proportion of correction keypresses (Backspace / Delete).

---

## 3. Machine Learning Models & Decision Logic
- **Algorithms:** Dual-model training using **One-Class Support Vector Machine (OneClassSVM)** and **Isolation Forest**.
- **Preprocessing:** All dimensions are scaled with `StandardScaler` to prevent feature magnitude bias.
- **Calibrated Confidence Thresholds:**
  - **Confidence > 85%:** `GENUINE` — Grant Access.
  - **Confidence 30% – 85%:** `UNCERTAIN` — Trigger Inline Step-Up Verification (Password Prompt).
  - **Confidence < 30%:** `SUSPICIOUS` — Block Access.

---

## 4. API Endpoint Documentation

### `POST /enroll`
Accepts 3–5 typing samples and trains One-Class SVM and Isolation Forest models.

**Request:**
```json
{
  "username": "sarah_connor",
  "samples": [
    [
      {"key_code": 84, "event_type": "keydown", "timestamp_ms": 100.0},
      {"key_code": 84, "event_type": "keyup", "timestamp_ms": 190.0}
    ]
  ]
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Biometric profile enrolled successfully for user 'sarah_connor'.",
  "user_id": 1,
  "samples_count": 4,
  "total_samples_stored": 4,
  "models_trained": ["ocsvm", "isolation_forest"]
}
```

---

### `POST /verify`
Verifies live keystroke dynamics against the enrolled biometric baseline.

**Request:**
```json
{
  "username": "sarah_connor",
  "sample": [
    {"key_code": 84, "event_type": "keydown", "timestamp_ms": 100.0},
    {"key_code": 84, "event_type": "keyup", "timestamp_ms": 185.0}
  ],
  "model_type": "ocsvm"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "username": "sarah_connor",
  "verdict": "genuine",
  "confidence_score": 93.4,
  "raw_anomaly_score": 0.0125,
  "model_type": "ocsvm",
  "requires_step_up": false,
  "explanation_summary": "Typing dynamics align tightly with your enrolled biometric baseline profile.",
  "deviations": [
    {
      "feature_name": "typing speed",
      "baseline_val": 340.2,
      "live_val": 345.1,
      "percentage_diff": 1.4,
      "description": "1.4% faster than baseline (345.1CPM vs normal 340.2CPM)"
    }
  ]
}
```

---

### `POST /verify/step-up`
Fallback password verification invoked when biometric score is uncertain (30% – 85%).

**Request:**
```json
{
  "username": "sarah_connor",
  "password": "mySecurePassword123"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "step_up_passed": true,
  "message": "Step-up credential verified. Access authorized."
}
```

---

### `POST /continuous-check`
Periodically invoked during active work to detect sudden mid-session identity drift (>25% drop).

**Request:**
```json
{
  "username": "sarah_connor",
  "sample": [ ... ],
  "previous_confidence": 92.0
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "verdict": "genuine",
  "confidence_score": 89.5,
  "previous_confidence": 92.0,
  "confidence_delta": 2.5,
  "alert": false,
  "session_anomaly_flag": false,
  "message": "Typing rhythm remains consistent with active session baseline."
}
```

---

## 5. Starting the Local System

1. **Start FastAPI Backend:**
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
2. **Open Frontend:**
   Open `frontend/index.html` in any web browser or serve it via a local static file server.
