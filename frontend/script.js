/**
 * Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine
 * Frontend Keystroke Capture, Telemetry & Biometric Client Logic
 *
 * ZERO-KNOWLEDGE PRINCIPLE:
 * Strictly logs: { key_code, event_type, timestamp_ms }
 * Never captures, logs, or stores typed characters or string content.
 */

const API_BASE_URL = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
  ? 'http://127.0.0.1:8000'
  : 'http://127.0.0.1:8000';

// Reference Prompt Library for Enrollment
const PROMPT_SENTENCES = [
  "The quick brown fox jumps over the lazy dog to verify cybersecurity patterns.",
  "Keystroke dynamics establish non-intrusive biometric identity verification.",
  "Digital DNA biometrics leverage neurological motor control typing rhythms.",
  "Advanced anomaly detection safeguards digital systems without surveillance."
];

let currentPromptIndex = 0;

// State management
let enrollSamples = []; // Array of arrays: [ [event1, event2, ...], [event1, ...] ]
let currentEnrollEvents = [];
let currentVerifyEvents = [];
let currentContinuousEvents = [];
let continuousMonitoringActive = false;
let lastContinuousConfidence = null;

// ============================================================================
// 1. KEYSTROKE CAPTURE ENGINE
// ============================================================================

/**
 * Creates a keystroke recorder attached to a DOM input element.
 * Strictly adheres to privacy constraints: zero character recording.
 */
function attachKeystrokeRecorder(elementId, onEventCallback) {
  const element = document.getElementById(elementId);
  if (!element) return;

  const handleEvent = (e) => {
    // Collect ONLY the hardware/virtual key code and microsecond timestamp
    const rawEvent = {
      key_code: e.keyCode || e.which || (e.code ? e.code.charCodeAt(0) : 0),
      event_type: e.type, // 'keydown' or 'keyup'
      timestamp_ms: Math.round(performance.now() * 100) / 100
    };

    onEventCallback(rawEvent);
  };

  element.addEventListener('keydown', handleEvent);
  element.addEventListener('keyup', handleEvent);
}

// Calculate instantaneous Typing Speed (Keys Per Minute)
function computeLiveCPM(events) {
  if (events.length < 4) return 0;
  const downEvents = events.filter(e => e.event_type === 'keydown');
  if (downEvents.length < 2) return 0;

  const durationMs = downEvents[downEvents.length - 1].timestamp_ms - downEvents[0].timestamp_ms;
  if (durationMs <= 0) return 0;

  const minutes = durationMs / 60000;
  return Math.round(downEvents.length / minutes);
}

// Visual pulse indicator in live telemetry stream
function appendPulseNode(containerId, event) {
  const stream = document.getElementById(containerId);
  if (!stream) return;

  const placeholder = stream.querySelector('.placeholder-stream');
  if (placeholder) {
    placeholder.remove();
  }

  const node = document.createElement('div');
  const isDown = event.event_type === 'keydown';
  node.className = `pulse-node ${isDown ? 'keydown-node' : 'keyup-node'}`;
  // Show key_code and event type only - NO character!
  node.textContent = `${isDown ? 'KD' : 'KU'}:${event.key_code}`;

  stream.appendChild(node);

  // Keep stream scrolled to right and limit visible nodes
  stream.scrollLeft = stream.scrollWidth;
  if (stream.children.length > 25) {
    stream.removeChild(stream.firstElementChild);
  }
}

// ============================================================================
// 2. INITIALIZATION & LISTENERS
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Attach capture to Enrollment input
  attachKeystrokeRecorder('enrollInput', (event) => {
    currentEnrollEvents.push(event);
    appendPulseNode('enrollRhythmStream', event);
    const cpm = computeLiveCPM(currentEnrollEvents);
    document.getElementById('enrollHud').textContent = 
      `Events: ${currentEnrollEvents.length} | Live CPM: ${cpm}`;
  });

  // Attach capture to Verification input
  attachKeystrokeRecorder('verifyInput', (event) => {
    currentVerifyEvents.push(event);
    appendPulseNode('verifyRhythmStream', event);
    const cpm = computeLiveCPM(currentVerifyEvents);
    document.getElementById('verifyHud').textContent = 
      `Events: ${currentVerifyEvents.length} | Live CPM: ${cpm}`;
  });

  // Attach capture to Continuous Monitoring input
  attachKeystrokeRecorder('continuousInput', (event) => {
    if (!continuousMonitoringActive) return;

    currentContinuousEvents.push(event);
    const keydowns = currentContinuousEvents.filter(e => e.event_type === 'keydown');
    document.getElementById('hudKeyCount').textContent = `${keydowns.length} keys`;

    // Trigger interval check every 25 keydowns
    if (keydowns.length >= 25) {
      dispatchContinuousCheck();
    }
  });

  // Check backend health
  checkBackendHealth();
});

// ============================================================================
// 3. UI TAB SWITCHING & PROMPTS
// ============================================================================

function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

  if (tabName === 'enroll') {
    document.getElementById('tabEnrollBtn').classList.add('active');
    document.getElementById('enrollSection').classList.add('active');
  } else if (tabName === 'verify') {
    document.getElementById('tabVerifyBtn').classList.add('active');
    document.getElementById('verifySection').classList.add('active');
  } else if (tabName === 'continuous') {
    document.getElementById('tabContinuousBtn').classList.add('active');
    document.getElementById('continuousSection').classList.add('active');
  }
}

function cyclePrompt() {
  currentPromptIndex = (currentPromptIndex + 1) % PROMPT_SENTENCES.length;
  document.getElementById('enrollPromptText').textContent = PROMPT_SENTENCES[currentPromptIndex];
}

// ============================================================================
// 4. ENROLLMENT WORKFLOW
// ============================================================================

function recordCurrentSample() {
  const username = document.getElementById('enrollUsername').value.trim();
  const msgEl = document.getElementById('enrollMessage');

  if (!username) {
    showFeedback(msgEl, 'Please enter a username first.', 'error');
    return;
  }

  const keydownCount = currentEnrollEvents.filter(e => e.event_type === 'keydown').length;
  if (keydownCount < 15) {
    showFeedback(msgEl, `Sample too short (${keydownCount} keys). Please type the reference sentence naturally.`, 'error');
    return;
  }

  // Save sample copy
  enrollSamples.push([...currentEnrollEvents]);
  currentEnrollEvents = [];
  document.getElementById('enrollInput').value = '';
  document.getElementById('enrollRhythmStream').innerHTML = 
    '<div class="placeholder-stream">Sample recorded! Type the next prompt naturally...</div>';
  document.getElementById('enrollHud').textContent = 'Events: 0 | Live CPM: 0';

  updateEnrollmentUI();
  cyclePrompt();

  showFeedback(msgEl, `Sample ${enrollSamples.length} captured successfully!`, 'success');
}

function updateEnrollmentUI() {
  const count = enrollSamples.length;
  const target = 4;
  const pct = Math.min(100, Math.round((count / target) * 100));

  document.getElementById('sampleCounter').textContent = count;
  document.getElementById('sampleBtnCount').textContent = count;
  document.getElementById('sampleProgressBar').style.width = `${pct}%`;

  for (let i = 1; i <= target; i++) {
    const pill = document.getElementById(`pill-${i}`);
    if (i <= count) {
      pill.className = 'sample-pill completed';
      pill.textContent = `✓ Sample ${i}`;
    } else if (i === count + 1) {
      pill.className = 'sample-pill current';
      pill.textContent = `Sample ${i}`;
    } else {
      pill.className = 'sample-pill';
      pill.textContent = `Sample ${i}`;
    }
  }

  const submitBtn = document.getElementById('submitEnrollBtn');
  if (count >= target) {
    submitBtn.disabled = false;
  } else {
    submitBtn.disabled = true;
  }
}

function resetEnrollmentSamples() {
  enrollSamples = [];
  currentEnrollEvents = [];
  document.getElementById('enrollInput').value = '';
  document.getElementById('enrollRhythmStream').innerHTML = 
    '<div class="placeholder-stream">Start typing to see raw timing pulses (dwell & flight vectors)...</div>';
  document.getElementById('enrollHud').textContent = 'Events: 0 | Live CPM: 0';
  updateEnrollmentUI();
  const msgEl = document.getElementById('enrollMessage');
  msgEl.className = 'feedback-message';
}

async function submitEnrollment() {
  const username = document.getElementById('enrollUsername').value.trim();
  const msgEl = document.getElementById('enrollMessage');
  const submitBtn = document.getElementById('submitEnrollBtn');

  if (!username) {
    showFeedback(msgEl, 'Please provide a username.', 'error');
    return;
  }
  if (enrollSamples.length < 3) {
    showFeedback(msgEl, 'At least 3-4 typing samples are required to train the model.', 'error');
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = 'Extracting Features & Training Models...';

  try {
    const response = await fetch(`${API_BASE_URL}/enroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: username,
        samples: enrollSamples
      })
    });

    const data = await response.json();

    if (response.ok && data.status === 'success') {
      showFeedback(msgEl, `Biometric DNA successfully enrolled! Trained One-Class SVM & Isolation Forest on ${data.samples_count} feature vectors.`, 'success');
      // Autofill verification username for convenience
      document.getElementById('verifyUsername').value = username;
      document.getElementById('continuousUsername').value = username;
    } else {
      showFeedback(msgEl, `Enrollment Error: ${data.detail || 'Unable to enroll user.'}`, 'error');
    }
  } catch (err) {
    console.error('Enrollment fetch error:', err);
    showFeedback(msgEl, `Network Error: Cannot connect to backend (${err.message}). Is FastAPI running?`, 'error');
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path><polyline points="17 21 17 13 7 13 7 21"></polyline><polyline points="7 3 7 8 15 8"></polyline></svg>
      Train & Save Biometric Models`;
  }
}

// ============================================================================
// 5. VERIFICATION WORKFLOW & EXPLAINABILITY
// ============================================================================

async function submitVerification() {
  const username = document.getElementById('verifyUsername').value.trim();
  const modelType = document.getElementById('modelSelect').value;
  const resultCard = document.getElementById('verifyResultCard');
  const verifyBtn = document.getElementById('verifySubmitBtn');

  if (!username) {
    alert('Please enter a registered username to verify.');
    return;
  }

  const keydownCount = currentVerifyEvents.filter(e => e.event_type === 'keydown').length;
  if (keydownCount < 12) {
    alert(`Verification sample is too short (${keydownCount} keys). Please type the reference sentence.`);
    return;
  }

  verifyBtn.disabled = true;
  verifyBtn.textContent = 'Evaluating Biometrics...';

  try {
    const response = await fetch(`${API_BASE_URL}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: username,
        sample: currentVerifyEvents,
        model_type: modelType
      })
    });

    const data = await response.json();

    if (response.ok && data.status === 'success') {
      displayVerificationResult(data);
    } else {
      alert(`Verification failed: ${data.detail || 'Error communicating with backend'}`);
    }
  } catch (err) {
    console.error('Verification error:', err);
    alert(`Network Error: ${err.message}. Ensure backend FastAPI server is running.`);
  } finally {
    verifyBtn.disabled = false;
    verifyBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>
      Verify Biometric Signature`;
  }
}

function displayVerificationResult(result) {
  const card = document.getElementById('verifyResultCard');
  card.classList.remove('hidden');

  const badge = document.getElementById('verdictBadge');
  const bar = document.getElementById('confidenceBar');
  const scoreText = document.getElementById('confidencePercentage');
  const modelTag = document.getElementById('resultModelTag');
  const stepUp = document.getElementById('stepUpAuthContainer');
  const summaryEl = document.getElementById('explainSummary');
  const listEl = document.getElementById('featureDivergenceList');

  const conf = Math.max(0, Math.min(100, result.confidence_score));
  scoreText.textContent = `${conf.toFixed(1)}%`;
  bar.style.width = `${conf}%`;
  modelTag.textContent = `Model: ${result.model_type === 'ocsvm' ? 'One-Class SVM' : 'Isolation Forest'}`;

  // Reset classes
  badge.className = 'verdict-badge';
  bar.className = 'confidence-fill';

  // Apply Decision Logic
  if (result.verdict === 'genuine') {
    badge.classList.add('verdict-genuine');
    badge.textContent = 'GENUINE';
    bar.classList.add('fill-genuine');
    stepUp.classList.add('hidden');
  } else if (result.verdict === 'uncertain') {
    badge.classList.add('verdict-uncertain');
    badge.textContent = 'UNCERTAIN';
    bar.classList.add('fill-uncertain');
    stepUp.classList.remove('hidden'); // Show inline password step-up!
  } else {
    badge.classList.add('verdict-suspicious');
    badge.textContent = 'SUSPICIOUS';
    bar.classList.add('fill-suspicious');
    stepUp.classList.add('hidden');
  }

  // Populate Explainability Section
  summaryEl.textContent = result.explanation_summary || 'Analysis complete.';
  listEl.innerHTML = '';

  if (result.deviations && result.deviations.length > 0) {
    result.deviations.forEach(dev => {
      const item = document.createElement('div');
      const isHigh = Math.abs(dev.percentage_diff) > 40;
      item.className = `feature-item ${isHigh ? 'high-dev' : 'med-dev'}`;
      item.innerHTML = `
        <span class="feature-name">${dev.feature_name}</span>
        <span class="feature-stat">${dev.description} (Baseline: ${dev.baseline_val.toFixed(1)}, Live: ${dev.live_val.toFixed(1)})</span>
      `;
      listEl.appendChild(item);
    });
  }

  // Scroll into view smoothly
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function clearVerifyField() {
  document.getElementById('verifyInput').value = '';
  currentVerifyEvents = [];
  document.getElementById('verifyRhythmStream').innerHTML = '<div class="placeholder-stream">Awaiting keystrokes...</div>';
  document.getElementById('verifyHud').textContent = 'Events: 0 | Live CPM: 0';
  document.getElementById('verifyResultCard').classList.add('hidden');
}

function submitStepUp() {
  const pwd = document.getElementById('stepUpPassword').value;
  const fb = document.getElementById('stepUpFeedback');
  if (!pwd) {
    showFeedback(fb, 'Please enter password to complete step-up authentication.', 'error');
    return;
  }
  // Step-up verification simulation
  showFeedback(fb, 'Step-up password authenticated! Identity verified.', 'success');
}

// ============================================================================
// 6. CONTINUOUS AUTHENTICATION ENGINE
// ============================================================================

function toggleContinuousMonitoring() {
  const btn = document.getElementById('toggleContinuousBtn');
  const statusBadge = document.getElementById('sessionStatusBadge');
  const username = document.getElementById('continuousUsername').value.trim();

  if (!continuousMonitoringActive) {
    if (!username) {
      alert('Please enter enrolled username for continuous monitoring.');
      return;
    }
    continuousMonitoringActive = true;
    btn.textContent = 'Stop Continuous Session';
    btn.classList.add('btn-primary');
    btn.classList.remove('btn-secondary');
    statusBadge.textContent = 'ACTIVE MONITORING';
    statusBadge.className = 'hud-value status-active';
    document.getElementById('continuousInput').focus();
    appendTimeline('Session started. Monitoring live keystroke buffer...', '--');
  } else {
    continuousMonitoringActive = false;
    btn.textContent = 'Start Continuous Session';
    btn.classList.add('btn-secondary');
    btn.classList.remove('btn-primary');
    statusBadge.textContent = 'STANDBY';
    statusBadge.className = 'hud-value status-idle';
    document.getElementById('continuousAlertBox').classList.add('alert-hidden');
    appendTimeline('Continuous monitoring stopped.', '--');
  }
}

async function dispatchContinuousCheck() {
  const username = document.getElementById('continuousUsername').value.trim();
  const sampleToSend = [...currentContinuousEvents];
  currentContinuousEvents = []; // Flush buffer

  try {
    const response = await fetch(`${API_BASE_URL}/continuous-check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: username,
        sample: sampleToSend,
        previous_confidence: lastContinuousConfidence
      })
    });

    const data = await response.json();
    if (response.ok && data.status === 'success') {
      const conf = data.confidence_score;
      document.getElementById('hudConfidence').textContent = `${conf.toFixed(1)}%`;
      lastContinuousConfidence = conf;

      const driftEl = document.getElementById('hudDrift');
      const alertBox = document.getElementById('continuousAlertBox');

      if (data.session_anomaly_flag) {
        driftEl.textContent = 'DRIFT DETECTED';
        driftEl.style.color = '#fb7185';
        alertBox.classList.remove('alert-hidden');
        document.getElementById('alertTitle').textContent = 'Identity Drift Warning!';
        document.getElementById('alertDesc').textContent = data.message || 'Identity confidence dropped significantly mid-session.';
        appendTimeline(`⚠️ Anomaly: ${data.message}`, `${conf.toFixed(1)}%`);
      } else {
        driftEl.textContent = 'Consistent';
        driftEl.style.color = '#34d399';
        alertBox.classList.add('alert-hidden');
        appendTimeline('Biometric cadence verified consistent.', `${conf.toFixed(1)}%`);
      }
    }
  } catch (err) {
    console.warn('Continuous evaluation error:', err);
  }
}

function appendTimeline(message, score) {
  const log = document.getElementById('continuousLog');
  const empty = log.querySelector('.timeline-empty');
  if (empty) empty.remove();

  const now = new Date().toLocaleTimeString();
  const entry = document.createElement('div');
  entry.className = 'timeline-entry';
  entry.innerHTML = `
    <span class="timeline-time">${now}</span>
    <span>${message}</span>
    <span class="timeline-score">${score}</span>
  `;
  log.prepend(entry);
}

// ============================================================================
// 7. HELPER UTILITIES
// ============================================================================

function showFeedback(element, message, type) {
  element.textContent = message;
  element.className = `feedback-message show ${type === 'success' ? 'feedback-success' : 'feedback-error'}`;
}

async function checkBackendHealth() {
  const badge = document.getElementById('apiHealthBadge');
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      badge.textContent = 'FastAPI Connected';
      badge.className = 'badge-healthy';
    } else {
      badge.textContent = 'API Initializing';
      badge.style.color = '#f59e0b';
    }
  } catch (e) {
    badge.textContent = 'Backend Offline (Ready to start)';
    badge.style.color = '#94a3b8';
  }
}


