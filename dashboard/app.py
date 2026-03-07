import streamlit as st
import numpy as np
import pandas as pd
import time
import joblib
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

from scipy.stats import kurtosis
from scipy.fft import fft


#  MODEL LOADING

model   = joblib.load("models/predictive_maintenance_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")


#  PAGE CONFIG

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("⚙️ AI Predictive Maintenance Dashboard")
st.write("Real-Time Industrial Motor Monitoring")


#  SIDEBAR — Simulation & Alert Settings

st.sidebar.header("🛠️ Motor Settings")
fault_mode = st.sidebar.selectbox(
    "Simulate Motor Condition",
    ["normal", "imbalance", "misalignment", "bearing_fault"]
)

st.sidebar.markdown("---")
st.sidebar.header("📧 Email Alert Settings")

alert_enabled   = st.sidebar.toggle("Enable Email Alerts", value=False)
sender_email    = st.sidebar.text_input("Sender Email (Gmail)",  placeholder="you@gmail.com")
sender_password = st.sidebar.text_input("App Password",          type="password", placeholder="Gmail App Password")
receiver_email  = st.sidebar.text_input("Recipient Email",       placeholder="engineer@plant.com")

# How many consecutive fault cycles before firing an alert
alert_threshold = st.sidebar.slider(
    "Alert after N consecutive faults", min_value=1, max_value=10, value=3
)

st.sidebar.markdown("---")
st.sidebar.header("⚠️ Health Thresholds")
warn_threshold     = st.sidebar.slider("Warning below (%)",  min_value=10, max_value=90, value=60)
critical_threshold = st.sidebar.slider("Critical below (%)", min_value=10, max_value=80, value=35)


#  SESSION STATE

defaults = {
    "vibration_data":     [],
    "temperature_data":   [],
    "loop_counter":       0,
    "consecutive_faults": 0,    # counts back-to-back fault detections
    "last_alert_time":    None, # datetime of last sent alert (avoids spam)
    "alert_log":          [],   # list of dicts shown in the log table
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

WINDOW           = 512  # rolling chart data points to keep
ALERT_COOLDOWN_S = 60   # minimum seconds between repeated email alerts


#  EMAIL ALERT FUNCTION

def send_alert_email(fault_type: str, health_score: float, probabilities, labels):
    """
    Sends an HTML alert email via Gmail SMTP (SSL, port 465).
    Requires a Gmail App Password — not the regular account password.
    Returns (success: bool, message: str).
    """
    if not all([sender_email, sender_password, receiver_email]):
        return False, "Email credentials incomplete."

    prob_rows = "".join(
        f"<tr><td>{l}</td><td>{p * 100:.1f}%</td></tr>"
        for l, p in zip(labels, probabilities)
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <html><body>
    <h2 style="color:#c0392b;">⚠️ Motor Fault Alert</h2>
    <p><b>Timestamp:</b> {timestamp}</p>
    <p><b>Detected Fault:</b> <span style="color:red;">{fault_type.upper()}</span></p>
    <p><b>Motor Health Score:</b> {health_score:.1f}%</p>
    <h3>Fault Probabilities</h3>
    <table border="1" cellpadding="6" style="border-collapse:collapse;">
      <tr><th>Fault Type</th><th>Probability</th></tr>
      {prob_rows}
    </table>
    <br>
    <p style="color:grey;font-size:12px;">Sent by AI Predictive Maintenance Dashboard</p>
    </body></html>
    """

    msg            = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 Motor Fault: {fault_type.upper()} — Health {health_score:.1f}%"
    msg["From"]    = sender_email
    msg["To"]      = receiver_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True, f"Alert sent at {timestamp}"
    except Exception as e:
        return False, str(e)



#  SIGNAL SIMULATION

def generate_sensor_window(mode):
    signal_length = 256
    t = np.linspace(0, 1, signal_length)

    vibration   = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(signal_length)
    current     = 5   + 0.2 * np.random.randn(signal_length)
    rpm         = 1500 + 5  * np.random.randn(signal_length)
    temperature = 60  + np.random.normal(0, 2, signal_length)

    if mode == "imbalance":
        vibration   += 1.0 * np.sin(2 * np.pi * 50 * t)
        current     += 1.2 * np.sin(2 * np.pi * 50 * t)
        temperature += np.random.uniform(5, 10)

    elif mode == "misalignment":
        vibration   += 0.8 * np.sin(2 * np.pi * 100 * t)
        vibration   += 0.5 * np.sin(2 * np.pi * 150 * t)
        current     += 0.5 * np.sin(2 * np.pi * 100 * t)
        rpm         += np.random.uniform(-20, 20)
        temperature += np.random.uniform(8, 15)

    elif mode == "bearing_fault":
        impulses = np.zeros(signal_length)
        impulse_positions = np.random.choice(signal_length, 6)
        impulses[impulse_positions] = np.random.uniform(2, 4, 6)
        vibration   += impulses
        current     += 0.3 * np.random.randn(signal_length)
        temperature += np.random.uniform(10, 20)

    return vibration, current, rpm, temperature



#  FEATURE EXTRACTION

def extract_features(vibration, current, rpm, temperature):
    vibration_rms             = np.sqrt(np.mean(vibration ** 2))
    vibration_kurtosis        = kurtosis(vibration)
    vibration_fft             = np.abs(fft(vibration))[:len(vibration) // 2]
    vibration_spectral_energy = np.sum(vibration_fft ** 2)
    freqs                     = np.fft.fftfreq(len(vibration))[:len(vibration) // 2]
    dominant_frequency        = freqs[np.argmax(vibration_fft)]

    current_rms               = np.sqrt(np.mean(current ** 2))
    current_kurtosis          = kurtosis(current)
    current_fft               = np.abs(fft(current))[:len(current) // 2]
    current_spectral_energy   = np.sum(current_fft ** 2)

    rpm_std                   = np.std(rpm)
    temperature_mean          = np.mean(temperature)

    return [
        vibration_rms, vibration_kurtosis, vibration_spectral_energy,
        dominant_frequency, current_rms, current_kurtosis,
        current_spectral_energy, rpm_std, temperature_mean
    ]


FEATURE_COLUMNS = [
    "vibration_rms", "vibration_kurtosis", "vibration_spectral_energy",
    "dominant_frequency", "current_rms", "current_kurtosis",
    "current_spectral_energy", "rpm_std", "temperature"
]


#  LAYOUT PLACEHOLDERS

col1, col2 = st.columns(2)
vibration_chart = col1.empty()
temp_chart      = col2.empty()

st.markdown("---")
gauge_col, prob_col    = st.columns([1, 1])
health_placeholder     = gauge_col.empty()
prob_placeholder       = prob_col.empty()

st.markdown("---")
status_placeholder = st.empty()
alert_placeholder  = st.empty()

st.markdown("---")
st.subheader("📋 Alert Log")
log_placeholder = st.empty()



#  MAIN LOOP

while True:
    st.session_state.loop_counter += 1
    n = st.session_state.loop_counter

    vibration, current, rpm, temperature = generate_sensor_window(fault_mode)

    # Rolling chart window
    st.session_state.vibration_data.extend(vibration.tolist())
    st.session_state.temperature_data.extend(temperature.tolist())
    st.session_state.vibration_data   = st.session_state.vibration_data[-WINDOW:]
    st.session_state.temperature_data = st.session_state.temperature_data[-WINDOW:]

    vibration_chart.line_chart(
        pd.DataFrame(st.session_state.vibration_data, columns=["vibration"]),
        width="stretch"
    )
    temp_chart.line_chart(
        pd.DataFrame(st.session_state.temperature_data, columns=["temperature"]),
        width="stretch"
    )

    # Model inference
    features      = extract_features(vibration, current, rpm, temperature)
    X             = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction    = model.predict(X)
    probabilities = model.predict_proba(X)[0]
    labels        = encoder.classes_
    fault         = encoder.inverse_transform(prediction)[0]

    # Health Score 
    # Health = probability the motor is NORMAL
   
    normal_index = list(labels).index("normal")
    health_score = probabilities[normal_index] * 100

    # Gauge colour tracks actual health level
    if health_score >= warn_threshold:
        bar_color = "#27ae60"   # green  — healthy
    elif health_score >= critical_threshold:
        bar_color = "#f39c12"   # orange — warning
    else:
        bar_color = "#c0392b"   # red    — critical

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        delta={"reference": 100, "valueformat": ".1f"},  # shows drop from 100%
        title={"text": "Motor Health (%)"},
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": bar_color},
            "steps": [
                {"range": [0,                  critical_threshold], "color": "#fadbd8"},
                {"range": [critical_threshold, warn_threshold],     "color": "#fef9e7"},
                {"range": [warn_threshold,     100],                "color": "#eafaf1"},
            ],
            "threshold": {
                "line":      {"color": "black", "width": 3},
                "thickness": 0.8,
                "value":     warn_threshold,   # black line marks the warning boundary
            }
        }
    ))
    fig.update_layout(margin=dict(t=60, b=0))
    health_placeholder.plotly_chart(fig, width="stretch", key=f"health_gauge_{n}")

    # Consecutive Fault Counter & Email Alert ───────────────
    if fault != "normal":
        st.session_state.consecutive_faults += 1
    else:
        st.session_state.consecutive_faults = 0   # reset on any clean reading

    # Check all conditions for sending an alert
    should_alert = (
        alert_enabled
        and fault != "normal"
        and st.session_state.consecutive_faults >= alert_threshold
    )

    # Enforce cooldown — prevent inbox flooding on persistent faults
    if should_alert and st.session_state.last_alert_time is not None:
        elapsed = (datetime.now() - st.session_state.last_alert_time).total_seconds()
        if elapsed < ALERT_COOLDOWN_S:
            should_alert = False

    if should_alert:
        success, msg = send_alert_email(fault, health_score, probabilities, labels)
        timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = {
            "Time":       timestamp,
            "Fault":      fault.upper(),
            "Health (%)": f"{health_score:.1f}",
            "Email Sent": "✅ Yes" if success else f"❌ {msg}",
        }
        st.session_state.alert_log.insert(0, log_entry)   # newest at top
        st.session_state.last_alert_time    = datetime.now()
        st.session_state.consecutive_faults = 0            # reset after alert fires

        if success:
            alert_placeholder.success(f"📧 Alert email sent — {msg}")
        else:
            alert_placeholder.warning(f"📧 Email failed: {msg}")
    else:
        alert_placeholder.empty()

    # Fault probability bar chart
    prob_df = pd.DataFrame({"Fault": labels, "Probability": probabilities})
    prob_placeholder.bar_chart(prob_df.set_index("Fault"), width="stretch")

    # Status banner — also shows fault counter progress
    if fault == "normal":
        status_placeholder.success(
            f"✅  Motor Status: NORMAL  |  Health: {health_score:.1f}%  |  Consecutive faults: 0"
        )
    else:
        status_placeholder.error(
            f"⚠️  Fault Detected: {fault.upper()}  |  Health: {health_score:.1f}%  "
            f"|  Consecutive faults: {st.session_state.consecutive_faults} / {alert_threshold}"
        )

    # Alert log table
    if st.session_state.alert_log:
        log_placeholder.dataframe(
            pd.DataFrame(st.session_state.alert_log),
            use_container_width=True,
            hide_index=True
        )
    else:
        log_placeholder.info("No alerts fired yet.")

    time.sleep(2)
