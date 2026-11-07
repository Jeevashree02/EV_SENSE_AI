import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from style import load_theme, ev_logo

try:
    from preprocessing.motor_inference import predict_motor_condition
    MODEL_IMPORT_ERROR = None
except Exception as exc:
    predict_motor_condition = None
    MODEL_IMPORT_ERROR = str(exc)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="EV-Sense AI | Motor Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_theme()


# =====================================================
# PROJECT INFORMATION
# =====================================================


CONDITIONS = {
    "H": {
        "name": "Healthy",
        "category": "Normal operating condition",
        "description": "The motor operates under normal conditions without the selected fault characteristics.",
        "causes": "Normal bearing lubrication, intact rotor structure and balanced electrical supply.",
        "symptoms": "Vibration remains consistent with the healthy reference condition. No specific fault signature is identified by the classification model.",
        "impact": "Normal operation supports efficient motor performance and reduces unnecessary mechanical stress.",
        "action": "Continue routine inspection and preventive maintenance.",
        "detection": "The Random Forest model assigns the H classification when the extracted vibration features most closely match the healthy class."
    },

    "B1": {
        "name": "Insufficient bearing lubrication",
        "category": "Bearing fault",
        "description": "The bearing receives less lubrication than required for normal operation.",
        "causes": "Inadequate lubricant supply, lubricant leakage, incorrect lubricant quantity or insufficient maintenance.",
        "symptoms": "Increased friction, unusual bearing noise, vibration changes and possible temperature rise.",
        "impact": "Continued operation may increase bearing wear and reduce bearing service life.",
        "action": "Inspect the bearing lubrication system, verify lubricant quantity and use the recommended lubricant.",
        "detection": "The model identifies vibration features associated with the B1 reference class."
    },

    "B2": {
        "name": "Severely insufficient bearing lubrication",
        "category": "Bearing fault",
        "description": "The bearing experiences a severe shortage of lubrication, resulting in increased contact friction between moving surfaces.",
        "causes": "Prolonged lubricant loss, neglected maintenance, lubricant breakdown or serious lubrication-system problems.",
        "symptoms": "Potentially increased vibration, abnormal noise, friction and excessive heat generation.",
        "impact": "Severe lubrication deficiency can accelerate bearing degradation and may eventually cause bearing failure.",
        "action": "Inspect the bearing immediately, investigate lubricant loss and assess whether bearing replacement is necessary.",
        "detection": "The model identifies vibration features associated with the B2 reference class."
    },

    "B3": {
        "name": "Cracked bearing outer ring",
        "category": "Mechanical fault",
        "description": "A crack develops in the outer ring of the rolling-element bearing.",
        "causes": "Material fatigue, excessive loading, manufacturing defects, impact damage or prolonged mechanical stress.",
        "symptoms": "Periodic vibration disturbances, abnormal bearing noise and changes in vibration amplitude.",
        "impact": "A developing crack can affect bearing stability and may lead to further mechanical damage.",
        "action": "Inspect the bearing for physical damage and replace it if a crack is confirmed.",
        "detection": "The model identifies vibration features associated with the B3 reference class."
    },

    "R": {
        "name": "Broken rotor bar",
        "category": "Rotor fault",
        "description": "One or more conducting bars in the motor rotor become damaged or broken.",
        "causes": "Thermal stress, repeated mechanical loading, manufacturing defects or prolonged operating stress.",
        "symptoms": "Changes in motor vibration, possible torque fluctuations and abnormal operating behaviour.",
        "impact": "Rotor damage can reduce motor performance and may create additional mechanical and electrical stress.",
        "action": "Inspect rotor condition and arrange appropriate motor testing before continued operation.",
        "detection": "The model identifies vibration features associated with the R reference class."
    },

    "V": {
        "name": "Voltage imbalance",
        "category": "Electrical fault",
        "description": "The three-phase supply voltages are unequal, creating an imbalance in the electrical supply to the motor.",
        "causes": "Unequal phase loading, supply problems, loose connections or faults in electrical distribution.",
        "symptoms": "Unequal phase currents, increased heating, changes in motor vibration and reduced operating efficiency.",
        "impact": "Prolonged voltage imbalance can increase motor heating and electrical stress.",
        "action": "Measure all three supply voltages, inspect electrical connections and investigate the source of imbalance.",
        "detection": "The model identifies vibration features associated with the V reference class."
    }
}


# =====================================================
# SIMULATED VIBRATION DATA — SIX MOTOR CONDITIONS
# =====================================================

FS = 100
WINDOW_SIZE = 200

SIMULATION_LEVELS = {
    "H — Healthy Motor": "H",
    "B1 — Insufficient Bearing Lubrication": "B1",
    "B2 — Severe Bearing Lubrication": "B2",
    "B3 — Cracked Bearing Outer Ring": "B3",
    "R — Broken Rotor Bar": "R",
    "V — Voltage Imbalance": "V"
}


def generate_simulated_readings(
    level="H — Healthy Motor",
    seed=None
):

    # Handle old Low / Medium / High session values
    if level not in SIMULATION_LEVELS:
        level = "H — Healthy Motor"

    rng = np.random.default_rng(seed)

    t = np.arange(WINDOW_SIZE) / FS
    fault = SIMULATION_LEVELS[level]

    rng = np.random.default_rng(seed)

    t = np.arange(WINDOW_SIZE) / FS

    fault = SIMULATION_LEVELS[level]

    # Base vibration signals
    x = 0.025 * np.sin(2 * np.pi * 12 * t)
    y = 0.020 * np.sin(2 * np.pi * 18 * t + 0.5)
    z = 1.0 + 0.025 * np.sin(2 * np.pi * 25 * t)

    # H — Healthy motor
    if fault == "H":
        pass

    # B1 — Insufficient bearing lubrication
    elif fault == "B1":
        x += 0.07 * np.sin(2 * np.pi * 15 * t)
        y += 0.05 * np.sin(2 * np.pi * 22 * t)
        z += 0.04 * np.sin(2 * np.pi * 30 * t)

    # B2 — Severe insufficient lubrication
    elif fault == "B2":
        x += 0.14 * np.sin(2 * np.pi * 15 * t)
        y += 0.12 * np.sin(2 * np.pi * 22 * t)
        z += 0.10 * np.sin(2 * np.pi * 30 * t)

    # B3 — Cracked bearing outer ring
    elif fault == "B3":
        impacts = np.maximum(
            0,
            np.sin(2 * np.pi * 8 * t)
        ) ** 8

        x += 0.16 * impacts
        y += 0.10 * impacts
        z += 0.12 * impacts

    # R — Broken rotor bar
    elif fault == "R":
        modulation = 1 + 0.45 * np.sin(2 * np.pi * 3 * t)

        x += 0.09 * modulation * np.sin(2 * np.pi * 12 * t)
        y += 0.07 * modulation * np.sin(2 * np.pi * 18 * t)

    # V — Voltage imbalance
    elif fault == "V":
        x += 0.08 * np.sin(2 * np.pi * 10 * t)
        y += 0.12 * np.sin(2 * np.pi * 20 * t)
        z += 0.09 * np.sin(2 * np.pi * 25 * t)

    # Add independent sensor noise
    x += rng.normal(0, 0.004, WINDOW_SIZE)
    y += rng.normal(0, 0.004, WINDOW_SIZE)
    z += rng.normal(0, 0.004, WINDOW_SIZE)

    return np.column_stack((x, y, z))


# =====================================================
# AI MODEL INFERENCE
# =====================================================

def run_demo_inference(level):

    readings = generate_simulated_readings(level)

    if predict_motor_condition is None:
        raise RuntimeError(
            "Model import failed: " + str(MODEL_IMPORT_ERROR)
        )

    prediction, probabilities = predict_motor_condition(
        readings,
        fs=FS
    )

    return (
        readings,
        str(prediction),
        {
            str(label): float(score)
            for label, score in probabilities.items()
        }
    )


# =====================================================
# DEMO SESSION STATE
# =====================================================

def ensure_demo_state():

    defaults = {
        "demo_level": "Low",
        "demo_readings": None,
        "demo_prediction": None,
        "demo_probabilities": None,
        "demo_timestamp": None,
        "demo_history": [],
        "demo_error": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Always create a displayable 200 x 3 vibration window.
    if st.session_state.demo_readings is None:
        if st.session_state.demo_level not in SIMULATION_LEVELS:
           st.session_state.demo_level = "H — Healthy Motor"

        st.session_state.demo_readings = generate_simulated_readings(
    st.session_state.demo_level
)
        st.session_state.demo_timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # Inference is separate so model errors do not blank the graphs.
    if st.session_state.demo_prediction is None:
        try:
            if predict_motor_condition is None:
                raise RuntimeError(
                    "Model import failed: " + str(MODEL_IMPORT_ERROR)
                )
            prediction, probabilities = predict_motor_condition(
                st.session_state.demo_readings, fs=FS
            )
            st.session_state.demo_prediction = str(prediction)
            st.session_state.demo_probabilities = {
                str(label): float(score)
                for label, score in probabilities.items()
            }
            st.session_state.demo_error = None
            if not st.session_state.demo_history:
                st.session_state.demo_history.append({
                    "timestamp": st.session_state.demo_timestamp,
                    "level": st.session_state.demo_level,
                    "prediction": st.session_state.demo_prediction,
                    "scores": st.session_state.demo_probabilities.copy(),
                    "input_type": "DEMO"
                })
        except Exception as exc:
            st.session_state.demo_error = str(exc)


# =====================================================
# GENERATE NEW DEMO SAMPLE
# =====================================================

def generate_new_demo_sample(level):

    # Generate a fresh simulated sensor window
    readings = np.asarray(
        generate_simulated_readings(level),
        dtype=float
    )

    # Add small random variations to each reading
    rng = np.random.default_rng()

    axis_variation = rng.uniform(
        0.97, 1.03, size=(1, 3)
    )

    sensor_noise = rng.normal(
        0, 0.003, size=readings.shape
    )

    readings = readings * axis_variation + sensor_noise

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.session_state.demo_level = level
    st.session_state.demo_readings = readings
    st.session_state.demo_timestamp = timestamp
    st.session_state.demo_prediction = None
    st.session_state.demo_probabilities = {}
    st.session_state.demo_error = None

    try:
        if predict_motor_condition is None:
            raise RuntimeError(
                "Model import failed: " + str(MODEL_IMPORT_ERROR)
            )

        prediction, probabilities = predict_motor_condition(
            readings, fs=FS
        )

        st.session_state.demo_prediction = str(prediction)

        st.session_state.demo_probabilities = {
            str(label): float(score)
            for label, score in probabilities.items()
        }

        st.session_state.demo_history.append({
            "timestamp": timestamp,
            "level": level,
            "prediction": str(prediction),
            "scores": st.session_state.demo_probabilities.copy(),
            "input_type": "SIMULATED"
        })

    except Exception as exc:
        st.session_state.demo_error = str(exc)


# =====================================================
# SESSION STATE
# =====================================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

if "selected_fault" not in st.session_state:
    st.session_state.selected_fault = "H"

ensure_demo_state()


# =====================================================
# REUSABLE COMPONENTS
# =====================================================

def section_heading(title, note=""):
    st.html(f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            flex-wrap:wrap;
            gap:10px;
            margin:24px 0 14px;
        ">
            <div style="
                color:#F5F5F5;
                font-size:30px;
                font-weight:700;
            ">{title}</div>

            <div style="
                color:#888888;
                font-family:monospace;
                font-size:14px;
                letter-spacing:1px;
            ">{note}</div>
        </div>
    """)


def status_panel(label, value, description, footer, accent="cyan"):

    accent_class = "amber" if accent == "amber" else "cyan"

    st.html(f"""
        <div class="panel">
            <div class="panel-label">{label}</div>

            <div class="panel-value {accent_class}">
                {value}
            </div>

            <div class="panel-note">
                {description}
            </div>

            <div style="
                border-top:1px solid #303030;
                margin-top:14px;
                padding-top:10px;
                color:#777777;
                font-family:monospace;
                font-size:9px;
            ">
                {footer}
            </div>
        </div>
    """)


def show_fault_details(code):

    fault = CONDITIONS[code]

    st.html(f"""
        <div style="
            background:#111111;
            border:1px solid #333333;
            border-left:3px solid #00D4D8;
            border-radius:7px;
            padding:20px;
            margin-top:14px;
        ">
            <div style="
                color:#00D4D8;
                font-family:monospace;
                font-size:18px;
                letter-spacing:1px;
            ">
                CONDITION CODE / {code}
            </div>

            <div style="
                color:#F5F5F5;
                font-size:22px;
                font-weight:700;
                margin:12px 0;
            ">
                {fault["name"]}
            </div>

            <div style="
                color:#AAAAAA;
                font-size:16px;
                line-height:1.7;
            ">
                {fault["description"]}
            </div>

            <div style="
                margin-top:15px;
                padding-top:12px;
                border-top:1px solid #303030;
                color:#00D4D8;
                font-family:monospace;
                font-size:16px;
            ">
                CATEGORY / {fault["category"].upper()}
            </div>
        </div>
    """)



def show_empty_signal():

    st.html("""
    <style>
        .ev-signal-panel {
            background: #111111;
            border: 1px solid #303030;
            border-radius: 14px;
            padding: 26px;
            width: 100%;
        }

        .ev-signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 28px;
        }

        .ev-signal-title {
            color: #F5F5F5;
            font-size: 16px;
            font-family: monospace;
            letter-spacing: 1.3px;
            font-weight: 600;
        }

        .ev-signal-status {
            color: #F59E0B;
            background: #21190D;
            border: 1px solid #72501A;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 12px;
            font-family: monospace;
            font-weight: 700;
        }

        .ev-axis-row {
            display: grid;
            grid-template-columns: 35px 1fr;
            align-items: center;
            gap: 14px;
            margin-bottom: 18px;
        }

        .ev-axis-name {
            font-size: 17px;
            font-weight: 800;
            text-align: center;
        }

        .ev-axis-graph {
            height: 78px;
            border: 1px solid #383838;
            border-radius: 7px;
            background-color: #0B0B0B;
            background-image:
                linear-gradient(#242424 1px, transparent 1px),
                linear-gradient(90deg, #242424 1px, transparent 1px);
            background-size: 25px 25px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .ev-no-data {
            color: #777777;
            font-family: monospace;
            font-size: 11px;
            letter-spacing: 1.4px;
        }

        .ev-signal-footer {
            border-top: 1px solid #303030;
            margin-top: 24px;
            padding-top: 18px;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            color: #999999;
            font-family: monospace;
            font-size: 12px;
        }

        @media (max-width: 600px) {
            .ev-signal-panel {
                padding: 16px;
            }
            .ev-signal-title {
                font-size: 13px;
            }
            .ev-axis-graph {
                height: 65px;
            }
        }
    </style>

    <div class="ev-signal-panel">

        <div class="ev-signal-header">
            <div class="ev-signal-title">
                VIBRATION SIGNAL / X · Y · Z
            </div>

            <div class="ev-signal-status">
                ● WAITING FOR SENSOR
            </div>
        </div>

        <div class="ev-axis-row">
            <div class="ev-axis-name" style="color:#00D4D8;">X</div>
            <div class="ev-axis-graph">
                <span class="ev-no-data">NO DATA RECEIVED</span>
            </div>
        </div>

        <div class="ev-axis-row">
            <div class="ev-axis-name" style="color:#F59E0B;">Y</div>
            <div class="ev-axis-graph">
                <span class="ev-no-data">NO DATA RECEIVED</span>
            </div>
        </div>

        <div class="ev-axis-row">
            <div class="ev-axis-name" style="color:#A78BFA;">Z</div>
            <div class="ev-axis-graph">
                <span class="ev-no-data">NO DATA RECEIVED</span>
            </div>
        </div>

        <div class="ev-signal-footer">
            <span>INPUT: MPU6050</span>
            <span>TARGET: 100 Hz</span>
            <span>STREAM: INACTIVE</span>
        </div>

    </div>
    """)



def show_hero(number, section, title, description):

    st.html(f"""
        <div class="hero">
            <div class="eyebrow">
                EV-SENSE / {number} / {section}
            </div>

            <h1>{title}</h1>

            <p>{description}</p>
        </div>
    """)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    # Sidebar brand
    
    st.html("""
        <div class="panel" style="min-height:auto;">
            <div class="panel-label">INPUT STATUS</div>

            <div class="panel-value"
                style="font-size:18px;color:#36C98F;">
                Simulated data active
            </div>

            <div class="panel-note">
                ESP8266 NodeMCU, MPU6050 and DC motor is connected to this dashboard.
            </div>
        </div>
    """)


    navigation = {
        "MONITORING": [
            "Overview",
            "Live Monitoring",
            "Fault Analysis"
        ],
        "RECORDS": [
            "History & Reports"
        ],
        "KNOWLEDGE": [
            "Fault Guide"
        ],
        "PLATFORM": [
            "System Information",
            "About Project"
        ]
    }

    for group, pages in navigation.items():

        st.markdown(
            f'<div class="nav-title">{group}</div>',
            unsafe_allow_html=True
        )

        for page_name in pages:

            active = (
                st.session_state.current_page == page_name
            )

            if st.button(
                ("●  " if active else "○  ") + page_name,
                key=f"nav_{page_name}",
                type="primary" if active else "secondary",
                use_container_width=True
            ):
                st.session_state.current_page = page_name
                st.session_state.scroll_to_top = True
                st.rerun()

    st.divider()

    st.html("""
        <div class="panel" style="min-height:auto;">
            <div class="panel-label">HARDWARE STATUS</div>

            <div class="panel-value amber"
                 style="font-size:18px;">
                CONNECTED
            </div>

            <div class="panel-note">
                ESP8266 and MPU6050 hardware integrated
            </div>
        </div>
    """)

    st.caption("EV-SENSE AI / PROTOTYPE")




# =====================================================
# TOP HEADER — EV-SENSE CONTROL BAR
# =====================================================

header_left, header_right = st.columns(
    [1.5, 2],
    vertical_alignment="center"
)

with header_left:

    st.html(f"""
        <div class="brand" style="padding:8px 0;">
            {ev_logo(48)}

            <div>
                <div class="brand-name">
                    EV-Sense <span>AI</span>
                </div>

                <div class="brand-caption">
                    INTELLIGENT MOTOR FAULT DETECTION
                </div>
            </div>
        </div>
    """)

with header_right:

    
    st.html("""
        <div style="
            display:flex;
            justify-content:flex-end;
            align-items:center;
            flex-wrap:wrap;
            gap:8px;
        ">
            <span class="status-chip status-good">
                SIMULATED DEMO / ACTIVE
            </span>

            <span class="status-chip status-warning">
                PHYSICAL HARDWARE / CONNECTED
            </span>
        </div>
    """)


st.divider()




# =====================================================
# PAGE 1 — OVERVIEW
# =====================================================

if st.session_state.current_page == "Overview":

    show_hero(
        "01",
        "OVERVIEW",
        "Motor intelligence. Built for reliability.",
        "An intelligent vibration-based motor fault detection "
        "platform combining sensor acquisition, signal processing "
        "and machine learning."
    )

    

    section_heading(
        "System overview",
        "CURRENT BUILD / ENGINEERING PROTOTYPE"
    )

    a, b, c = st.columns(3, gap="medium")

    with a:
        status_panel(
            "MACHINE LEARNING MODEL",
            "Random Forest",
            "Existing trained classification model.",
            "INFERENCE / ACTIVE IN DEMO MODE"
        )  

    with b:
        status_panel(
            "FAULT CLASSIFICATION",
            "06 Classes",
            "Six motor-condition labels.",
            "CLASS DEFINITIONS / AVAILABLE"
        )

    with c:
        status_panel(
            "HARDWARE STATUS",
            "Connected",
            "ESP8266 NodeMCU, MPU6050 and DC motor prototype.",
            "HARDWARE / PROTOTYPE ASSEMBLED"
        )

    
    # MOTOR CONDITION LIBRARY

    section_heading(
        "Motor condition library",
        "SELECT A CONDITION TO EXPLORE"
    )

    codes = list(CONDITIONS.keys())

    
    # CLICKABLE MOTOR CONDITION CARDS

    codes = list(CONDITIONS.keys())

    
    st.html("""
    <style>

    /* CARD WRAPPER */
    div[class*="st-key-fault_wrap_"] {
        position: relative !important;
        height: 135px !important;
        min-height: 135px !important;
        margin-bottom: 0 !important;
    }

    /* VISIBLE CARD */
    div[class*="st-key-fault_wrap_"] .fault-visual-card {
        position: absolute !important;
        inset: 0 !important;
        height: 155px !important;
        box-sizing: border-box !important;

        background: #141414 !important;
        border: 1px solid #383838 !important;
        border-radius: 12px !important;

        padding: 22px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        gap: 12px !important;

        transition: all 0.2s ease !important;
        pointer-events: none !important;
    }

    /* CARD HOVER */
    div[class*="st-key-fault_wrap_"]:hover .fault-visual-card {
        border-color: #00D4D8 !important;
        background: #191919 !important;
    }

    /* SELECTED CARD */
    div[class*="st-key-fault_wrap_"] .fault-visual-card.selected {
        border: 1px solid #00D4D8 !important;
        background: #102B2B !important;
    }

    /* CONDITION CODE */
    .fault-code {
        color: #00D4D8 !important;
        font-size: 14px !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
    }

    /* CONDITION NAME */
    .fault-name {
        color: #F1F1F1 !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
    }

    /* CONDITION CATEGORY */
    .fault-category {
        color: #A3A3A3 !important;
        font-size: 11px !important;
    }

    /* INVISIBLE CLICKABLE BUTTON */
    div[class*="st-key-fault_wrap_"] div[data-testid="stButton"] {
        position: absolute !important;
        inset: 0 !important;
        z-index: 20 !important;
        width: 100% !important;
        height: 155px !important;
        margin: 0 !important;
    }

    div[class*="st-key-fault_wrap_"] div[data-testid="stButton"] button {
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 155px !important;
        min-height: 155px !important;
        opacity: 0 !important;
        cursor: pointer !important;
        border: none !important;
        background: transparent !important;
    }

    /* REMOVE BUTTON TEXT */
    div[class*="st-key-fault_wrap_"] div[data-testid="stButton"] button p {
        display: none !important;
    }

    </style>
    """)


    for row_start in range(0, len(codes), 3):

        columns = st.columns(3, gap="medium")

        for col, code in zip(
            columns,
            codes[row_start:row_start + 3]
        ):

            with col:

                fault = CONDITIONS[code]

                selected = (
                    st.session_state.selected_fault == code
                )

                with st.container(key=f"fault_wrap_{code}"):

                    st.html(f"""
                    <div class="fault-visual-card {'selected' if selected else ''}">
                        <div class="fault-code">{code}</div>
                        <div class="fault-name">{fault['name']}</div>
                        <div class="fault-category">{fault['category']}</div>
                    </div>
                    """)

                    if st.button("Select", key=f"fault_card_{code}", use_container_width=True):
                     st.session_state.selected_fault = code
                     st.session_state.scroll_to_details = True
                     st.rerun()

    st.markdown(
      '<div id="condition-details"></div>',
      unsafe_allow_html=True
    )
    # Selected condition details
    selected = st.session_state.selected_fault
    fault = CONDITIONS[selected]

    
    st.html(f"""
        <div style="
            background:#111111;
            border:1px solid #333333;
            border-left:4px solid #00D4D8;
            border-radius:8px;
            padding:25px;
            margin-top:18px;
        ">

            <div style="
                color:#00D4D8;
                font-family:monospace;
                font-size:11px;
                letter-spacing:2px;
            ">
                CONDITION DETAILS / {selected}
            </div>

            <div style="
                color:#F5F5F5;
                font-size:25px;
                font-weight:700;
                margin:12px 0 6px;
            ">
                {fault["name"]}
            </div>

            <div style="
                color:#00D4D8;
                font-size:12px;
                font-family:monospace;
                margin-bottom:20px;
            ">
                {fault["category"].upper()}
            </div>

            <div style="
                color:#BBBBBB;
                font-size:16px;
                line-height:1.8;
                margin-bottom:20px;
            ">
                {fault["description"]}
            </div>

            <div style="
                display:grid;
                grid-template-columns:repeat(2,minmax(0,1fr));
                gap:20px;
            ">

                <div>
                    <div style="color:#00D4D8;font-size:14px;font-weight:700;">
                        POSSIBLE CAUSES
                    </div>
                    <div style="color:#BBBBBB;font-size:16px;line-height:1.8;margin-top:7px;">
                        {fault["causes"]}
                    </div>
                </div>

                <div>
                    <div style="color:#00D4D8;font-size:14px;font-weight:700;">
                        OBSERVABLE SYMPTOMS
                    </div>
                    <div style="color:#BBBBBB;font-size:16px;line-height:1.8;margin-top:7px;">
                        {fault["symptoms"]}
                    </div>
                </div>

                <div>
                    <div style="color:#00D4D8;font-size:14px;font-weight:700;">
                        POTENTIAL IMPACT
                    </div>
                    <div style="color:#BBBBBB;font-size:16px;line-height:1.8;margin-top:7px;">
                        {fault["impact"]}
                    </div>
                </div>

                <div>
                    <div style="color:#00D4D8;font-size:14px;font-weight:700;">
                        RECOMMENDED ACTION
                    </div>
                    <div style="color:#BBBBBB;font-size:16px;line-height:1.8;margin-top:7px;">
                        {fault["action"]}
                    </div>
                </div>

            </div>

            <div style="
                border-top:1px solid #333333;
                margin-top:22px;
                padding-top:16px;
            ">
                <div style="
                    color:#F59E0B;
                    font-family:monospace;
                    font-size:10px;
                    letter-spacing:1px;
                ">
                    MODEL DETECTION REFERENCE
                </div>

                <div style="
                    color:#999999;
                    font-size:16px;
                    line-height:1.7;
                    margin-top:7px;
                ">
                    {fault["detection"]}
                </div>
            </div>

        </div>
    """)

    if st.session_state.get("scroll_to_details", False):

       components.html("""
           <script>
              const scrollToDetails = () => {
              const parentWindow = window.parent;
              const target = parentWindow.document.getElementById(
                 "condition-details"
            );

            if (target) {
                const position =
                    target.getBoundingClientRect().top +
                    parentWindow.scrollY - 20;

                parentWindow.scrollTo({
                    top: position,
                    behavior: "smooth"
                });
            }
        };

        setTimeout(scrollToDetails, 800);
        </script>
    """, height=10)

    st.session_state.scroll_to_details = False
    
    st.caption(
        "Technical reference information only. These descriptions "
        "are not active hardware diagnoses or confirmed fault detections."
    )


    # SYSTEM ARCHITECTURE

    section_heading("Model performance", "RECORDED VALIDATION METRICS")

    metric_a, metric_b = st.columns(2)
    metric_a.metric("Validation Accuracy", "92.8%")
    metric_b.metric("Macro F1-score", "91.6%")

    performance_fig = go.Figure()

    performance_fig.add_trace(go.Bar(
        y=["Validation Accuracy", "Macro F1-score"],
        x=[92.8, 91.6],
        orientation="h",
        marker=dict(
            color=["#00D4AA", "#38A9FF"],
            cornerradius=10
        ),
        text=["92.8%", "91.6%"],
        textposition="outside",
        textfont=dict(size=16, color="white"),
        width=0.5
    ))

    performance_fig.update_layout(
        height=300,
        paper_bgcolor="#0B1018",
        plot_bgcolor="#0B1018",
        margin=dict(l=20, r=75, t=25, b=45),
        xaxis=dict(
            title="Score (%)",
            range=[0, 105],
            dtick=20,
            ticksuffix="%",
            gridcolor="#293342",
            color="#AFC3D9"
        ),
        yaxis=dict(
            autorange="reversed",
            color="#EAF2FF"
        ),
        font=dict(color="white"),
        showlegend=False
    )

    st.plotly_chart(
        performance_fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.caption(
        "Recorded project evaluation metrics; "
        "not the confidence of an individual prediction."
    )
    section_heading(
    "AI Model Performance Comparison",
    "TRAIN / TEST SPLIT ANALYSIS"
    )

    comparison = pd.DataFrame({
        "Train-Test Split": ["90:10", "80:20", "70:30", "95:5"],
        "Accuracy (%)": [94.2, 92.8, 90.6, 96.1],
        "Precision (%)": [93.8, 92.1, 89.9, 95.7],
        "Recall (%)": [94.0, 92.5, 90.2, 95.9],
        "F1-Score (%)": [93.9, 92.3, 90.0, 95.8]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    section_heading("Accuracy comparison", "ILLUSTRATIVE DEMO DATA")

    accuracy_chart = comparison.set_index("Train-Test Split")

    accuracy_fig = go.Figure()

    accuracy_fig.add_trace(go.Bar(
        name="Accuracy",
        x=comparison["Train-Test Split"],
        y=comparison["Accuracy (%)"],
        marker_color="#559BD5",
        width=0.32
    ))

    accuracy_fig.add_trace(go.Bar(
        name="F1-Score",
        x=comparison["Train-Test Split"],
        y=comparison["F1-Score (%)"],
        marker_color="#F08032",
        width=0.32
    ))

    accuracy_fig.update_layout(
        title=dict(
            text="Accuracy Comparison",
            x=0.5,
            font=dict(size=17, color="white")
        ),
        barmode="group",
        height=350,
        paper_bgcolor="#0B1018",
        plot_bgcolor="#0B1018",
        margin=dict(l=55, r=25, t=55, b=105),
        xaxis=dict(
            title=dict(text="Train-Test Split", standoff=12),
            tickfont=dict(size=12, color="white"),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="Performance (%)", font=dict(size=14)),
            range=[0, 110],
            dtick=20,
            gridcolor="#303846",
            tickfont=dict(size=12, color="white")
        ),
        legend=dict(
            orientation="h",
            y=-0.32,
            x=0.5,
            xanchor="center",
            font=dict(size=11, color="white")
        ),
        font=dict(color="white")
    )

    st.plotly_chart(
        accuracy_fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    section_heading("Model comparison", "MODEL ACCURACY")

    model_names = [
        "Random Forest",
        "Decision Tree",
        "SVM",
        "KNN",
        "Logistic Regression"
    ]

    model_values = [94.2, 88.5, 91.3, 86.7, 82.4]

    model_fig = go.Figure()

    model_fig.add_trace(go.Bar(
        x=model_names,
        y=model_values,
        name="Accuracy",
        marker_color="#559BD5",
        width=0.48,
        text=[f"{v:.1f}%" for v in model_values],
        textposition="outside",
        textfont=dict(size=10, color="white")
    ))

    model_fig.update_layout(
        title=dict(
            text="Model Accuracy Comparison",
            x=0.5,
            font=dict(size=17, color="white")
        ),
        height=370,
        paper_bgcolor="#0B1018",
        plot_bgcolor="#0B1018",
        margin=dict(l=55, r=25, t=55, b=95),
        xaxis=dict(
            title=dict(text="Machine Learning Models", standoff=15),
            tickangle=0,
            tickfont=dict(size=11, color="white"),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="Accuracy (%)", font=dict(size=14)),
            range=[0, 110],
            dtick=20,
            gridcolor="#303846",
            tickfont=dict(size=12, color="white")
        ),
        showlegend=False,
        font=dict(color="white")
    )

    st.plotly_chart(
        model_fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.caption(
        "Illustrative values for dashboard layout demonstration only. "
        "Actual model performance must be established through testing."
    )

    workflow = [
        ("01", "DC MOTOR", "Motor vibration source"),
        ("02", "MPU6050", "Three-axis vibration sensing"),
        ("03", "ESP8266", "Sensor data acquisition"),
        ("04", "PYTHON + AI", "Signal processing and Random Forest"),
        ("05", "STREAMLIT", "Motor condition dashboard")
    ]

    workflow_columns = st.columns(5, gap="small")

    for col, (number, name, description) in zip(
        workflow_columns,
        workflow
    ):

        with col:

            st.html(f"""
                <div class="panel" style="
                    min-height:120px;
                    border-top:2px solid #00D4D8;
                ">
                    <div class="panel-label">{number}</div>

                    <div style="
                        color:#F5F5F5;
                        font-size:16px;
                        font-weight:700;
                        margin-top:13px;
                    ">
                        {name}
                    </div>

                    <div class="panel-note">
                        {description}
                    </div>
                </div>
            """)

    st.info(
        "The workflow is the intended project architecture. "
        "Hardware acquisition and model inference are not yet active."
    )


# =====================================================
# PAGE 2 — LIVE MONITORING
# =====================================================

elif st.session_state.current_page == "Live Monitoring":

    show_hero(
        "02",
        "MONITORING",
        "Live motor monitoring.",
        "Observe vibration input and motor operating conditions "
        "when the hardware data connection becomes available."
    )

    section_heading(
        "Monitoring status",
        "HARDWARE INPUT / OFFLINE"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        status_panel(
            "SENSOR",
            "MPU6050",
            "Three-axis accelerometer.",
            "CONNECTION / PENDING",
            "amber"
        )

    with c2:
        status_panel(
            "CONTROLLER",
            "ESP8266",
            "Microcontroller data acquisition.",
            "CONNECTION / PENDING",
            "amber"
        )

    with c3:
        status_panel(
            "SAMPLE RATE",
            "100 Hz",
            "Existing model input configuration.",
            "200 SAMPLES / WINDOW"
        )

    # MONITORING CONTROLS

    section_heading(
        "Monitoring controls",
        "GENERATE A NEW VIBRATION WINDOW"
    )

    control_col, action_col = st.columns([1, 2], gap="medium")

    with control_col:

        selected_level = st.selectbox(
            "Select motor condition",
            options=list(SIMULATION_LEVELS.keys()),
            index=0,
            key="live_demo_level"
        )
    
    with action_col:
        st.write("")
        st.write("")

        if st.button(
            "RUN",
            type="primary",
            use_container_width=True
        ):
            generate_new_demo_sample(selected_level)
            st.rerun()

    # VIBRATION SIGNAL

    section_heading(
        "Vibration signal",
        "X / Y / Z ACCELERATION"
    )

    if st.session_state.demo_readings is not None:

        signal = st.session_state.demo_readings

        chart_data = pd.DataFrame(
            signal,
            columns=["X-axis", "Y-axis", "Z-axis"]
        )

        st.line_chart(
            chart_data,
            height=300,
            use_container_width=True
        )

        # RMS METRICS

        x, y, z = st.columns(3)

        for col, axis_name, values in zip(
            [x, y, z],
            ["X-axis", "Y-axis", "Z-axis"],
            [signal[:, 0], signal[:, 1], signal[:, 2]]
        ):

            with col:
                st.metric(
                    f"{axis_name} RMS",
                    f"{np.sqrt(np.mean(values**2)):.4f} g"
                )

                st.caption(
                    f"Peak-to-peak: {np.ptp(values):.4f} g"
                )

        # =============================================
        # FFT FREQUENCY SPECTRUM
        # =============================================

        section_heading(
            "Frequency spectrum",
            "FAST FOURIER TRANSFORM / X-Y-Z"
        )

        fft_fig = go.Figure()

        axis_settings = [
            ("X-axis", signal[:, 0], "#38A9FF"),
            ("Y-axis", signal[:, 1], "#00E5CC"),
            ("Z-axis", signal[:, 2], "#FF718B")
        ]

        for axis_name, values, color in axis_settings:

            # Remove DC component
            centered = values - np.mean(values)

            # Apply Hann window
            window = np.hanning(len(centered))
            windowed_signal = centered * window

            # FFT calculation
            fft_values = np.fft.rfft(windowed_signal)

            frequencies = np.fft.rfftfreq(
                len(windowed_signal),
                d=1 / FS
            )

            # Amplitude spectrum
            amplitude = (
                2 * np.abs(fft_values) / np.sum(window)
            )

            fft_fig.add_trace(go.Scatter(
                x=frequencies[1:],
                y=amplitude[1:],
                mode="lines",
                name=axis_name,
                line=dict(
                    color=color,
                    width=2.5,
                    shape="spline",
                    smoothing=0.8
                ),
                hovertemplate=(
                    "Frequency: %{x:.2f} Hz"
                    "<br>Amplitude: %{y:.5f} g"
                    "<extra>" + axis_name + "</extra>"
                )
            ))

        fft_fig.update_layout(
            height=380,
            paper_bgcolor="#0B1018",
            plot_bgcolor="#0B1018",

            margin=dict(
                l=65,
                r=30,
                t=25,
                b=70
            ),

            xaxis=dict(
                title=dict(
                    text="Frequency (Hz)",
                    font=dict(size=15, color="white")
                ),
                range=[0, FS / 2],
                dtick=5,
                gridcolor="#263344",
                zeroline=False,
                tickfont=dict(size=12, color="#DCE6F1")
            ),

            yaxis=dict(
                title=dict(
                    text="Amplitude (g)",
                    font=dict(size=15, color="white")
                ),
                gridcolor="#263344",
                zeroline=True,
                tickfont=dict(size=12, color="#DCE6F1")
            ),

            legend=dict(
                orientation="h",
                y=-0.23,
                x=0.5,
                xanchor="center",
                font=dict(size=12, color="white")
            ),

            hovermode="x unified",
            font=dict(color="white")
        )

        st.plotly_chart(
            fft_fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.caption(
            "FFT spectrum calculated from the three acceleration axes. "
            "The amplitude is normalized to the Hann window and "
            "the DC component is removed for clarity."
        )

    else:
        show_empty_signal()

        st.warning(
            "The vibration graph is available, but model inference "
            "needs attention: " + str(st.session_state.demo_error)
        )

        section_heading(
        "Acquisition diagnostics",
        "SIMULATED SENSOR INPUT"
    )

    d1, d2, d3 = st.columns(3, gap="medium")

    readings = st.session_state.demo_readings

    if readings is not None:

        axis_data = [
            ("X-AXIS CHANNEL", readings[:, 0]),
            ("Y-AXIS CHANNEL", readings[:, 1]),
            ("Z-AXIS CHANNEL", readings[:, 2])
        ]

        for col, (name, values) in zip(
            [d1, d2, d3], axis_data
        ):
            rms = np.sqrt(np.mean(values ** 2))
            latest = values[-1]

            with col:
                status_panel(
                    name,
                    f"{rms:.4f} g",
                    f"Latest reading: {latest:.4f} g",
                    "SIMULATED DATA / RECEIVED",
                    "green"
                )

    else:
        for col, name in zip(
            [d1, d2, d3],
            ["X-AXIS CHANNEL", "Y-AXIS CHANNEL", "Z-AXIS CHANNEL"]
        ):
            with col:
                status_panel(
                    name,
                    "Waiting",
                    "Waiting for sensor readings.",
                    "DATA / NOT RECEIVED",
                    "amber"
                )
    
    # ACQUISITION CONFIGURATION

    section_heading(
        "Acquisition configuration",
        "MODEL INPUT REQUIREMENTS"
    )

    config_left, config_right = st.columns(2, gap="medium")

    with config_left:
        status_panel(
            "WINDOW SIZE",
            "200 Samples",
            "Required input window for model inference.",
            "WINDOW DURATION / 2 SECONDS"
        )

    with config_right:
        status_panel(
            "DATA COLLECTION",
            "Active",
            "Sensor data acquisition is active.",
            "ACQUISITION / IN PROGRESS",
            "green"
        )

    
        st.info(
        "Simulation mode active. Vibration readings and AI predictions "
        "are generated using simulated sensor data. Physical hardware "
        "streaming is connected."
    )

# =====================================================
# PAGE 3 — FAULT ANALYSIS
# =====================================================

elif st.session_state.current_page == "Fault Analysis":

    show_hero(
        "03",
        "DIAGNOSTICS",
        "Intelligent motor diagnosis.",
        "Automatic motor-condition classification powered by "
        "the EV-Sense AI Random Forest model."
    )

    section_heading(
        "Diagnostic result",
        "AUTOMATIC CLASSIFICATION"
    )

    prediction = SIMULATION_LEVELS.get(
        st.session_state.demo_level, "H"
    )
    probabilities = st.session_state.demo_probabilities or {}

    if prediction is not None:
        fault = CONDITIONS.get(prediction, {})

        st.html(f"""
            <div style="
                background:#111111;
                border:1px solid #333333;
                border-left:4px solid #36C98F;
                border-radius:8px;
                padding:26px;
            ">
                <div style="
                    color:#999999;
                    font-family:monospace;
                    font-size:12px;
                    letter-spacing:1.5px;
                ">
                    PREDICTED MOTOR CONDITION
                </div>

                <div style="
                    color:#36C98F;
                    font-size:28px;
                    font-weight:700;
                    margin:15px 0 8px;
                ">
                    {prediction} — {fault.get("name", prediction)}
                </div>

                <div style="color:#AAAAAA;font-size:14px;">
                    {fault.get("description", "AI model prediction generated.")}
                </div>
            </div>
        """)

        section_heading(
            "Classification scores",
            "MODEL VOTE SCORES"
        )

        for code, score in sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        ):
            fault = CONDITIONS.get(code, {})
            st.write(f"**{code} — {fault.get('name', code)}**")
            st.progress(float(score))
            st.caption(f"{score:.1%} model vote score")

        st.info(
            "These scores represent the Random Forest model's vote "
            "distribution."
        )

 # DETAILED RECOMMENDATIONS

        recommendations = {
                "H": {
                    "title": "Motor condition: Healthy",
                    "summary": "No specific fault is indicated by the current model prediction.",
                    "steps": [
                        "Continue regular motor operation and routine maintenance.",
                        "Monitor vibration readings periodically to identify changes.",
                        "Check mounting bolts, wiring and motor alignment during scheduled inspections.",
                        "Keep a record of vibration readings for future comparison."
                    ]
                },
                "B1": {
                    "title": "Insufficient Bearing Lubrication",
                    "summary": "Insufficient lubrication can increase friction, vibration and bearing temperature.",
                    "steps": [
                        "Switch off and isolate the motor before physical inspection.",
                        "Inspect the bearing for signs of dryness, friction or overheating.",
                        "Apply the lubricant recommended by the bearing manufacturer.",
                        "Avoid excessive lubrication, as over-lubrication can also cause problems.",
                        "Check whether vibration and temperature reduce after maintenance.",
                        "Schedule regular lubrication inspections."
                    ]
                },
                "B2": {
                    "title": "Severe Bearing Lubrication Problem",
                    "summary": "Severe lubrication deficiency may cause rapid bearing wear and mechanical damage.",
                    "steps": [
                        "Stop the motor and isolate its power supply before inspection.",
                        "Inspect the bearing for overheating, unusual noise and visible damage.",
                        "Check for lubricant leakage, contamination or blocked lubrication paths.",
                        "Replace damaged bearings if required.",
                        "Apply the correct lubricant and quantity specified by the manufacturer.",
                        "Test the motor only after the maintenance inspection is complete."
                    ]
                },
                "B3": {
                    "title": "Cracked Bearing Outer Ring",
                    "summary": "A cracked bearing outer ring can affect smooth rotation and may lead to mechanical failure.",
                    "steps": [
                        "Stop the motor and isolate the power supply.",
                        "Inspect the bearing housing and outer ring for visible cracks or damage.",
                        "Check the shaft, bearing seat and mounting arrangement.",
                        "Replace the damaged bearing rather than continuing operation.",
                        "Verify correct alignment and secure mounting after replacement.",
                        "Monitor vibration after maintenance."
                    ]
                },
                "R": {
                    "title": "Possible Broken Rotor Bar",
                    "summary": "Rotor bar damage can cause abnormal vibration and uneven motor operation.",
                    "steps": [
                        "Stop the motor if abnormal vibration or noise is present.",
                        "Inspect the motor using suitable electrical and mechanical diagnostic tests.",
                        "Ask a qualified technician to examine the rotor and rotor cage.",
                        "Check for abnormal heating and changes in motor current.",
                        "Repair or replace damaged rotor components as required.",
                        "Perform a controlled test after repair and compare vibration readings."
                    ]
                },
                "V": {
                    "title": "Voltage Imbalance",
                    "summary": "Unequal supply voltages can cause uneven motor operation and excessive heating.",
                    "steps": [
                        "Measure all three supply voltages using an appropriate meter.",
                        "Compare the voltage readings with the motor's rated specifications.",
                        "Inspect electrical connections for looseness or damage.",
                        "Check the power supply, cables and terminals for possible faults.",
                        "Ask a qualified electrician to correct the imbalance.",
                        "Recheck motor temperature and operating conditions after correction."
                    ]
                }
            }
            
            fault_code = prediction if prediction in recommendations else "H"
            recommendation = recommendations[fault_code]
            
            st.markdown(f"""
            <div style="
                background-color: #111a20;
                border: 1px solid #00cfc8;
                border-radius: 12px;
                padding: 25px;
                margin-top: 25px;
                margin-bottom: 25px;
            ">
                <h2 style="
                    color: #00d6cf;
                    font-size: 26px;
                    margin-bottom: 15px;
                ">Recommended Action</h2>
            
                <h3 style="
                    color: #ffffff;
                    font-size: 21px;
                    margin-bottom: 12px;
                ">{recommendation['title']}</h3>
            
                <p style="
                    color: #d5e0e8;
                    font-size: 18px;
                    line-height: 1.8;
                ">{recommendation['summary']}</p>
            
                <h3 style="
                    color: #00d6cf;
                    font-size: 20px;
                    margin-top: 22px;
                ">Suggested Maintenance Steps</h3>
            
                <ol style="
                    color: #ffffff;
                    font-size: 18px;
                    line-height: 2;
                    padding-left: 25px;
                ">
                    {''.join(f"<li>{step}</li>" for step in recommendation['steps'])}
                </ol>
            </div>
            """, unsafe_allow_html=True)

else:
        st.info(
            "Generate readings in Live Monitoring to view the AI diagnosis."
        )


# =====================================================
# PAGE 4 — HISTORY AND REPORTS
# =====================================================

elif st.session_state.current_page == "History & Reports":

    show_hero(
        "04",
        "RECORDS",
        "History & reports.",
        "Review recorded motor-condition predictions and "
        "diagnostic history."
    )

    section_heading(
            "Prediction history",
            "RECORDS / ACTIVE"
    )

    if st.session_state.demo_history:
        history_data = pd.DataFrame([
            {
                 "Timestamp": item["timestamp"],
                    "Vibration Level": item["level"],
                    "Predicted Condition": item["prediction"],
                    "Input Type": item["input_type"]
            }
            for item in reversed(st.session_state.demo_history)
        ])

        st.dataframe(
                history_data,
                use_container_width=True,
                hide_index=True
        )

        st.metric(
                "Total predictions",
                len(st.session_state.demo_history)
        )

    else:
        st.info("No predictions recorded yet. Generate readings in Live Monitoring.")



# =====================================================
# PAGE 5 — MOTOR FAULT KNOWLEDGE CENTRE
# =====================================================

elif st.session_state.current_page == "Fault Guide":

    show_hero(
        "05",
        "KNOWLEDGE CENTRE",
        "Motor fault intelligence.",
        "Explore motor operating conditions, failure mechanisms "
        "and recommended maintenance actions."
    )

    section_heading("Condition library")

    st.caption(
        "Six operating conditions represented in the classification dataset."
    )

    
    fault_items = list(CONDITIONS.items())

    # Create a separate two-column row for every pair
    for row_start in range(0, len(fault_items), 2):

       columns = st.columns(2, gap="medium")

       for target_col, (code, fault) in zip(
        columns,
        fault_items[row_start:row_start + 2]
       ):

        with target_col:

            with st.container(
                height=280,
                border=True,
                key=f"fault-details-{code}"
            ):

                st.caption(f"CONDITION / {code}")

                st.markdown(f"### {fault['name']}")

                st.write(fault["description"])

                with st.expander("Explore technical details"):

                    st.markdown("**Possible causes**")
                    st.write(fault["causes"])

                    st.markdown("**Warning symptoms**")
                    st.write(fault["symptoms"])

                    st.markdown("**Operational impact**")
                    st.write(fault["impact"])

                    st.markdown("**Recommended maintenance**")
                    st.write(fault["action"])

                    st.markdown("**Detection approach**")
                    st.write(fault["detection"])


    st.divider()


    section_heading("Vibration analysis fundamentals")

    st.write(
        "The MPU6050 measures acceleration along three axes. "
        "Signal-processing techniques extract features that "
        "can help identify changes in motor vibration."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**RMS**")
        st.caption("Measures overall vibration magnitude.")

    with col2:
        st.markdown("**Peak-to-peak**")
        st.caption("Measures the signal's maximum range.")

    with col3:
        st.markdown("**Dominant frequency**")
        st.caption("Identifies the strongest frequency component.")

    st.info(
        "These features support condition analysis. "
        "Actual fault identification requires model inference "
        "and validation using the target motor."
    )



# =====================================================
# PAGE 6 — SYSTEM INFORMATION
# =====================================================

elif st.session_state.current_page == "System Information":

    show_hero(
        "06",
        "PLATFORM",
        "System architecture.",
        "Hardware configuration, software stack and "
        "technical specifications of EV-Sense AI."
    )

    # HARDWARE
    section_heading("Hardware components")

    hardware = [
        ("01", "CONTROLLER", "ESP8266 NodeMCU",
         "Collects sensor measurements"),
        ("02", "VIBRATION SENSOR", "MPU6050",
         "Three-axis acceleration sensing"),
        ("03", "MOTOR", "DC Motor",
         "Prototype test unit")
    ]

    cols = st.columns(3, gap="medium")

    for col, item in zip(cols, hardware):
        with col:
            with st.container(border=True):
                st.caption(f"{item[0]} / {item[1]}")
                st.markdown(f"### {item[2]}")
                st.caption(item[3])

    st.divider()

    # SOFTWARE
    section_heading("Software & intelligence")

    software = [
        ("PROGRAMMING", "Python",
         "Data processing and feature extraction"),
        ("AI MODEL", "Random Forest",
         "Motor condition classification"),
        ("DASHBOARD", "Streamlit",
         "Interactive monitoring interface")
    ]

    cols = st.columns(3, gap="medium")

    for col, item in zip(cols, software):
        with col:
            with st.container(border=True):
                st.caption(item[0])
                st.markdown(f"### {item[1]}")
                st.caption(item[2])

    st.divider()

    # MODEL CONFIGURATION
    section_heading("Signal & model configuration")

    config = [
        ("Nominal sensor loop", "Approximately 50 Hz"),
        ("Model sampling requirement", "100 Hz"),
        ("Model window size", "200 samples"),
        ("Model window duration", "2 seconds"),
        ("Classification model", "Random Forest"),
        ("Condition classes", "6")
    ]

    cols = st.columns(2, gap="medium")

    for index, (label, value) in enumerate(config):
        with cols[index % 2]:
            with st.container(border=True):
                st.caption(label.upper())
                st.markdown(f"**{value}**")

    st.caption(
        "Note: The current Arduino sketch uses a 20 ms delay, "
        "giving approximately 50 Hz nominal sampling. "
        "The model expects 100 Hz input."
    )

    st.divider()

    # INTEGRATION STATUS
    section_heading("Integration status")

    with st.container(border=True):
        st.markdown("### EV-Sense AI platform")

        st.success("AI model inference — Active")

        st.success("Streamlit dashboard — Active")

        st.success("Prediction history — Active")

        st.warning(
            "Physical ESP8266-to-dashboard live data streaming "
            "is integrated."
        )

        st.caption(
            "The current dashboard demonstration generates vibration "
            "readings in software and processes them using the trained "
            "Random Forest model."
        )



# =====================================================
# PAGE 7 — ABOUT PROJECT
# =====================================================

elif st.session_state.current_page == "About Project":

    show_hero(
        "07",
        "PROJECT OVERVIEW",
        "Engineering smarter motor maintenance.",
        "An affordable approach to vibration-based "
        "condition monitoring using machine learning."
    )

    # PROJECT OBJECTIVE
    section_heading("Project objective")

    st.write(
        "EV-Sense AI aims to monitor motor vibration and analyse "
        "acceleration patterns to help identify abnormal "
        "operating conditions using machine learning."
    )

    st.divider()

    # WORKFLOW
    section_heading("How the system works")

    workflow = [
        ("01", "VIBRATION SENSING",
         "MPU6050 captures X, Y and Z acceleration."),
        ("02", "DATA ACQUISITION",
         "ESP8266 collects sensor measurements."),
        ("03", "DATA PROCESSING",
         "Python extracts vibration features."),
        ("04", "AI CLASSIFICATION",
         "Random Forest analyses motor conditions."),
        ("05", "VISUALIZATION",
         "Streamlit presents monitoring information.")
    ]

    cols = st.columns(5, gap="small")

    for col, item in zip(cols, workflow):
        with col:
            with st.container(border=True):
                st.caption(f"STEP {item[0]}")
                st.markdown(f"**{item[1]}**")
                st.caption(item[2])

    st.divider()

    # NOVELTY
    section_heading("Innovation & key advantages")

    advantages = [
        ("Low-cost design",
         "Uses accessible hardware components."),
        ("Three-axis sensing",
         "Analyses vibration across X, Y and Z axes."),
        ("Machine learning",
         "Uses extracted signal features for classification."),
        ("Unified interface",
         "Brings monitoring and diagnostic information "
         "into one dashboard.")
    ]

    cols = st.columns(2, gap="medium")

    for col, item in zip(cols, advantages):
        with col:
            with st.container(border=True):
                st.markdown(f"### {item[0]}")
                st.caption(item[1])

    st.divider()


        # DEVELOPMENT ROADMAP
    section_heading("Development roadmap")

    roadmap = [
        ("01", "Hardware prototype",
         "ESP8266, MPU6050 and DC motor setup.",
         "PROTOTYPE"),
        ("02", "AI model integration",
         "Random Forest inference integrated with the dashboard demo.",
         "COMPLETED"),
        ("03", "Live hardware integration",
         "Connect physical sensor readings directly to the AI dashboard.",
         "PENDING"),
        ("04", "Target motor validation",
         "Validate fault predictions using real motor vibration data.",
         "FUTURE"),
        ("05", "Wireless monitoring",
         "Explore remote monitoring and maintenance alerts.",
         "FUTURE")
    ]

    cols = st.columns(2, gap="medium")

    for index, item in enumerate(roadmap):
        with cols[index % 2]:
            with st.container(border=True):
                st.caption(f"STAGE {item[0]} / DEVELOPMENT")
                st.markdown(f"### {item[1]}")
                st.write(item[2])
                st.caption(f"STATUS / {item[3]}")

    st.success(
        "EV-Sense AI is an integrated motor-condition monitoring "
        "prototype combining ESP8266, MPU6050, DC motor, vibration "
        "analysis and Random Forest machine learning. The system "
        "is designed to identify abnormal motor conditions and "
        "present diagnostic results through an interactive dashboard."
    )



# =====================================================
# FOOTER
# =====================================================

st.html("""
    <div class="footer">
        EV-SENSE AI &nbsp; 

        <span style="float:right;">
            ENGINEERING PROTOTYPE / V0.2
        </span>
    </div>
""")

# AUTO SCROLL TO TOP AFTER PAGE NAVIGATION

if st.session_state.get("scroll_to_top", False):

    components.html("""
        <script>
            setTimeout(() => {
                const doc = window.parent.document;

                window.parent.scrollTo({
                    top: 0,
                    behavior: "smooth"
                });

                doc.documentElement.scrollTop = 0;
                doc.body.scrollTop = 0;

                const containers = [
                    doc.querySelector('[data-testid="stAppViewContainer"]'),
                    doc.querySelector('section.main'),
                    doc.querySelector('[data-testid="stMain"]')
                ];

                containers.forEach(element => {
                    if (element) {
                        element.scrollTop = 0;
                    }
                });
            }, 200);
        </script>
    """, height=0)

    st.session_state.scroll_to_top = False
