
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.logistic import hypothesis


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GridGuard | Fault Intelligence",
    page_icon="⚡",
    layout="wide"
)


# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #102541);
    color: #edf4ff;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label {
    color: #edf4ff !important;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(110deg, #12385c, #12617c);
    border: 1px solid #287b9d;
    margin-bottom: 20px;
}
.hero small {
    color: #80e6ff;
    letter-spacing: 3px;
    font-weight: 800;
}
.hero h1 {
    font-size: 38px;
    margin: 8px 0;
}
.hero p {
    color: #d3e8f8 !important;
}
div.stButton > button, div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #22a7e8, #57d9ef);
    color: #07111f;
    border: 0;
    border-radius: 10px;
    font-weight: 800;
    min-height: 48px;
    width: 100%;
}
div[data-testid="stMetric"] {
    background: #112842;
    border: 1px solid #284c6c;
    padding: 16px;
    border-radius: 14px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 40, 66, 0.55);
    border-radius: 16px;
}
.section-label {
    color: #57d9ef !important;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_rbf_model():
    return joblib.load("results/best_rbf_svm.joblib")


@st.cache_data
def load_parameters():
    df = pd.read_csv("data/detect_dataset.csv")
    df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])
    df = df.rename(columns={"Output (S)": "fault"})

    features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train = int(0.6 * len(df))
    train_data = df.iloc[idx[:n_train]]

    X_train = train_data[features].to_numpy()
    X_train_Q = np.column_stack((X_train, X_train ** 2))

    mean_Q = X_train_Q.mean(axis=0)
    std_Q = X_train_Q.std(axis=0)
    std_Q[std_Q == 0] = 1

    theta_Q = np.loadtxt("results/theta_Q.txt")

    return df, mean_Q, std_Q, theta_Q


rbf_data = load_rbf_model()
df, mean_Q, std_Q, theta_Q = load_parameters()

FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <small>POWER SYSTEMS • MACHINE LEARNING</small>
    <h1>⚡ GridGuard</h1>
    <h3>Three-Phase Transmission Line Fault Intelligence</h3>
    <p>Analyze electrical measurements and compare two trained classifiers.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- INPUTS ----------------
st.markdown(
    '<p class="section-label">01 / MEASUREMENT INPUTS</p>',
    unsafe_allow_html=True
)

st.caption(
    "Enter values in the same units and measurement scale used by your dataset."
)

# Load an actual row for a convenient demo
if "sample_number" not in st.session_state:
    st.session_state.sample_number = 0


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.logistic import hypothesis


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="GridGuard | Fault Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. CUSTOM DESIGN
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #102541);
    color: #edf4ff;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label {
    color: #edf4ff !important;
}
.hero {
    background: linear-gradient(120deg, #12385c, #12617c);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #287b9d;
    margin-bottom: 24px;
}
.eyebrow {
    color: #80e6ff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}
.section-label {
    color: #57d9ef !important;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
}
div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #22a7e8, #57d9ef);
    color: #07111f;
    border: none;
    border-radius: 10px;
    font-weight: 800;
    min-height: 46px;
    width: 100%;
}
div[data-testid="stMetric"] {
    background: #112842;
    border: 1px solid #284c6c;
    padding: 15px;
    border-radius: 14px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 40, 66, 0.65);
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD DATA AND MODELS
# ==================================================
FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


@st.cache_data
def load_dataset_and_logistic_parameters():
    data = pd.read_csv("data/detect_dataset.csv")

    data = data.drop(
        columns=["Unnamed: 7", "Unnamed: 8"]
    )
    data = data.rename(
        columns={"Output (S)": "fault"}
    )

    # Reproduce the assignment's exact split
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(data))
    n_train = int(0.6 * len(data))

    training_data = data.iloc[indices[:n_train]]
    X_train = training_data[FEATURES].to_numpy()

    # Set Q: six original features and their squares
    X_train_Q = np.column_stack(
        (X_train, X_train ** 2)
    )

    mean_Q = X_train_Q.mean(axis=0)
    std_Q = X_train_Q.std(axis=0)
    std_Q[std_Q == 0] = 1

    theta_Q = np.loadtxt("results/theta_Q.txt")

    return data, mean_Q, std_Q, theta_Q


@st.cache_resource
def load_rbf_model():
    return joblib.load("results/best_rbf_svm.joblib")


df, mean_Q, std_Q, theta_Q = (
    load_dataset_and_logistic_parameters()
)
rbf_data = load_rbf_model()


# ==================================================
# 4. HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">
        POWER SYSTEMS • INTELLIGENT MONITORING
    </div>
    <h1 style="font-size:40px; margin:10px 0;">
        ⚡ GridGuard
    </h1>
    <h3>Three-Phase Transmission Line Fault Detection</h3>
    <p>
        Analyze electrical measurements, detect possible faults,
        and compare two machine learning models.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 5. VISIBLE DATASET SAMPLE SELECTOR
# ==================================================
st.markdown(
    '<p class="section-label">01 / INPUT MEASUREMENTS</p>',
    unsafe_allow_html=True
)

st.write(
    "Load an actual dataset row for a quick demonstration, "
    "or enter your own measurements below."
)

sample_row = st.selectbox(
    "Choose a dataset row",
    options=list(range(len(df))),
    index=0
)

if st.button("📥 Load Selected Sample"):
    row = df.iloc[int(sample_row)]
    st.write("Actual dataset fault label:", row["fault"])
    for feature in FEATURES:
        st.session_state[feature] = float(row[feature])

    st.session_state["loaded_sample"] = int(sample_row)
    st.rerun()


# Initialize inputs from the first dataset row
default_row = df.iloc[0]

for feature in FEATURES:
    if feature not in st.session_state:
        st.session_state[feature] = float(default_row[feature])


# ==================================================
# 6. MEASUREMENT INPUT FORM
# ==================================================
with st.form("measurement_form"):

    phase_a, phase_b, phase_c = st.columns(3)

    with phase_a:
        st.subheader("🔵 Phase A")

        Ia = st.number_input(
            "Current Ia",
            key="Ia",
            format="%.6f"
        )

        Va = st.number_input(
            "Voltage Va",
            key="Va",
            format="%.6f"
        )

    with phase_b:
        st.subheader("🟢 Phase B")

        Ib = st.number_input(
            "Current Ib",
            key="Ib",
            format="%.6f"
        )

        Vb = st.number_input(
            "Voltage Vb",
            key="Vb",
            format="%.6f"
        )

    with phase_c:
        st.subheader("🟡 Phase C")

        Ic = st.number_input(
            "Current Ic",
            key="Ic",
            format="%.6f"
        )

        Vc = st.number_input(
            "Voltage Vc",
            key="Vc",
            format="%.6f"
        )

    submitted = st.form_submit_button(
        "⚡ ANALYZE TRANSMISSION LINE"
    )


# ==================================================
# 7. PREDICTION AND RESULTS
# ==================================================
if submitted:

    values = np.array(
        [[Ia, Ib, Ic, Va, Vb, Vc]],
        dtype=float
    )

    # ----- Set Q Logistic Regression -----
    values_Q = np.column_stack(
        (values, values ** 2)
    )

    values_Q = (
        (values_Q - mean_Q) / std_Q
    )

    # Add intercept
    values_Q = np.column_stack(
        (np.ones(len(values_Q)), values_Q)
    )

    probability_Q = float(
        hypothesis(values_Q, theta_Q)[0]
    )

    prediction_Q = int(probability_Q >= 0.5)

    # ----- RBF SVM -----
    mean_R = np.asarray(rbf_data["mean"])
    std_R = np.asarray(rbf_data["std"])
    std_R = np.where(std_R == 0, 1, std_R)

    values_R = (values - mean_R) / std_R

    prediction_RBF = int(
        rbf_data["model"].predict(values_R)[0]
    )

    # Save predictions so results remain visible
    st.session_state["last_result"] = {
        "values": values[0].copy(),
        "probability_Q": probability_Q,
        "prediction_Q": prediction_Q,
        "prediction_RBF": prediction_RBF
    }


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================
result = st.session_state.get("last_result")

if result is not None:

    st.markdown("---")

    st.markdown(
        '<p class="section-label">02 / MODEL PREDICTIONS</p>',
        unsafe_allow_html=True
    )

    probability_Q = result["probability_Q"]
    prediction_Q = result["prediction_Q"]
    prediction_RBF = result["prediction_RBF"]

    left, right = st.columns(2)

    with left:
        with st.container(border=True):

            st.subheader("Set Q Logistic Regression")

            if prediction_Q == 1:
                st.error("⚠ FAULT DETECTED")
            else:
                st.success("✓ NO FAULT DETECTED")

            st.metric(
                "Predicted fault probability",
                f"{probability_Q:.2%}"
            )

            st.progress(
                float(np.clip(probability_Q, 0, 1))
            )

    with right:
        with st.container(border=True):

            st.subheader("RBF Support Vector Machine")

            if prediction_RBF == 1:
                st.error("⚠ FAULT DETECTED")
            else:
                st.success("✓ NO FAULT DETECTED")

            st.metric(
                "Predicted class",
                str(prediction_RBF)
            )

            st.caption(
                "Class 1 = Fault | Class 0 = No fault"
            )

    if prediction_Q == prediction_RBF:
        st.success("Both models agree on the predicted class.")
    else:
        st.warning(
            "The models disagree. Review this sample carefully."
        )

    # ==================================================
    # 9. CURRENT AND VOLTAGE GRAPHS
    # ==================================================
    st.markdown("---")

    st.markdown(
        '<p class="section-label">03 / SIGNAL VISUALIZATION</p>',
        unsafe_allow_html=True
    )

    current_values = result["values"][:3]
    voltage_values = result["values"][3:]

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Three-Phase Current")

        fig1, ax1 = plt.subplots(figsize=(6, 4))

        ax1.bar(
            ["Ia", "Ib", "Ic"],
            current_values
        )

        ax1.set_ylabel("Current value")
        ax1.set_title("Current by Phase")
        ax1.grid(axis="y", alpha=0.25)

        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with chart_right:
        st.subheader("Three-Phase Voltage")

        fig2, ax2 = plt.subplots(figsize=(6, 4))

        ax2.bar(
            ["Va", "Vb", "Vc"],
            voltage_values
        )

        ax2.set_ylabel("Voltage value")
        ax2.set_title("Voltage by Phase")
        ax2.grid(axis="y", alpha=0.25)

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ==================================================
    # 10. MODEL COMPARISON GRAPH
    # ==================================================
    st.markdown("---")
    st.subheader("Model Comparison")

    model_names = [
        "Set Q Logistic Regression",
        "RBF SVM"
    ]

    predicted_classes = [
        prediction_Q,
        prediction_RBF
    ]

    fig3, ax3 = plt.subplots(figsize=(9, 3.5))

    bars = ax3.barh(
        model_names,
        predicted_classes,
        height=0.5
    )

    ax3.set_xlim(0, 1.2)
    ax3.set_xticks(
        [0, 1],
        labels=["No Fault (0)", "Fault (1)"]
    )

    ax3.set_title("Predicted Class by Model")
    ax3.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, predicted_classes):
        ax3.text(
            value + 0.03,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontweight="bold"
        )

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ==================================================
    # 11. INPUT SUMMARY
    # ==================================================
    st.markdown("---")
    st.subheader("Measurement Summary")

    input_table = pd.DataFrame({
        "Measurement": FEATURES,
        "Input Value": result["values"]
    })

    st.dataframe(
        input_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Educational project prototype. Not certified for real-time "
        "transmission-line protection."
    )

else:
    st.info(
        "Choose a dataset row and click Load Selected Sample, "
        "or enter your own values. Then click Analyze Transmission Line."
    )


# ==================================================
# 12. FOOTER
# ==================================================
st.markdown("---")

st.caption(
    "GRIDGUARD | MACHINE LEARNING | TRANSMISSION LINE FAULT DETECTION"
)



with st.form("measurement_form_2"):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Phase A")
        Ia = st.number_input(
            "Current Ia", value=float(st.session_state.get("Ia", 0.0)),
            format="%.4f"
        )
        Va = st.number_input(
            "Voltage Va", value=float(st.session_state.get("Va", 0.0)),
            format="%.4f"
        )

    with col_b:
        st.subheader("Phase B")
        Ib = st.number_input(
            "Current Ib", value=float(st.session_state.get("Ib", 0.0)),
            format="%.4f"
        )
        Vb = st.number_input(
            "Voltage Vb", value=float(st.session_state.get("Vb", 0.0)),
            format="%.4f"
        )

    with col_c:
        st.subheader("Phase C")
        Ic = st.number_input(
            "Current Ic", value=float(st.session_state.get("Ic", 0.0)),
            format="%.4f"
        )
        Vc = st.number_input(
            "Voltage Vc", value=float(st.session_state.get("Vc", 0.0)),
            format="%.4f"
        )

    submitted = st.form_submit_button("⚡ ANALYZE MEASUREMENTS")


# ---------------- PREDICTIONS ----------------
if submitted:

    values = np.array([[Ia, Ib, Ic, Va, Vb, Vc]], dtype=float)

    # Set Q: original features and squared features
    values_Q = np.column_stack((values, values ** 2))
    values_Q = (values_Q - mean_Q) / std_Q
    values_Q = np.column_stack((np.ones(1), values_Q))

    probability_Q = float(hypothesis(values_Q, theta_Q)[0])
    prediction_Q = int(probability_Q >= 0.5)

    # RBF SVM uses saved training statistics
    mean_R = np.asarray(rbf_data["mean"])
    std_R = np.asarray(rbf_data["std"])
    std_R = np.where(std_R == 0, 1, std_R)

    values_R = (values - mean_R) / std_R

    prediction_RBF = int(
        rbf_data["model"].predict(values_R)[0]
    )

    st.session_state.last_result = {
        "values": values[0],
        "probability_Q": probability_Q,
        "prediction_Q": prediction_Q,
        "prediction_RBF": prediction_RBF
    }


# ---------------- DISPLAY RESULTS ----------------
result = st.session_state.get("last_result")

if result is not None:

    st.markdown("---")
    st.markdown(
        '<p class="section-label">02 / PREDICTION RESULTS</p>',
        unsafe_allow_html=True
    )

    prediction_Q = result["prediction_Q"]
    prediction_RBF = result["prediction_RBF"]
    probability_Q = result["probability_Q"]

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("### Model 01")
            st.subheader("Set Q Logistic Regression")

            if prediction_Q == 1:
                st.error("⚠ FAULT DETECTED")
            else:
                st.success("✓ NO FAULT DETECTED")

            st.metric(
                "Fault probability",
                f"{probability_Q:.2%}"
            )
            st.progress(float(np.clip(probability_Q, 0, 1)))

    with right:
        with st.container(border=True):
            st.markdown("### Model 02")
            st.subheader("RBF Support Vector Machine")

            if prediction_RBF == 1:
                st.error("⚠ FAULT DETECTED")
            else:
                st.success("✓ NO FAULT DETECTED")

            st.metric("Predicted class", str(prediction_RBF))
            st.caption("Class 1 = Fault | Class 0 = No fault")
            st.info(
                "The saved SVM was trained as a classifier. "
                "This display shows its predicted class, not a probability."
            )

    if prediction_Q == prediction_RBF:
        st.success("Both models agree on this input.")
    else:
        st.warning(
            "The models disagree. Review the input and investigate this sample."
        )

    # ---------------- INPUT GRAPHS ----------------
    st.markdown("---")
    st.markdown(
        '<p class="section-label">03 / SIGNAL VISUALIZATION</p>',
        unsafe_allow_html=True
    )

    currents = [Ia, Ib, Ic]
    voltages = [Va, Vb, Vc]

    graph_left, graph_right = st.columns(2)

    with graph_left:
        st.subheader("Three-Phase Currents")

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(
            ["Ia", "Ib", "Ic"],
            currents,
            color=["#28b8f0", "#32d6b0", "#f5b942"]
        )
        ax1.set_ylabel("Current")
        ax1.set_title("Current by Phase")
        ax1.grid(axis="y", alpha=0.25)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with graph_right:
        st.subheader("Three-Phase Voltages")

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(
            ["Va", "Vb", "Vc"],
            voltages,
            color=["#28b8f0", "#32d6b0", "#f5b942"]
        )
        ax2.set_ylabel("Voltage")
        ax2.set_title("Voltage by Phase")
        ax2.grid(axis="y", alpha=0.25)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ---------------- MODEL COMPARISON GRAPH ----------------
    st.subheader("Model Prediction Comparison")

    fig3, ax3 = plt.subplots(figsize=(8, 3.5))

    model_names = ["Set Q Logistic Regression", "RBF SVM"]
    model_predictions = [prediction_Q, prediction_RBF]
    display_values = [1 if p == 1 else 0 for p in model_predictions]

    ax3.barh(
        model_names,
        display_values,
        color=["#f05d75", "#26c6a5"],
        height=0.48
    )
    ax3.set_xlim(0, 1.15)
    ax3.set_xticks([0, 1], labels=["No Fault (0)", "Fault (1)"])
    ax3.set_title("Predicted Class by Model")
    ax3.grid(axis="x", alpha=0.25)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ---------------- INPUT SUMMARY ----------------
    st.subheader("Input Summary")

    input_table = pd.DataFrame({
        "Measurement": FEATURES,
        "Value": result["values"]
    })

    st.dataframe(
        input_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Educational prototype only. This model has not been certified "
        "for real-time electrical protection."
    )

else:
    st.info(
        "Enter measurements above, or load a dataset sample, "
        "then click ANALYZE MEASUREMENTS to see predictions and graphs."
    )


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "GRIDGUARD | Three-Phase Fault Detection | Set Q Logistic Regression + RBF SVM"
)
