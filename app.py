import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TATA STEEL | Predictive Maintenance",
    layout="wide",
    page_icon="🏭"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pickle.load(open("best_model.pkl", "rb"))

model = load_model()

# ---------------- LOAD DATA ----------------
df = pd.read_csv("ai4i2020.csv")

# ---------------- LOGO PATH ----------------
logo_path = os.path.join(os.path.dirname(__file__), "tata_logo.png")

# ---------------- CLEAN UI STYLE ----------------
st.markdown("""
<style>

.stApp {
    background-color: #0b0f19;
    color: white;
}

/* labels fix */
label {
    color: white !important;
    font-size: 15px !important;
    font-weight: 500;
}

/* selectbox text */
div[data-baseweb="select"] {
    color: black !important;
}

/* input text */
input {
    color: black !important;
}

/* button */
.stButton>button {
    background-color: #E53935;
    color: white;
    height: 50px;
    width: 100%;
    border-radius: 10px;
    font-size: 16px;
    font-weight: bold;
}

/* title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #E53935;
}

/* card */
.card {
    background: #151a26;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏭 TATA STEEL AI SYSTEM")

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
else:
    st.sidebar.warning("Logo missing in folder")

menu = st.sidebar.radio("Navigation", ["Dashboard", "Prediction"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":

    st.markdown('<div class="title">Predictive Maintenance Dashboard</div>', unsafe_allow_html=True)

    total = len(df)
    failures = int(df["Machine failure"].sum())
    failure_rate = (failures / total) * 100

    avg_temp = df["Air temperature [K]"].mean()
    avg_torque = df["Torque [Nm]"].mean()

    health = "🟢 Good" if failure_rate < 5 else "🟡 Moderate" if failure_rate < 10 else "🔴 Critical"

    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""<div class="card"><h4>Total</h4><h2>{total}</h2></div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="card"><h4>Failures</h4><h2>{failures}</h2></div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class="card"><h4>Failure Rate</h4><h2>{failure_rate:.2f}%</h2></div>""", unsafe_allow_html=True)
    col4.markdown(f"""<div class="card"><h4>Health</h4><h2>{health}</h2></div>""", unsafe_allow_html=True)

    st.markdown("## 📊 Key Insights")

    col5, col6 = st.columns(2)

    with col5:
        fig, ax = plt.subplots()
        df["Machine failure"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Failure vs No Failure")
        st.pyplot(fig)

    with col6:
        fig, ax = plt.subplots()
        df["Type"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_title("Machine Types Distribution")
        st.pyplot(fig)

    st.markdown("## 📈 Operational Insights")

    col7, col8 = st.columns(2)

    with col7:
        fig, ax = plt.subplots()
        df["Air temperature [K]"].hist(bins=20, ax=ax)
        ax.set_title("Air Temperature Distribution")
        st.pyplot(fig)

    with col8:
        fig, ax = plt.subplots()
        df["Torque [Nm]"].hist(bins=20, ax=ax)
        ax.set_title("Torque Distribution")
        st.pyplot(fig)

# ---------------- PREDICTION ----------------
elif menu == "Prediction":

    st.markdown('<div class="title">Machine Failure Prediction</div>', unsafe_allow_html=True)

    st.write("### Enter Machine Parameters")

    col1, col2 = st.columns(2)

    with col1:
        type_val = st.selectbox("Machine Type", [0, 1, 2])
        air_temp = st.slider("Air Temperature (K)", 290, 320, 300)
        process_temp = st.slider("Process Temperature (K)", 300, 350, 320)

    with col2:
        rot_speed = st.slider("Rotational Speed (RPM)", 1000, 3000, 1500)
        torque = st.slider("Torque (Nm)", 10, 80, 30)
        tool_wear = st.slider("Tool Wear (min)", 0, 300, 100)

    st.write("")

    if st.button("🚀 Predict Result"):

        input_data = np.array([[type_val, air_temp, process_temp,
                                rot_speed, torque, tool_wear]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠ HIGH FAILURE RISK DETECTED")
            st.warning("Recommendation: Immediate maintenance required")
        else:
            st.success("✅ MACHINE IS SAFE")
            st.info("No maintenance required at this time")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("TATA STEEL Internship Project | Predictive Maintenance System")