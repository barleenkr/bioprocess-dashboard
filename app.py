import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# IMPORT YOUR BACKEND FILE
from bioprocess_scaleup_v2 import (
    SyntheticDatasetGenerator,
    ScaleUpMLModel,
    BioprocessScaleUpPredictor,
    LabScaleInput,
    generate_charts
)

st.set_page_config(layout="wide")

# =========================
# 🚀 INITIALIZE MODEL
# =========================
@st.cache_resource
def load_model():
    gen = SyntheticDatasetGenerator(n=2000)
    df = gen.generate()

    ml = ScaleUpMLModel()
    ml.train(df)

    predictor = BioprocessScaleUpPredictor(ml)
    return predictor, df, ml

predictor, df, ml = load_model()

# =========================
# 🎛️ SIDEBAR INPUTS
# =========================
st.sidebar.title("⚙️ Process Inputs")

organism = st.sidebar.selectbox("Organism", ["ecoli", "bacillus", "yeast", "cho"])
pH = st.sidebar.slider("pH", 5.0, 9.0, 7.0)
temp = st.sidebar.slider("Temperature (°C)", 20.0, 45.0, 37.0)
rpm = st.sidebar.slider("Agitation (rpm)", 50, 800, 400)
vvm = st.sidebar.slider("Aeration (vvm)", 0.1, 3.0, 1.0)
yield_lab = st.sidebar.slider("Lab Yield (g/L)", 0.1, 15.0, 4.0)
mu_max = st.sidebar.slider("mu_max (h⁻¹)", 0.01, 1.5, 0.6)
do = st.sidebar.slider("DO (%)", 5.0, 100.0, 30.0)
substrate = st.sidebar.slider("Substrate (g/L)", 1.0, 50.0, 20.0)
volume = st.sidebar.selectbox("Target Volume (L)", [100, 300, 1000])
strategy = st.sidebar.selectbox(
    "Scale Strategy",
    ["constant_kla", "constant_pvv", "constant_tip_speed", "constant_Re"]
)

run = st.sidebar.button("🚀 Run Prediction")

# =========================
# 🧬 MAIN DASHBOARD
# =========================
st.title("🧬 BioProcess Scale-Up Predictor")
st.markdown("### Industrial Biotech AI Dashboard")

if run:

    inp = LabScaleInput(
        organism=organism,
        pH=pH,
        temperature=temp,
        agitation_rpm=rpm,
        aeration_vvm=vvm,
        lab_yield_gL=yield_lab,
        mu_max=mu_max,
        do_percent=do,
        substrate_conc=substrate,
        target_volume_L=volume,
        strategy=strategy
    )

    result = predictor.predict(inp)

    # =========================
    # 📊 TOP METRICS
    # =========================
    st.subheader("📊 Key Outputs")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Yield (g/L)", f"{result['ml']['yield_scale']:.2f}")
    col2.metric("Yield Drop (%)", f"{result['yield_drop_pct']}")
    col3.metric("DO (%)", f"{result['ml']['do_scale']:.1f}")
    col4.metric("kLa (scale)", f"{result['sc']['kla_scale']}")

    # =========================
    # 🚨 RISK DISPLAY
    # =========================
    st.subheader("🚨 Risk Assessment")

    risk = result["risk_level"]
    score = result["ml"]["risk_score"]

    if risk == "LOW":
        st.success(f"LOW RISK — Score: {score:.1f}")
    elif risk == "MEDIUM":
        st.warning(f"MEDIUM RISK — Score: {score:.1f}")
    else:
        st.error(f"HIGH RISK — Score: {score:.1f}")

    st.progress(score / 100)

    # =========================
    # 📉 MONOD CURVE
    # =========================
    st.subheader("📈 Monod Growth Kinetics")

    S = np.linspace(0, 40, 200)
    Ks = result["org"]["ks"]
    mu = result["input"].mu_max * S / (Ks + S)

    fig, ax = plt.subplots()
    ax.plot(S, mu)
    ax.scatter([result["input"].substrate_conc], [result["mu_monod"]])
    ax.set_xlabel("Substrate (g/L)")
    ax.set_ylabel("Growth rate (h⁻¹)")
    st.pyplot(fig)

    # =========================
    # ⚙️ ENGINEERING OUTPUTS
    # =========================
    st.subheader("⚙️ Engineering Parameters")

    colA, colB, colC = st.columns(3)

    colA.metric("Mixing Time (s)", result["sc"]["mixing_time_s"])
    colB.metric("Tip Speed (m/s)", result["sc"]["tip_speed_ms"])
    colC.metric("Power/Volume", result["sc"]["pvv"])

    # =========================
    # ⚠️ FLAGS
    # =========================
    st.subheader("⚠️ Risk Flags")

    for level, msg in result["flags"]:
        if level == "HIGH":
            st.error(msg)
        elif level == "MEDIUM":
            st.warning(msg)
        else:
            st.info(msg)

    # =========================
    # 🧠 RECOMMENDATIONS
    # =========================
    st.subheader("🧠 Engineering Recommendations")

    for rec in result["recs"]:
        st.write(f"• {rec}")

    # =========================
    # 📊 FULL 6-CHART DASHBOARD
    # =========================
    st.subheader("📊 Full Analysis Dashboard")

    chart_path = generate_charts([result], df, ml)
    st.image(chart_path)