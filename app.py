"""
Titanic Survival Prediction - Streamlit Web App
================================================
Loads the trained model (titanic_model.pkl) and provides an interactive
form to predict whether a passenger would survive or not.

Run with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon=" ship",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "titanic_model.pkl"

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #1e3a5f, #4fc3f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.4rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.7rem;
        font-weight: 700;
        color: white;
        margin: 0.6rem 0 1rem 0;
    }
    .survive { background: linear-gradient(135deg, #1b8a3f, #2ecc71); }
    .not-survive { background: linear-gradient(135deg, #a01818, #e74c3c); }
    .prob-caption { text-align: center; font-size: 1.05rem; color: #bbb; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    """Load the trained model bundle from disk."""
    try:
        bundle = joblib.load(path)
        return bundle
    except FileNotFoundError:
        return None


bundle = load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/960px-RMS_Titanic_3.jpg",
        caption="RMS Titanic (1912)",
        use_container_width=True,
    )
    st.title("About")
    st.markdown(
        """
        This app predicts whether a passenger on the **RMS Titanic**
        would **survive** or **not**, based on passenger details.

        The model is a **Random Forest Classifier** trained on the classic
        [Kaggle Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
        (~891 passengers).
        """
    )
    if bundle is not None:
        metrics = bundle.get("metrics", {})
        st.divider()
        st.subheader("Model Metrics (test set)")
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
        m2.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
    st.divider()
    st.caption("Built with Streamlit  |  For educational purposes only.")

# ---------------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">Titanic Survival Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Enter passenger details below to predict survival</p>',
    unsafe_allow_html=True,
)

if bundle is None:
    st.error(
        "**Model file `titanic_model.pkl` not found!**\n\n"
        "First train the model by running:  `python train_model.py`  "
        "then reload this app."
    )
    st.stop()

pipeline = bundle["pipeline"]

# ---- Input Form ----
with st.form("passenger_form"):
    st.subheader("Passenger Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="Age of the passenger in years.",
        )
        sex = st.radio(
            "Gender",
            options=["male", "female"],
            horizontal=True,
        )
        pclass = st.select_slider(
            "Ticket Class",
            options=[1, 2, 3],
            value=3,
            help="1 = Upper, 2 = Middle, 3 = Lower class.",
        )

    with col2:
        fare = st.number_input(
            "Fare (British pounds)",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=1.0,
            help="Ticket price paid. Historical range: ~0 to 512.",
        )
        sibsp = st.number_input(
            "Siblings/Spouses Aboard (SibSp)",
            min_value=0,
            max_value=8,
            value=0,
            step=1,
            help="Number of siblings or spouses traveling with the passenger.",
        )
        parch = st.number_input(
            "Parents/Children Aboard (Parch)",
            min_value=0,
            max_value=6,
            value=0,
            step=1,
            help="Number of parents or children traveling with the passenger.",
        )

    with col3:
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            index=0,
            help="S = Southampton, C = Cherbourg, Q = Queenstown.",
        )
        st.markdown(
            """
            | Code | Port |
            |------|------|
            | S | Southampton |
            | C | Cherbourg |
            | Q | Queenstown |
            """
        )

    submitted = st.form_submit_button(
        "Predict Survival", use_container_width=True, type="primary"
    )

# ---- Prediction ----
if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Age": float(age),
                "SibSp": int(sibsp),
                "Parch": int(parch),
                "Fare": float(fare),
                "Pclass": int(pclass),
                "Sex": sex,
                "Embarked": embarked,
            }
        ]
    )

    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.markdown(
            '<div class="result-box survive">SURVIVED</div>',
            unsafe_allow_html=True,
        )
        st.success("The model predicts this passenger would **survive** the sinking.")
    else:
        st.markdown(
            '<div class="result-box not-survive">NOT SURVIVED</div>',
            unsafe_allow_html=True,
        )
        st.error("The model predicts this passenger would **not survive** the sinking.")

    st.markdown(
        f'<p class="prob-caption">Survival probability: <b>{probability * 100:.1f}%</b></p>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0))

    # Probability metric cards
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("P(Survive)", f"{probability * 100:.1f}%")
    pc2.metric("P(Not Survive)", f"{(1 - probability) * 100:.1f}%")
    pc3.metric("Confidence", f"{max(probability, 1 - probability) * 100:.1f}%")

    with st.expander("See input summary"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

else:
    st.info(
        "Fill in the passenger details above and click **Predict Survival** to see the result."
    )
    st.markdown(
        """
        ---
        **Example:** A 22-year-old **female** in **1st class**, fare **80**,
        traveling alone (**SibSp=0, Parch=0**), embarking from **Cherbourg (C)**
        is predicted to survive with high probability — consistent with the
        historical *"women and children first"* evacuation protocol.
        """
    )
