"""Streamlit app for the churn model.   Run:  streamlit run app.py
(Run all cells in churn_prediction.ipynb first, so models/churn_model.joblib exists.)
"""
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Risk Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Bank Customer Churn Predictor")

MODEL_FILE = "models/churn_model.joblib"
if not os.path.exists(MODEL_FILE):
    st.error("Model file not found. Run churn_prediction.ipynb first.")
    st.stop()

saved = joblib.load(MODEL_FILE)
model, columns = saved["model"], saved["columns"]

tab1, tab2 = st.tabs(["🔍 Predict", "📊 Data & Model Insights"])

# ---------------- Tab 1: prediction ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 92, 40)
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        credit_score = st.slider("Credit score", 350, 850, 650)
    with c2:
        balance = st.number_input("Balance", 0.0, 260000.0, 75000.0, step=1000.0)
        salary = st.number_input("Estimated salary", 0.0, 200000.0, 100000.0, step=1000.0)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        num_products = st.selectbox("Number of products", [1, 2, 3, 4], index=1)
    with c3:
        is_active = st.radio("Active member?", ["Yes", "No"], horizontal=True)
        has_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)
        satisfaction = st.slider("Satisfaction score", 1, 5, 3)
        card_type = st.selectbox("Card type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
        points = st.number_input("Points earned", 100, 1000, 600)

    if st.button("Predict churn risk", type="primary"):
        # 1. put the inputs in a one-row table
        row = pd.DataFrame([{
            "CreditScore": credit_score, "Geography": geography, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": num_products, "HasCrCard": int(has_card == "Yes"),
            "IsActiveMember": int(is_active == "Yes"), "EstimatedSalary": salary,
            "Satisfaction Score": satisfaction, "Card Type": card_type,
            "Point Earned": points,
        }])

        # 2. same feature engineering as in the notebook
        row["HasBalance"] = (row["Balance"] > 0).astype(int)
        row["BalanceSalaryRatio"] = row["Balance"] / (row["EstimatedSalary"] + 1)
        row["AgeGroup"] = pd.cut(row["Age"], [17, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)
        row["Gender"] = row["Gender"].map({"Female": 0, "Male": 1})
        row = pd.get_dummies(row, columns=["Geography", "Card Type"])

        # 3. make sure we have exactly the columns the model was trained on
        row = row.reindex(columns=columns, fill_value=0)

        prob = model.predict_proba(row)[0, 1]
        st.metric("Churn probability", f"{prob:.1%}")
        st.progress(float(prob))
        if prob >= 0.6:
            st.error("⚠️ HIGH risk: this customer is likely to leave. Offer a retention deal.")
        elif prob >= 0.35:
            st.warning("MEDIUM risk: keep an eye on this customer.")
        else:
            st.success("✅ LOW risk: this customer is likely to stay.")

# ---------------- Tab 2: insights ----------------
with tab2:
    df = pd.read_csv("data/Customer-Churn-Records.csv")
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers", f"{len(df):,}")
    m2.metric("Churn rate", f"{df['Exited'].mean():.1%}")
    m3.metric("Average age", f"{df['Age'].mean():.1f}")

    col = st.selectbox("Churn rate by:", ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "Card Type"])
    st.bar_chart(df.groupby(col)["Exited"].mean())

    st.subheader("Model results")
    for name, caption in [("06_evaluation.png", "Confusion matrix & ROC curve"),
                          ("07_feature_importance.png", "What drives churn?")]:
        path = f"reports/{name}"
        if os.path.exists(path):
            st.image(path, caption=caption)
