import os
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Starbucks Calorie Predictor",
    page_icon="☕"
)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "starbucks_decision_tree.pkl")
saved = joblib.load(model_path)
model = saved["model"]
features = saved["features"]

st.title("☕ Starbucks Calorie Level Prediction")
st.write("Decision Tree Machine Learning Model trained on the Starbucks drinks menu")

# EDA results: Top 5 drinks with most / least calories
st.subheader("📊 Top 5 Starbucks Drinks by Calories")

most = pd.DataFrame(saved["top5_most"], columns=["Drink", "Calories"])
least = pd.DataFrame(saved["top5_least"], columns=["Drink", "Calories"])

calorie_bar = {
    "Calories": st.column_config.ProgressColumn(
        "Average Calories (kcal)", format="%.0f kcal",
        min_value=0, max_value=float(most["Calories"].max())
    )
}

tab1, tab2 = st.tabs(["🔺 Most Calories", "🔻 Least Calories"])
with tab1:
    st.dataframe(most, hide_index=True, width="stretch", column_config=calorie_bar)
with tab2:
    st.dataframe(least, hide_index=True, width="stretch", column_config=calorie_bar)

# User inputs
st.subheader("🧮 Enter Drink Nutrition Values")

total_fat = st.number_input(
    "Total Fat (g)", min_value=0.0, max_value=20.0, value=3.0
)

sugars = st.number_input(
    "Sugars (g)", min_value=0.0, max_value=90.0, value=30.0
)

protein = st.number_input(
    "Protein (g)", min_value=0.0, max_value=20.0, value=7.0
)

# Predict calorie level
if st.button("Predict Calorie Level"):
    data = pd.DataFrame([[
        total_fat,
        sugars,
        protein
    ]], columns=features)

    level = model.predict(data)[0]

    if level == "Low":
        st.success("Predicted Calorie Level: LOW (0 – 150 kcal) 🟢")
    elif level == "Medium":
        st.warning("Predicted Calorie Level: MEDIUM (151 – 250 kcal) 🟡")
    else:
        st.error("Predicted Calorie Level: HIGH (more than 250 kcal) 🔴")

st.caption("Calorie levels: Low = 0–150 kcal · Medium = 151–250 kcal · High = more than 250 kcal")
