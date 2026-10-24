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

# Real Starbucks drinks from the dataset (Total Fat, Sugars, Protein, actual calories)
examples = {
    "🟢 Brewed Coffee": [0.1, 0.0, 1.0, "Brewed Coffee (Grande)", 5],
    "🟡 Caffè Latte": [7.0, 17.0, 12.0, "Caffè Latte (Grande, 2% Milk)", 190],
    "🔴 White Chocolate Mocha": [15.0, 74.0, 19.0, "White Chocolate Mocha (Venti, 2% Milk)", 510],
}

# Default input values
if "fat" not in st.session_state:
    st.session_state.fat = 3.0
    st.session_state.sugars = 30.0
    st.session_state.protein = 7.0
    st.session_state.example = None
    st.session_state.show_result = False


def use_example(name):
    fat, sugar, protein, _, _ = examples[name]
    st.session_state.fat = fat
    st.session_state.sugars = sugar
    st.session_state.protein = protein
    st.session_state.example = name
    st.session_state.show_result = True


def predict_clicked():
    st.session_state.example = None
    st.session_state.show_result = True


def values_changed():
    st.session_state.example = None
    st.session_state.show_result = False


# Example buttons
st.subheader("☕ Try an Example")
st.caption("Click a real Starbucks drink — its values fill in automatically and the model predicts its calorie level.")

cols = st.columns(len(examples))
for col, name in zip(cols, examples):
    with col:
        st.button(name, on_click=use_example, args=(name,), width="stretch")

# User inputs
st.subheader("🧮 Enter Drink Nutrition Values")

st.number_input(
    "Total Fat (g)", min_value=0.0, max_value=20.0, key="fat", on_change=values_changed
)

st.number_input(
    "Sugars (g)", min_value=0.0, max_value=90.0, key="sugars", on_change=values_changed
)

st.number_input(
    "Protein (g)", min_value=0.0, max_value=20.0, key="protein", on_change=values_changed
)

st.button("Predict Calorie Level", on_click=predict_clicked)

# Predict calorie level
if st.session_state.show_result:
    data = pd.DataFrame([[
        st.session_state.fat,
        st.session_state.sugars,
        st.session_state.protein
    ]], columns=features)

    level = model.predict(data)[0]

    if level == "Low":
        st.success("Predicted Calorie Level: LOW (0 – 150 kcal) 🟢")
    elif level == "Medium":
        st.warning("Predicted Calorie Level: MEDIUM (151 – 250 kcal) 🟡")
    else:
        st.error("Predicted Calorie Level: HIGH (more than 250 kcal) 🔴")

    if st.session_state.example:
        _, _, _, drink, calories = examples[st.session_state.example]
        st.info(f"Example drink: **{drink}** — actual calories: **{calories} kcal**")

st.caption("Calorie levels: Low = 0–150 kcal · Medium = 151–250 kcal · High = more than 250 kcal")
