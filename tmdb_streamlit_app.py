
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

# Load trained model and imputer
@st.cache_resource
def load_model():
    model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4)
    model.load_model("xgb_model.json")
    imputer = joblib.load("imputer.pkl")
    return model, imputer

model, imputer = load_model()

st.title("🎬 TMDB Box Office Revenue Predictor")

st.markdown("Enter movie metadata below and get a predicted worldwide revenue.")

# User inputs
log_budget = st.number_input("Budget (in USD)", min_value=0, value=10000000)
popularity = st.slider("TMDB Popularity Score", 0.0, 1000.0, 50.0)
runtime = st.slider("Runtime (minutes)", 60, 240, 120)
num_genres = st.slider("Number of Genres", 1, 5, 2)
num_cast = st.slider("Number of Cast Members", 1, 20, 5)
num_crew = st.slider("Number of Crew Members", 1, 50, 10)
num_keywords = st.slider("Number of Plot Keywords", 0, 20, 3)
has_homepage = st.selectbox("Has Homepage?", ["No", "Yes"]) == "Yes"
has_collection = st.selectbox("Part of a Collection?", ["No", "Yes"]) == "Yes"
has_tagline = st.selectbox("Has Tagline?", ["No", "Yes"]) == "Yes"
release_month = st.slider("Release Month", 1, 12, 6)
release_year = st.slider("Release Year", 1990, 2025, 2010)
release_dayofweek = st.slider("Release Day of Week (0=Mon, 6=Sun)", 0, 6, 4)

# Feature vector
input_data = pd.DataFrame([[
    np.log1p(log_budget), popularity, runtime, num_genres, num_cast,
    num_crew, num_keywords, int(has_homepage), int(has_collection),
    int(has_tagline), release_month, release_year, release_dayofweek
]], columns=[
    'log_budget', 'popularity', 'runtime', 'num_genres', 'num_cast',
    'num_crew', 'num_keywords', 'has_homepage', 'has_collection',
    'has_tagline', 'release_month', 'release_year', 'release_dayofweek'
])

# Impute missing values (if any)
input_data_imputed = imputer.transform(input_data)

# Predict revenue
if st.button("Predict Revenue"):
    prediction_log = model.predict(input_data_imputed)[0]
    prediction = np.expm1(prediction_log)
    st.success(f"🎉 Predicted Worldwide Revenue: **${prediction:,.2f}**")
