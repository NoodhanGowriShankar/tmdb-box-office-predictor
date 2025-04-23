
# TMDB Box Office Prediction - Final Project
# Models: Linear Regression, Random Forest, XGBoost

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("Project 2_data.csv")

# Feature engineering functions
def count_items(json_str):
    try:
        items = ast.literal_eval(json_str)
        return len(items) if isinstance(items, list) else 0
    except:
        return 0

def extract_director(json_str):
    try:
        crew = ast.literal_eval(json_str)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
        return None
    except:
        return None

# Feature Engineering
df['num_genres'] = df['genres'].apply(count_items)
df['num_cast'] = df['cast'].apply(count_items)
df['num_crew'] = df['crew'].apply(count_items)
df['num_keywords'] = df['Keywords'].apply(count_items)
df['has_homepage'] = df['homepage'].notna().astype(int)
df['has_collection'] = df['belongs_to_collection'].notna().astype(int)
df['has_tagline'] = df['tagline'].notna().astype(int)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month
df['release_year'] = df['release_date'].dt.year
df['release_dayofweek'] = df['release_date'].dt.dayofweek
df['log_budget'] = np.log1p(df['budget'])
df['log_revenue'] = np.log1p(df['revenue'])
df['director'] = df['crew'].apply(extract_director)

# Modeling dataset
model_data = df.dropna(subset=['log_budget', 'log_revenue'])
features = [
    'log_budget', 'popularity', 'runtime', 'num_genres', 'num_cast',
    'num_crew', 'num_keywords', 'has_homepage', 'has_collection',
    'has_tagline', 'release_month', 'release_year', 'release_dayofweek'
]
X = model_data[features]
y = model_data['log_revenue']

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# --- Model 1: Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmsle_lr = np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(y_pred_lr)))
r2_lr = r2_score(y_test, y_pred_lr)

# --- Model 2: Random Forest Regressor ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmsle_rf = np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(y_pred_rf)))
r2_rf = r2_score(y_test, y_pred_rf)

# --- Model 3: XGBoost Regressor ---
xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
rmsle_xgb = np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(y_pred_xgb)))
r2_xgb = r2_score(y_test, y_pred_xgb)

# Save XGBoost model and imputer for use in Streamlit app
xgb.save_model("xgb_model.json")

import joblib
joblib.dump(imputer, "imputer.pkl")

print("✅ Model and imputer saved for Streamlit.")


# --- Results ---
print("Linear Regression:")
print("RMSLE:", rmsle_lr)
print("R² Score:", r2_lr)

print("\nRandom Forest:")
print("RMSLE:", rmsle_rf)
print("R² Score:", r2_rf)

print("\nXGBoost:")
print("RMSLE:", rmsle_xgb)
print("R² Score:", r2_xgb)

# === Show XGBoost Predictions ===
try:
    import numpy as np
    import pandas as pd

    y_pred_xgb = xgb.predict(X_test)

    actual_revenue = np.expm1(y_test).reset_index(drop=True)
    predicted_revenue = pd.Series(np.expm1(y_pred_xgb))

    comparison_df = pd.DataFrame({
        "Actual Revenue": actual_revenue,
        "Predicted Revenue": predicted_revenue
    })

    # Print confirmation and first 10 rows
    print("\n===== Actual vs Predicted (XGBoost) =====")
    print(comparison_df.head(10).to_string(index=False))

    # Optional: Export predictions
    comparison_df.to_csv("xgboost_predictions.csv", index=False)
    print("\n✅ Predictions saved as 'xgboost_predictions.csv' in project folder.")

except Exception as e:
    print(f"\n❌ Error while generating predictions: {e}")

# --- Feature Importance for XGBoost ---
feature_importance = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()




