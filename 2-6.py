# --- Fixed Default Model ---
model_option = "Ridge Regression"  # Default model
model_files = {
    "Linear Regression": "linear_regression_model.pkl",
    "Ridge Regression": "ridge_regression_model.pkl",
    "Lasso Regression": "lasso_regression_model.pkl",
    "ElasticNet Regression": "elasticnet_regression_model.pkl"
}
model_path = model_files[model_option]
try:
    model = joblib.load(model_path)
    # Optionally show success in main area
    st.success(f"✅ Loaded model: {model_option}")
except FileNotFoundError:
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()
