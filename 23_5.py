import streamlit as st
import joblib
import numpy as np
import pandas as pd
import random
import openai
import json
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt

# --- Azure OpenAI Configuration ---
openai.api_key = "14560021aaf84772835d76246b53397a"
openai.api_base = "https://amrxgenai.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2024-02-15-preview'
deployment_name = 'gpt'

# --- Mapping Constants ---
age_map = {'25-45': 1, '46-59': 2, '60+': 3}
income_map = {'20K-40K': 1, '40K-60K': 2, '60K-80K': 3, '80K+': 4}
city_map = {'Tier-1': 3, 'Tier-2': 2, 'Tier-3': 1}
product_map = {'Kobald': 0, 'Thermomix': 1}

# --- Sidebar Model Selection ---
st.sidebar.title("⚙️ Settings")
model_option = st.sidebar.selectbox(
    "Select Model to Load:",
    ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression"]
)
model_files = {
    "Linear Regression": "linear_regression_model.pkl",
    "Ridge Regression": "ridge_regression_model.pkl",
    "Lasso Regression": "lasso_regression_model.pkl",
    "ElasticNet Regression": "elasticnet_regression_model.pkl"
}

model_path = model_files[model_option]
try:
    model = joblib.load(model_path)
    st.sidebar.success(f"✅ Loaded model: {model_option}")
except FileNotFoundError:
    st.sidebar.error(f"❌ Model file not found: {model_path}")
    st.stop()

st.title("📈Vorverk GEN-AI Use Cases")

# --- Tabs UI ---
tabs = st.tabs([ "🌍Sales Prediction for New Country","💰 Predict Product Price"])

with tabs[0]:
    st.header("🌍 Gen AI-Feature Extraction")

    selected_country = st.selectbox("Select Country", ["USA", "Malasiya", "Taiwan"])
    selected_product = st.selectbox("Select Product", ["Kobald", "Thermomix"])
    product_code = product_map[selected_product]

    if st.button("Generate & Predict", key="gen_ai"):
        prompt_data = f"""
        Generate 5 rows of synthetic customer segment data for product '{selected_product}' in {selected_country}.
        Each row should include: Age_Bracket (25-45, 46-59, 60+), Income_Range (20K-40K, 40K-60K, 60K-80K, 80K+),
        City_Tier (Tier-1, Tier-2, Tier-3), Age_Group_Pct (0.1 to 0.9), Income_Pct (0.1 to 0.9), Product Names.
        Return data as a JSON array without any explanation.
        """
        try:
            response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=[{"role": "system", "content": "You are a helpful AI data assistant."},
                          {"role": "user", "content": prompt_data}],
                temperature=0.7,
                max_tokens=1000
            )
            content = response['choices'][0]['message']['content']
            synthetic_data = json.loads(content)

            df_gen = pd.DataFrame(synthetic_data)
            df_gen['Product Names'] = selected_product
            df_gen['Age_Level'] = df_gen['Age_Bracket'].map(age_map)
            df_gen['Income_Level'] = df_gen['Income_Range'].map(income_map)
            df_gen['City_Tier_Level'] = df_gen['City_Tier'].map(city_map)
            df_gen['Product_Code'] = product_code

            features = df_gen[['Age_Group_Pct', 'Income_Pct', 'Age_Level', 'Income_Level', 'City_Tier_Level', 'Product_Code']]
            df_gen['Predicted_Qty'] = model.predict(features).astype(int)
            df_gen['Purchase_Intent'] = df_gen['Predicted_Qty'].apply(lambda x: "Yes" if x > 5 else "No")
            df_gen['Confidence'] = df_gen['Predicted_Qty'].apply(lambda x: min(1.0, max(0.0, x / 10)))

            st.session_state['df_gen'] = df_gen
        except Exception as e:
            st.error(f"❌ Error generating AI data or analysis: {e}")

    if 'df_gen' in st.session_state:
        df_gen = st.session_state['df_gen']
        df_gen_yes = df_gen[df_gen['Purchase_Intent'] == "Yes"]
        st.subheader("📊 Purchase Intent: Yes")
        st.dataframe(df_gen_yes[['Age_Bracket', 'Income_Range', 'City_Tier', 'Product Names']])

        if st.button("Predict Quantity", key="predict_quantity_genai"):
            st.dataframe(df_gen_yes[['Age_Bracket', 'Income_Range', 'City_Tier', 'Product Names', 'Predicted_Qty', 'Confidence']])

            # Plotting for Age_Bracket vs Predicted_Qty
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Age_Bracket', y='Predicted_Qty', data=df_gen_yes)
            plt.title('Predicted Quantity by Age Bracket')
            st.pyplot(plt)

            # Plotting for Income_Range vs Predicted_Qty
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Income_Range', y='Predicted_Qty', data=df_gen_yes)
            plt.title('Predicted Quantity by Income Range')
            st.pyplot(plt)

            # Plotting for City_Tier vs Predicted_Qty
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='City_Tier', y='Predicted_Qty', data=df_gen_yes)
            plt.title('Predicted Quantity by City Tier')
            st.pyplot(plt)

            # Plotting for Product Names vs Predicted_Qty
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Product Names', y='Predicted_Qty', data=df_gen_yes)
            plt.title('Predicted Quantity by Product Names')
            st.pyplot(plt)

        # Excel Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_gen.to_excel(writer, index=False, sheet_name='AI Data')
        output.seek(0)
        st.download_button("📥 Download as Excel", data=output.getvalue(), file_name="ai_generated_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------------------

with tabs[1]:
    st.header("💰 Predict Product Price")

    selected_country = st.selectbox("Select Country", ["USA", "Canada", "Germany", "India", "Brazil"], key="price_country")
    selected_product = st.selectbox("Select Product", ["Kobald", "Thermomix"], key="price_product")

    if st.button("Predict Product Price", key="predict_price_tab"):
        analysis_prompt = f"""
        For country {selected_country}: and Product:{selected_product} , predict the price based on below factors:
        1. GDP per Capita
        2. Purchasing Power
        3. Demographics (Age, Family Size, Lifestyle)
        4. Market Segmentation Strategy
        5. Price Anchoring vs. Competitors.
        Predict the price range only.
        Give the output that is product predicted price only with currency of {selected_country}.
        Give one justification point for the price prediction for each factor above.
        List the competitors with product names with price and similar features and similar price range to {selected_product}.
        If the competitors product has lower price than {selected_product}, then give the reason for that.
        If {selected_product} is expensive than competitors, then give the reason for that.
        """
        try:
            analysis_response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=[{"role": "system", "content": "You are price predictor"},
                          {"role": "user", "content": analysis_prompt}],
                temperature=0.7,
                max_tokens=900
            )
            analysis_text = analysis_response['choices'][0]['message']['content']
            st.markdown("### 📌 Product AVG Predicted Price")
            st.markdown(analysis_text)
        except Exception as e:
            st.error(f"❌ Error predicting price: {e}")
