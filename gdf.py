# app_all_in_tabs.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import openai
import json
import os

# ------------------ Azure OpenAI Configuration ------------------
openai.api_key = "14560021aaf84772835d76246b53397a"
openai.api_base = "https://amrxgenai.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2024-02-15-preview'
deployment_name = 'gpt'

# ------------------ Helper: cached model loader ------------------
@st.cache_resource
def load_model(path="best_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        return e

# ------------------ App Title ------------------
st.set_page_config(page_title="Multi-Tab GEN-AI Apps", layout="wide")
st.title("📊 Vorwerk ML/GEN-AI Demos")
tabs = st.tabs(["⚙️ Step 1: Model Training", "🔍 Step 2: Sales Prediction", "🧠 Step 3: Sales Conversion"])

required_columns = [
    'Age_Bracket', 'Income_Range', 'City_Tier', 'LeadSource', 'ProductComplexity',
    'PreviousCustomer', 'CompetitorMentioned', 'FollowUps', 'SalesLeaderCloseRate',
    'DaysSinceOrder', 'EmailClicks', 'DealSize', 'ProductPrice', 'BCB', 'PCB', 'NET_EARNINGS'
]

# ------------------ STEP 1: Model Training ------------------
with tabs[0]:
    st.header("⚙️ Step 1 - Model Training")
    model = load_model("best_model.pkl")
    upload1 = st.file_uploader("📁 Upload historical order data (orders-sales-data)", type=["csv", "xlsx"])

    if model and upload1:
        df1 = pd.read_excel(upload1) if upload1.name.endswith(".xlsx") else pd.read_csv(upload1)
        st.dataframe(df1.head())

        if all(col in df1.columns for col in required_columns):
            if hasattr(model, "predict_proba"):
                predictions = model.predict_proba(df1)[:, 1]
            else:
                pred = model.predict(df1)
                predictions = (pred - np.min(pred)) / (np.ptp(pred) if np.ptp(pred) != 0 else 1)

            df1['Closure_Prediction(%)'] = np.round(predictions * 100, 2)
            df1['LikelyToClose'] = df1['Closure_Prediction(%)'] >= 70
            closed_df = df1[df1['LikelyToClose'] == True]
            closed_df.to_csv("closed_orders_history.csv", index=False)
            st.success("✅ Closed orders (>=70%) saved to closed_orders_history.csv")
            st.dataframe(closed_df)

# ------------------ STEP 2: Sales Prediction ------------------
with tabs[1]:
    st.header("🔍 Step 2 - Sales Prediction")
    upload2 = st.file_uploader("📁 Upload test orders file (orders_test.csv)", type=["csv", "xlsx"])

    if model and upload2:
        df2 = pd.read_excel(upload2) if upload2.name.endswith(".xlsx") else pd.read_csv(upload2)
        st.dataframe(df2.head())

        if all(col in df2.columns for col in required_columns):
            if hasattr(model, "predict_proba"):
                predictions = model.predict_proba(df2)[:, 1]
            else:
                pred = model.predict(df2)
                predictions = (pred - np.min(pred)) / (np.ptp(pred) if np.ptp(pred) != 0 else 1)

            df2['Closure_Prediction(%)'] = np.round(predictions * 100, 2)
            df2['LikelyToClose'] = df2['Closure_Prediction(%)'] >= 70

            st.success("✅ Predictions generated")
            st.dataframe(df2.sort_values(by='Closure_Prediction(%)', ascending=False))

            df2.to_csv("predicted_orders.csv", index=False)
            st.download_button("📥 Download Predicted Orders", df2.to_csv(index=False).encode(), file_name="predicted_orders.csv")

# ------------------ STEP 3: Sales Conversion ------------------
with tabs[2]:
    st.header("🧠 Step 3 - Sales Conversion Analysis")
    try:
        predicted_df = pd.read_csv("predicted_orders.csv")
        closed_df = pd.read_csv("closed_orders_history.csv")
    except Exception as e:
        st.warning("Please complete Steps 1 and 2 first.")
        st.stop()

    # Filter for high-prediction orders
    high_pred_df = predicted_df[predicted_df['Closure_Prediction(%)'] >= 70]

    if high_pred_df.empty:
        st.warning("No orders found with prediction ≥ 70%.")
    else:
        st.dataframe(high_pred_df)

        for idx, row in high_pred_df.iterrows():
            current_order = row[required_columns + ['Closure_Prediction(%)']].to_dict()
            top_records = closed_df[required_columns + ['Closure_Prediction(%)']].to_dict(orient="records")

            with st.expander(f"📝 Analyze Order {idx+1} ({current_order['Closure_Prediction(%)']}%)"):
                st.json(current_order)

                if st.button(f"🔍 Generate Insights for Order {idx+1}", key=f"btn_{idx}"):
                    prompt = f"""
Role: You are an expert AI Sales Assistant trained on historical sales data and predictive models. Your objective is to help the sales team improve their chances of closing open orders by providing actionable insights rooted in successful past patterns.

Given:

Top Performing orders (with higher closure prediction):
{top_records}

Current Order:
{current_order}

Tasks:

1. Prescriptive Actions for Current Order:
Compare the current order with Top-performing order . Based on the differences in key influencing features, recommend a list of targeted actions the salesperson can take to increase the likelihood of closing the deal. These actions should be:
- Data-driven
- Personalized to the record's context
- Feasible within the sales process

2. Summary of Sales Closure Drivers:
Based on high-prediction records, summarize the critical success factors that most frequently led to order closure.

3. Root Cause Analysis – Top vs Current order Prediction:
Explain why the highest-prediction orders performed better than the current one. Pinpoint the missing or under-leveraged features in the current case and recommend how to align them.

Keep the response action-oriented, clear, and backed by feature-level insights.
"""

                    with st.spinner("Generating insights..."):
                        try:
                            response = openai.ChatCompletion.create(
                                engine=deployment_name,
                                messages=[
                                    {"role": "system", "content": "You are a data-driven sales assistant."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.7,
                                max_tokens=1000
                            )
                            message = response.choices[0].message.get('content')
                            st.success("✅ Insights Generated:")
                            st.markdown(message)
                        except Exception as e:
                            st.error(f"❌ Error from Azure OpenAI: {str(e)}")
