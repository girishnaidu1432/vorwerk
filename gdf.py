# app_all_in_tabs_combined.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


# ------------------ Helper: Cached Model Loaders ------------------
@st.cache_resource
def load_model(path="best_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        return e

@st.cache_resource
def load_regression_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return e

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Vorwerk Sales Intelligence", layout="wide")
st.title("📊 Vorwerk ML/GEN-AI Dashboard")

# ------------------ Tab Setup ------------------
tab1, tab2 = st.tabs(["🔮 Sales Closure", "📈 Sales Prediction"])

# ------------------ Shared Columns ------------------
required_columns = [
    'Age_Bracket', 'Income_Range', 'City_Tier', 'LeadSource', 'ProductComplexity',
    'PreviousCustomer', 'CompetitorMentioned', 'FollowUps', 'SalesLeaderCloseRate',
    'DaysSinceOrder', 'EmailClicks', 'DealSize', 'ProductPrice', 'BCB', 'PCB', 'NET_EARNINGS'
]

# ======================================================================================
# TAB 1: FULL GEN-AI PIPELINE (STEP 1 TO STEP 3)
# ======================================================================================
with tab1:
    tabs = st.tabs(["⚙️ Step 1: Model Training", "🔍 Step 2: Sales Prediction", "🧠 Step 3: Sales Conversion"])

    # STEP 1
    #with tabs[0]:
        #st.header("⚙️ Step 1 - Model Training")
        #model = load_model("best_model.pkl")
        #upload1 = st.file_uploader("📁 Upload historical order data", type=["csv", "xlsx"])

        #if model and upload1:
            #df1 = pd.read_excel(upload1) if upload1.name.endswith(".xlsx") else pd.read_csv(upload1)
            #st.dataframe(df1.head())

            #if all(col in df1.columns for col in required_columns):
                #if hasattr(model, "predict_proba"):
                    #predictions = model.predict_proba(df1)[:, 1]
                #else:
                    #pred = model.predict(df1)
                    #predictions = (pred - np.min(pred)) / (np.ptp(pred) if np.ptp(pred) != 0 else 1)

                #df1['Closure_Prediction(%)'] = np.round(predictions * 100, 2)
                #df1['LikelyToClose'] = df1['Closure_Prediction(%)'] >= 70
                #closed_df = df1[df1['LikelyToClose'] == True]
                #closed_df.to_csv("closed_orders_history.csv", index=False)
                #st.success("✅ Closed orders saved to closed_orders_history.csv")
                #st.dataframe(closed_df)


    # STEP 1
    with tabs[0]:
        st.header("⚙️ Step 1 - Model Training")
        model = load_model("best_model.pkl")
        upload1 = st.file_uploader("📁 Upload historical order data", type=["csv", "xlsx"])

        if model and upload1:
            df1 = pd.read_excel(upload1) if upload1.name.endswith(".xlsx") else pd.read_csv(upload1)
            st.subheader("📄 Uploaded Data")
            st.dataframe(df1.head())

            if all(col in df1.columns for col in required_columns):
            # Predict
                if hasattr(model, "predict_proba"):
                    predictions = model.predict_proba(df1)[:, 1]
                else:
                    pred = model.predict(df1)
                    predictions = (pred - np.min(pred)) / (np.ptp(pred) if np.ptp(pred) != 0 else 1)

                df1['Closure_Prediction(%)'] = np.round(predictions * 100, 2)
                df1['LikelyToClose'] = df1['Closure_Prediction(%)'] >= 70

            # Show all data
                st.success("✅ Predictions generated")
                st.dataframe(df1.sort_values(by='Closure_Prediction(%)', ascending=False))

            # Save only closed orders (≥70%)
                closed_df = df1[df1['LikelyToClose']]
                closed_df.to_csv("closed_orders_history.csv", index=False)
                st.info(f"💾 {len(closed_df)} closed orders saved to closed_orders_history.csv")


    # STEP 2
    with tabs[1]:
        st.header("🔍 Step 2 - Sales Prediction")
        upload2 = st.file_uploader("📁 Upload test orders file", type=["csv", "xlsx"])

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

    # STEP 3
    with tabs[2]:
        st.header("🧠 Step 3 - Sales Conversion Analysis")
        try:
            predicted_df = pd.read_csv("predicted_orders.csv")
            closed_df = pd.read_csv("closed_orders_history.csv")
        except Exception:
            st.warning("Please complete Steps 1 and 2 first.")
            st.stop()

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
                        prompt = f"""...Role: You are an expert AI Sales Assistant trained on historical sales data and predictive models. Your objective is to help the sales team improve their chances of closing open orders by providing actionable insights rooted in successful past patterns.
Given:
Top Performing orders (with higher closure prediction):
{top_records}
current Order:
{current_order}
Task:
Compare the current order with all Top-performing orders.
Based on the differences in key influencing features, recommend a list of targeted actions the salesperson can take to increase the likelihood of closing the deal.
These actions should be:
- Data-driven
- Personalized to the record's context
- Feasible within the sales process
Keep the response action-oriented, clear, and backed by feature-level insights.
Please summarise the response as action items with a reasoning....."""
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


# ======================================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Function to load model
@st.cache_resource
def load_regression_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return e

# Tab selector in the main page
tab1, tab2, tab3 = st.tabs(["Tab 1", "Monthly Sales Prediction", "Tab 3"])

# Sidebar: choose tab (only Tab 2 has inputs)
active_tab = st.session_state.get("active_tab", "Tab 1")

# Tab 2 code
with tab2:
    st.header("📈 Monthly Sales Prediction")

    # Sidebar inputs only for Tab 2
    selected_country_1 = st.sidebar.selectbox(
        "Select Country", ["USA", "Malasiya", "Taiwan"], key="country_tab2"
    )
    selected_product_1 = st.sidebar.selectbox(
        "Select Product", ["Kobold", "Thermomix"], key="product_tab2"
    )
    run_button = st.sidebar.button("Run Prediction", key="run_prediction_tab2")

    # Mappings
    age_map = {'25-45': 1, '46-59': 2, '60+': 3}
    income_map = {'20K-40K': 1, '40K-60K': 2, '60K-80K': 3, '80K+': 4}
    city_map = {'Tier-1': 3, 'Tier-2': 2, 'Tier-3': 1}
    product_map = {'Kobold': 0, 'Thermomix': 1}

    # Load model
    model_path = "lasso_regression_model.pkl"
    reg_model = load_regression_model(model_path)

    if isinstance(reg_model, Exception):
        st.error(f"❌ Could not load regression model: {model_path} -> {reg_model}")
    else:
        st.success(f"✅ Loaded model: Best Performing Model")

    # Prediction logic
    if run_button:
        features = np.array([[age_map['25-45'], income_map['40K-60K'],
                               city_map['Tier-1'], product_map[selected_product_1]]])
        prediction = reg_model.predict(features)[0]
        st.metric("Predicted Sales", f"{prediction:,.2f}")

        # Chart
        months = np.arange(1, 13)
        sales = prediction + np.random.randn(12) * 1000
        fig, ax = plt.subplots()
        ax.plot(months, sales, marker='o')
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        ax.set_title("Monthly Sales Forecast")
        st.pyplot(fig)



