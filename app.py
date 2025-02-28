import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="QEMO", layout="wide")

# Inject CSS for a full-page background with overlay hi
st.markdown(
    """
    <style>
    .stApp {
        background: url("./Wind+Solar Energy.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    .overlay {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and introductory text in an overlay container
st.markdown(
    """
    <div class="overlay">
    <h1>QEMO - Quarterly Energy Forecasting and Optimization</h1>
    <p>
        <strong>QEMO</strong> optimizes hybrid power plant operations by forecasting seasonal variations in solar and wind energy.
        <br>
        <strong>Key Features:</strong> Quarterly Models (Q1–Q4), Meta-model Ensemble, Hybrid Optimization.
        <br>
        <strong>Evaluation Metrics:</strong> RMSE=0.7525, MAE=0.4702, R²=0.6708.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load prediction data from pickle
with open("predictions.pkl", "rb") as f:
    y_pred, y_fused = pickle.load(f)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Overview", "Forecasts", "Recommendations"])

# Helper function to wrap content in an overlay container
def content_container(content_func):
    st.markdown('<div class="overlay">', unsafe_allow_html=True)
    content_func()
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# OVERVIEW PAGE
# ----------------------------
if page == "Overview":
    def overview_content():
        st.header("Project Overview")
        # Display background image again (optional if you want a different image)
        st.image("./Wind+Solar Energy.jpg", width=800)
        st.markdown(
            """
            **QEMO** leverages deep learning and ensemble techniques to forecast and optimize energy production from solar and wind sources.
            The project builds separate quarterly models, fuses them using a meta-model, and provides actionable recommendations for hybrid power plant operations.
            """
        )
    content_container(overview_content)

# ----------------------------
# FORECASTS PAGE
# ----------------------------
elif page == "Forecasts":
    def forecasts_content():
        st.header("Energy Forecasts")
        num_samples = len(y_pred)
        dates = pd.date_range("2025-03-01", periods=num_samples, freq="h")
        df_forecasts = pd.DataFrame({
            "Datetime": dates,
            "Predicted": y_pred.flatten(),
            "Actual": y_fused[:num_samples]
        })
        
        # Create 4 subplots in a 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1) Hybrid Model Forecasts (Predicted only)
        axs[0, 0].plot(df_forecasts["Datetime"], df_forecasts["Predicted"], marker='o', color='green')
        axs[0, 0].set_title("Hybrid Model Forecasts")
        axs[0, 0].set_xlabel("Datetime")
        axs[0, 0].set_ylabel("Predicted Power Output")
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # 2) Predicted vs. Actual Over Time
        axs[0, 1].plot(df_forecasts["Datetime"], df_forecasts["Predicted"], label="Predicted", marker='o')
        axs[0, 1].plot(df_forecasts["Datetime"], df_forecasts["Actual"], label="Actual", marker='o')
        axs[0, 1].set_title("Predicted vs. Actual Over Time")
        axs[0, 1].set_xlabel("Datetime")
        axs[0, 1].set_ylabel("Power Output")
        axs[0, 1].legend()
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # 3) Distribution of Predicted Values
        sns.histplot(df_forecasts["Predicted"], kde=True, ax=axs[1, 0], color="blue")
        axs[1, 0].set_title("Distribution of Predicted Values")
        axs[1, 0].set_xlabel("Predicted Power Output")
        axs[1, 0].set_ylabel("Frequency")
        
        # 4) Predicted vs. Actual Scatter Plot
        axs[1, 1].scatter(df_forecasts["Actual"], df_forecasts["Predicted"], alpha=0.5)
        axs[1, 1].set_title("Predicted vs. Actual Scatter Plot")
        axs[1, 1].set_xlabel("Actual Power Output")
        axs[1, 1].set_ylabel("Predicted Power Output")
        
        fig.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### Forecast Data (First 10 Rows)")
        st.write(df_forecasts.head(10))
    content_container(forecasts_content)

# ----------------------------
# RECOMMENDATIONS PAGE
# ----------------------------
elif page == "Recommendations":
    def recommendations_content():
        st.header("Operational Recommendations")
        def generate_recommendations(predictions):
            recs = []
            for pred in predictions:
                if pred > 500:
                    recs.append("High Production: Maximize Grid Contribution")
                elif pred > 300:
                    recs.append("Moderate Production: Optimize Storage and Grid")
                else:
                    recs.append("Low Production: Store Energy for Peak Demand")
            return recs
        recs = generate_recommendations(y_pred)
        num_samples = len(y_pred)
        dates = pd.date_range("2025-03-01", periods=num_samples, freq="h")
        df_recs = pd.DataFrame({
            "Datetime": dates,
            "Total Forecast": y_pred.flatten(),
            "Recommendation": recs
        })
        st.markdown("### Recommendations (First 10 Rows)")
        st.write(df_recs.head(10))
    content_container(recommendations_content)
