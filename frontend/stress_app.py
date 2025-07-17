import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Flask API Endpoint
API_URL = "https://biology-backend-production.up.railway.app/predict"


# --- Page Configuration ---
st.set_page_config(page_title="Stress Detector", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        max-width: 600px;
        margin: auto;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    h1, h2 {
        color: #333;
        text-align: center;
    }

    input, .stNumberInput>div>div>input {
        background-color: #fafafa;
        border-radius: 8px;
        border: 1px solid #dbdbdb;
        padding: 0.8rem;
        margin-bottom: 1rem;
        font-size: 16px;
    }

    .stButton>button {
        background-color: #0095f6;
        color: white;
        border-radius: 8px;
        padding: 1rem;
        width: 100%;
        border: none;
        font-size: 16px;
        cursor: pointer;
    }

    .stButton>button:hover {
        background-color: #007bb5;
    }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit Sidebar Navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose Section", ["Stress Prediction", "Visual Insights"])

# --- Stress Prediction Section ---
if section == "Stress Prediction":
    st.title("Human Stress Detector")
    st.subheader("Based on Sleep Data")

    # --- Input Fields ---
    snoring_rate = st.number_input("Snoring Rate", min_value=0.0, max_value=100.0, step=0.1)
    body_temperature = st.number_input("Body Temperature (°F)", min_value=77.0, max_value=98.0, step=0.1)
    sleeping_hours = st.number_input("Sleeping Hours", min_value=0.0, max_value=24.0, step=0.1)
    heart_rate = st.number_input("Heart Rate Per Half Minute", min_value=30.0, max_value=200.0, step=1.0)

    # --- Predict Button ---
    if st.button("Predict Stress Level"):
        payload = {
            "heart_rate": heart_rate,
            "sleeping_hours": sleeping_hours,
            "snoring_rate": snoring_rate,
            "body_temperature": body_temperature
        }

        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(f"Predicted Stress Level: {result['description']} (Level {result['stress_level']})")

        except Exception as e:
            st.error("Could not connect to the prediction server.")
            st.error(f"Details: {e}")

# --- Data Visualization Section ---
elif section == "Visual Insights":
    st.markdown("---")
    st.header("📊 Visual Insights from Sleep Data")

    # Path to your global CSV file
    csv_path = "stress_data.csv"  # Update this if needed

    try:
        df = pd.read_csv(csv_path)

        # Adjust stress levels to 1–5 if they start from 0
        if df['stresslevel'].min() == 0:
            df['stresslevel'] += 1

        # List of features to analyze
        features = ['heartrate', 'sleepinghours', 'snoring', 'bodytemperature']

        # Create a new dataframe to hold the correlation values between features and stress level
        correlation_data = df[features + ['stresslevel']]

        # Compute the correlation matrix
        correlation_matrix = correlation_data.corr()

        # Plotting the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap of Features and Stress Level")

        # Show the heatmap
        st.pyplot(plt)

        # Button to show line plots
        if st.button("Logical Relation Behind Parameters"):
            # Line plots for each feature
            st.markdown("### Line Plots for Features vs Stress Level")

            # Create a figure with subplots
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.lineplot(data=df, x='sleepinghours', y='stresslevel', ax=ax, label='Sleeping Hours', color='blue')
            sns.lineplot(data=df, x='heartrate', y='stresslevel', ax=ax, label='Heart Rate', color='green')
            sns.lineplot(data=df, x='snoring', y='stresslevel', ax=ax, label='Snoring Rate', color='red')
            sns.lineplot(data=df, x='bodytemperature', y='stresslevel', ax=ax, label='Body Temperature', color='purple')

            ax.set_title("Line Plots: Features vs Stress Level")
            ax.set_xlabel("Feature Values")
            ax.set_ylabel("Stress Level")
            ax.legend()

            st.pyplot(fig)

    except Exception as e:
        st.error("Could not load or plot data from the CSV.")
        st.error(f"Details: {e}")
