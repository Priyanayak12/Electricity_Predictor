import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import kagglehub
import os

# 🎯 App Configuration
st.set_page_config(page_title="Electricity Consumption Predictor", layout="centered")

st.title("⚡ Electricity Consumption Prediction App")

st.write("""
This app predicts **electricity consumption** using trained Machine Learning models.  
You can choose between **Daily** or **Weekly** prediction modes below.
""")

# 🧭 Sidebar to select mode
mode = st.sidebar.selectbox("Select Prediction Mode", ["Daily", "Weekly"])

# Load datasets (to plot trends)
path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
file_path = os.path.join(path, "COMED_hourly.csv")

# Load and prepare dataset
df = pd.read_csv(file_path)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.dropna()
df = df.set_index('Datetime')

# Resample to daily
daily = df.resample('D')['COMED_MW'].sum().reset_index()
daily = daily.rename(columns={'COMED_MW': 'Daily_Consumption'})

# Resample to weekly
weekly = daily.resample('W', on='Datetime')['Daily_Consumption'].sum().reset_index()
weekly = weekly.rename(columns={'Daily_Consumption': 'Weekly_Consumption'})

# ----------------------------------------------------
# MODE 1️⃣: DAILY PREDICTION
# ----------------------------------------------------
if mode == "Daily":
    st.subheader("📅 Daily Electricity Consumption Prediction")
    model = joblib.load("electricity_daily_model.pkl")

    year = st.number_input("Enter Year", min_value=2010, max_value=2030, value=2024)
    month = st.number_input("Enter Month (1-12)", min_value=1, max_value=12, value=11)
    day = st.number_input("Enter Day (1-31)", min_value=1, max_value=31, value=13)

    date_obj = datetime(year, month, day)
    weekday = date_obj.weekday()

    X_input = pd.DataFrame([[day, month, year, weekday]],
                           columns=['Day', 'Month', 'Year', 'Weekday'])

    if st.button("🔮 Predict Daily Consumption"):
        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Daily Consumption: **{prediction:.2f} MW**")

        # 🧭 Graph: show trend + predicted point
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily['Datetime'], daily['Daily_Consumption'], label='Historical Data', color='blue')
        ax.scatter(date_obj, prediction, color='red', label='Predicted', s=100)
        ax.set_xlabel("Date")
        ax.set_ylabel("Electricity Consumption (MW)")
        ax.set_title("Daily Electricity Consumption Trend")
        ax.legend()
        st.pyplot(fig)

# ----------------------------------------------------
# MODE 2️⃣: WEEKLY PREDICTION
# ----------------------------------------------------
elif mode == "Weekly":
    st.subheader("📆 Weekly Electricity Consumption Prediction")
    model_weekly = joblib.load("electricity_weekly_model.pkl")

    year = st.number_input("Enter Year", min_value=2010, max_value=2030, value=2024)
    week = st.number_input("Enter Week Number (1-52)", min_value=1, max_value=52, value=45)
    weekday = st.number_input("Enter Last Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=6)

    X_week = pd.DataFrame([[year, week, weekday]], columns=['Year', 'Week', 'Weekday'])

    if st.button("🔮 Predict Weekly Consumption"):
        prediction = model_weekly.predict(X_week)[0]
        st.success(f"Predicted Weekly Consumption: **{prediction:.2f} MW**")

        # 🧭 Graph: show weekly trend + predicted point
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(weekly['Datetime'], weekly['Weekly_Consumption'], label='Historical Data', color='green')
        # Find approximate date of that week
        predicted_date = datetime.strptime(f'{year}-W{int(week)}-1', "%Y-W%W-%w")
        ax.scatter(predicted_date, prediction, color='red', label='Predicted', s=100)
        ax.set_xlabel("Week")
        ax.set_ylabel("Electricity Consumption (MW)")
        ax.set_title("Weekly Electricity Consumption Trend")
        ax.legend()
        st.pyplot(fig)
