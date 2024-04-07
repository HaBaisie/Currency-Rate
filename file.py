import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
st.title("US Dollars Rate Prediction Over Time")
# Load the pickled model
with open('exchange_rate_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset
df = pd.read_csv('exchange24032024.csv')

# Data Preprocessing
df['Rate Date'] = pd.to_datetime(df['Rate Date'])
usd_df = df[df['Currency'] == 'US DOLLAR'][['Rate Date', 'Buying Rate', 'Central Rate', 'Selling Rate']]

# Normalization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(usd_df[['Buying Rate', 'Central Rate', 'Selling Rate']])

# User input for future date
future_date_str = st.text_input("Enter a future date (YYYY-MM-DD): ")

# Predict button
if st.button("Predict"):
    if future_date_str:
        future_date = pd.to_datetime(future_date_str)

        # Prepare input data for prediction
        historical_data = usd_df.copy()  # Use all historical data
        scaled_input_data = scaler.transform(historical_data[['Buying Rate', 'Central Rate', 'Selling Rate']])

        # Extend time range for prediction (e.g., 30 days into the future)
        future_dates = pd.date_range(start=historical_data['Rate Date'].iloc[-1], periods=30, freq='D')[1:]  # Exclude the last date
        future_scaled_input_data = np.vstack([scaled_input_data, np.zeros((len(future_dates), scaled_input_data.shape[1]))])

        # Predict future exchange rates
        future_exchange_rates = np.zeros((len(future_dates), 3))
        for i, future_date in enumerate(future_dates):
            future_exchange_rates_scaled = model.predict(future_scaled_input_data[i].reshape(1, -1))
            future_exchange_rates[i] = scaler.inverse_transform(future_exchange_rates_scaled)

        # Display predicted future exchange rates
        st.write("Predicted future exchange rates:")
        for i, date in enumerate(future_dates):
            st.write("Date:", date)
            st.write("Buying Rate:", future_exchange_rates[i, 0])
            st.write("Central Rate:", future_exchange_rates[i, 1])
            st.write("Selling Rate:", future_exchange_rates[i, 2])
