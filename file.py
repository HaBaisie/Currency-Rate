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

        # Find index of the input future date
        future_date_index = historical_data['Rate Date'].searchsorted(future_date)

        # Predict exchange rates for the input future date
        future_exchange_rates_scaled = model.predict(scaled_input_data[future_date_index].reshape(1, -1))
        future_exchange_rates = scaler.inverse_transform(future_exchange_rates_scaled)

        # Display predicted exchange rates for the input future date
        st.write("Predicted exchange rates for", future_date)
        st.write("Buying Rate:", future_exchange_rates[0, 0])
        st.write("Central Rate:", future_exchange_rates[0, 1])
        st.write("Selling Rate:", future_exchange_rates[0, 2])
