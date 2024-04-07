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
        print (future_date_str)
        future_date = pd.to_datetime(future_date_str)

        # Prepare input data for prediction
        historical_data = usd_df[usd_df['Rate Date'] <= future_date]
        scaled_input_data = scaler.transform(historical_data[['Buying Rate', 'Central Rate', 'Selling Rate']])

        # Predict future exchange rates
        future_exchange_rates_scaled = model.predict(scaled_input_data[-1].reshape(1, -1))

        # Inverse transform to get the actual predicted values
        future_exchange_rates = scaler.inverse_transform(future_exchange_rates_scaled)

        # Display predicted future exchange rates
        st.write("Predicted future exchange rates:")
        st.write("Buying Rate:", future_exchange_rates[0, 0])
        st.write("Central Rate:", future_exchange_rates[0, 1])
        st.write("Selling Rate:", future_exchange_rates[0, 2])
