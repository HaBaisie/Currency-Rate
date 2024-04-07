import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

st.title("US Dollars Rate Prediction Over Time")

# Load the pickled model
#with open('exchange_rate_model.pkl', 'rb') as f:
    #model = pickle.load(f)

# Load the dataset
df = pd.read_csv('exchange24032024.csv')

# Data Preprocessing
df['Rate Date'] = pd.to_datetime(df['Rate Date'])
usd_df = df[df['Currency'] == 'US DOLLAR'][['Rate Date', 'Buying Rate', 'Central Rate', 'Selling Rate']]
usd_df.set_index('Rate Date', inplace=True)

# User input for future date
future_date_str = st.text_input("Enter a future date (YYYY-MM-DD): ")

# Predict button
if st.button("Predict"):
    if future_date_str:
        future_date = pd.to_datetime(future_date_str)

        # Prepare input data for prediction
        historical_data = usd_df['Buying Rate']  # Using 'Buying Rate' as an example
        train_data = historical_data[:future_date]

        # Fit ARIMA model
        model = ARIMA(train_data, order=(5,1,0))  # Example order, you might need to tune this
        fitted_model = model.fit()

        # Predict future exchange rates
        forecast = fitted_model.forecast(steps=1)[0]

        # Display predicted future exchange rate
        st.write("Predicted future exchange rate for", future_date)
        st.write("Buying Rate:", forecast)
