import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('train.csv')

# Select the first 500 rows of the DataFrame
data = data.head(1000)

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Handle missing values (e.g., fill with the mean value)
data['Weekly_Sales'].fillna(data['Weekly_Sales'].mean(), inplace=True)

# Check for infinite values and handle them if needed
if np.isinf(data['Weekly_Sales']).any():
    data['Weekly_Sales'] = data['Weekly_Sales'].replace([np.inf, -np.inf], np.nan)
    data['Weekly_Sales'].fillna(data['Weekly_Sales'].mean(), inplace=True)

# Calculate ADF test p-value for 'Weekly_Sales'
result = adfuller(data['Weekly_Sales'])
adf_statistic = result[0]
p_value = result[1]

# Decompose the time series
decomposition = seasonal_decompose(data['Weekly_Sales'], model='additive', period=1)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Drop rows with missing values
data = data.dropna()

# Select relevant columns (features for forecasting)
features = ["Date", 'Weekly_Sales', 'IsHoliday']

# Keep only the selected features
data2 = data[features]

# Convert the 'Date' column to datetime objects
data2['Date'] = pd.to_datetime(data2['Date'])

# Set 'Date' as the index
data2.set_index('Date', inplace=True)

# Encode the 'IsHoliday' column
label_encoder = LabelEncoder()
data2['IsHoliday'] = label_encoder.fit_transform(data2['IsHoliday'])

# Create and fit the VAR model
model = VAR(data2)
results = model.fit()

# Forecast for the next 2 weeks (14 days)
forecast = results.forecast(y=data2.values, steps=14)

# Create a date range for the next 2 weeks starting from '2012-10-19'
next_two_weeks_dates = pd.date_range(start='2012-10-26', periods=14)

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame(forecast, columns=data2.columns, index=next_two_weeks_dates)

# Calculate the date range for the last 5 weeks
last_five_weeks_start = data['Date'].max() - pd.DateOffset(weeks=5)
last_five_weeks_end = data['Date'].max()

# Filter the historical data for the last 5 weeks
historical_data = data[(data['Date'] >= last_five_weeks_start) & (data['Date'] <= last_five_weeks_end)]

# Streamlit app
st.title("Time Series Forecasting App")

# Display ADF test results
st.write("ADF Statistic:", adf_statistic)
st.write("p-value:", p_value)

# Plot the decomposed components
st.write("Decomposed Components:")
st.subheader("Original")
st.line_chart(data['Weekly_Sales'])
st.subheader("Trend")
st.line_chart(trend)
st.subheader("Seasonal")
st.line_chart(seasonal)
st.subheader("Residual")
st.line_chart(residual)

# Plot historical and forecasted sales
st.subheader("Sales Forecast for the Last 5 Weeks and Next 2 Weeks")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(historical_data['Date'], historical_data['Weekly_Sales'], label='Historical Sales (Last 5 Weeks)', marker='o')
ax.plot(forecast_df.index, forecast_df['Weekly_Sales'], label='Forecasted Sales (Next 2 Weeks)', linestyle='--', marker='o')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_title('Sales Forecast for the Last 5 Weeks and Next 2 Weeks')
ax.legend()
ax.grid(True)
st.pyplot(fig)
