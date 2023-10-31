# Import necessary libraries
import pandas as pd  # Import the Pandas library for data manipulation
import streamlit as st  # Import Streamlit for creating a web application
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for creating plots
from statsmodels.tsa.seasonal import seasonal_decompose  # Import seasonal decomposition function from statsmodels
from statsmodels.tsa.api import VAR  # Import Vector Autoregression (VAR) model from statsmodels
from statsmodels.tsa.base.datetools import dates_from_str  # Import date handling functions from statsmodels
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder for encoding categorical data
from statsmodels.tsa.stattools import adfuller  # Import Augmented Dickey-Fuller test for time series stationarity check

# Load and preprocess the dataset
# Load the dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('train.csv')  # Load the dataset from a CSV file

# Select the first 1000 rows of the DataFrame
data = data.head(1000)  # Take the first 1000 rows of the dataset

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])  # Convert the 'Date' column to datetime objects

# Handle missing values
data['Weekly_Sales'].fillna(data['Weekly_Sales'].mean(), inplace=True)  # Fill missing 'Weekly_Sales' values with the mean

# Check for infinite values and handle them if needed
if np.isinf(data['Weekly_Sales']).any():  # Check if there are infinite values in 'Weekly_Sales'
    data['Weekly_Sales'] = data['Weekly_Sales'].replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
    data['Weekly_Sales'].fillna(data['Weekly_Sales'].mean(), inplace=True)  # Fill NaN values with the mean

# Analyze the time series data
# Calculate ADF test p-value for 'Weekly_Sales'
result = adfuller(data['Weekly_Sales'])  # Perform the Augmented Dickey-Fuller test
adf_statistic = result[0]  # Extract the ADF statistic
p_value = result[1]  # Extract the p-value

# Decompose the time series
decomposition = seasonal_decompose(data['Weekly_Sales'], model='additive', period=1)  # Decompose the time series into trend, seasonal, and residual components
trend = decomposition.trend  # Extract the trend component
seasonal = decomposition.seasonal  # Extract the seasonal component
residual = decomposition.resid  # Extract the residual component

# Preprocess the data for the VAR model
data_preprocessed = data.dropna()  # Remove rows with missing values

# Select relevant columns (features for forecasting)
features = ["Date", 'Weekly_Sales', 'IsHoliday']  # Define the columns to use as features

# Keep only the selected features
data2 = data_preprocessed[features]  # Create a new DataFrame with the selected features

# Convert the 'Date' column to datetime objects
data2['Date'] = pd.to_datetime(data2['Date'])  # Convert the 'Date' column to datetime objects

# Set 'Date' as the index
data2.set_index('Date', inplace=True)  # Set the 'Date' column as the index of the DataFrame

# Encode the 'IsHoliday' column
label_encoder = LabelEncoder()  # Initialize a LabelEncoder
data2['IsHoliday'] = label_encoder.fit_transform(data2['IsHoliday'])  # Encode the 'IsHoliday' column as numeric values

# Build and run the VAR model
# Create and fit the VAR model
model = VAR(data2)  # Create a Vector Autoregression (VAR) model
results = model.fit()  # Fit the model to the data

# Forecast for the next 2 weeks (14 days)
forecast = results.forecast(y=data2.values, steps=14)  # Forecast the next 14 days

# Create a date range for the next 2 weeks starting from '2012-10-19'
next_two_weeks_dates = pd.date_range(start='2012-10-26', periods=14)  # Create a date range for the next 14 days

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame(forecast, columns=data2.columns, index=next_two_weeks_dates)  # Create a DataFrame for the forecast

# Calculate the date range for the last 5 weeks
last_five_weeks_start = data2['Date'].max() - pd.DateOffset(weeks=5)  # Calculate the start date of the last 5 weeks
last_five_weeks_end = data2['Date'].max()  # Calculate the end date of the last 5 weeks

# Filter the historical data for the last 5 weeks
historical_data = data_preprocessed[(data_preprocessed['Date'] >= last_five_weeks_start) & (data_preprocessed['Date'] <= last_five_weeks_end)]  # Filter the data for the last 5 weeks

# Plot the 'Weekly_Sales' column for the last 5 weeks and forecast for the next 2 weeks
plt.figure(figsize=(12, 6))  # Create a plot with a specified size
plt.plot(historical_data['Date'], historical_data['Weekly_Sales'], label='Historical Sales (Last 5 Weeks)', marker='o')  # Plot historical sales with markers
plt.plot(forecast_df.index, forecast_df['Weekly_Sales'], label='Forecasted Sales (Next 2 Weeks)', linestyle='--', marker='o')  # Plot forecasted sales with markers and dashed line
plt.xlabel('Date')  # Set the x-axis label
plt.ylabel('Sales')  # Set the y-axis label
plt.title('Sales Forecast for the Last 5 Weeks and Next 2 Weeks')  # Set the plot title
plt.legend()  # Add a legend to the plot
plt.grid(True)  # Display gridlines on the plot

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility

plt.show()  # Display the plot

# Streamlit web app
st.title('Multi-Variate Sales Forecasting Web App')  # Set the title of the Streamlit web app
st.write("Data Overview:")  # Display a section title
st.write(data.head())  # Display the first few rows of the dataset

st.write("ADF Statistic:", adf_statistic)  # Display the ADF statistic
st.write("p-value:", p_value)  # Display the ADF test p-value

st.write("Decomposed Components:")  # Display a section title
st.write("Trend:")  # Display a subsection title
st.line_chart(trend)  # Display the trend component as a line chart
st.write("Seasonal:")  # Display a subsection title
st.line_chart(seasonal)  # Display the seasonal component as a line chart
st.write("Residual:")  # Display a subsection title
st.line_chart(residual)  # Display the residual component as a line chart

st.write("Last 5 Weeks of Historical Data:")  # Display a section title
st.line_chart(data.set_index('Date').tail(5))  # Display the last 5 rows of historical data as a line chart

st.write("Forecast for the Next 2 Weeks:")  # Display a section title
st.line_chart(forecast_df)  # Display the forecast for the next 2 weeks as a line chart

# You can add more content or interactivity as needed

# To run the app, use the following command in your terminal:
# streamlit run sales_forecast_app.py
