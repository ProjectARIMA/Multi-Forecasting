# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.stattools import adfuller

# Define the Streamlit app
st.title("Multivariate Anomaly Detection and Time Series Forecasting Web Application") #It sets the website title to 'Anomaly Detection and Time Series Forecasting Web Application'

st.set_option('deprecation.showPyplotGlobalUse', False) #It hides all the warnings in python output

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"]) # A widget to let users input his/her CSV files.
if uploaded_file is not None:
    # Load and preprocess the dataset
    # Load the dataset (replace 'your_dataset.csv' with the actual file path)
    data = pd.read_csv(uploaded_file)
    
    # Select the first 1000 rows of the DataFrame
    # Only the first 1000 rows are used for data analysis in this case because there are 4+ lakh rows in the dataset and the program was crashing by handling such a large amount of dataset.
    data = data.head(1000)
    
    # Convert the 'Date' column to datetime to perform time-based analysis.
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Handle missing values
    # Replaces missing values in 'Weekly_Sales' column with the mean of 'Weekly_Sales' column values.
    # Note: NaN represents missing value.
    data['Weekly_Sales'].fillna(data['Weekly_Sales'].mean(), inplace=True)
    
    # Check for infinite values and handle them if needed
    # Replaces infinite values with NaN values and then replace those NaN with the mean of 'Weekly_Sales' column values.
    if np.isinf(data['Weekly_Sales']).any():
        data['Weekly_Sales'] = data['Weekly_Sales'].replace([np.inf, -np.inf], np.nan)
        data['Weekly_Sales'].fillna(data['Weekly_Sales'].mean(), inplace=True)
    
    # Analyze the time series data
    # Calculate ADF test p-value for 'Weekly_Sales'
    # Checks stationarity of 'Weekly_Sales' column data and resultant array is associated with 'result' variable.
    result = adfuller(data['Weekly_Sales'])
    
    # 'result[0]' yeilds statistic value.
    adf_statistic = result[0]
    
    # 'result[1]' yeilds p-value.
    p_value = result[1]
    
    # Decompose the time series
    #decomposition = seasonal_decompose(data['Weekly_Sales'], model='additive', period=1)
    decomposition = seasonal_decompose(data['Weekly_Sales'], period = 7)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Preprocess the data for the VAR model
    # Creates new DataFrame object and stores rows of data excluding rows containing NaN(Not-a-Number) values.
    data_preprocessed = data.dropna()
    
    # Select relevant columns (features for forecasting)
    features = ["Date", 'Weekly_Sales', 'IsHoliday']
    
    # Keep only the selected features
    data2 = data_preprocessed[features]
    
    # Convert the 'Date' column to datetime objects
    data2['Date'] = pd.to_datetime(data2['Date'])
    
    # Set 'Date' as the index
    data2.set_index('Date', inplace=True)
    
    # Encode the 'IsHoliday' column
    # LabelEncoder object is used to convert True/False to 1/0 values in this case.
    label_encoder = LabelEncoder()
    
    # Maps True to 1 and False to 0 in data2 object's 'isHoliday' column.
    data2['IsHoliday'] = label_encoder.fit_transform(data2['IsHoliday'])
    
    # Build and run the VAR model
    # Create and fit the VAR model
    model = VAR(data2)
    results = model.fit()
    
    # Forecast for the next 2 weeks (14 days)
    forecast = results.forecast(y=data2.values, steps=14)
    
    # Create a date range for the next 2 weeks starting from '2012-10-26'
    next_two_weeks_dates = pd.date_range(start='2012-10-26', periods=14)
    
    # Create a DataFrame for the forecast
    # The forecasted values are organized into a DataFrame named forecast_df. The columns of this DataFrame correspond to the columns in the data2 DataFrame, and the index is set to the date range created in the previous step.
    forecast_df = pd.DataFrame(forecast, columns=data2.columns, index=next_two_weeks_dates)
    
    # Calculate the date range for the last 5 weeks
    # The start date for the last 5 weeks of historical data is calculated by taking the maximum date from the 'Date' column in the original 'data' DataFrame and subtracting 5 weeks.
    last_five_weeks_start = data['Date'].max() - pd.DateOffset(weeks=5)

    #The end date for the last 5 weeks of historical data is set to the maximum date in the 'Date' column of the original 'data' DataFrame.
    last_five_weeks_end = data['Date'].max()
    
    # Filter the historical data for the last 5 weeks
    historical_data = data_preprocessed[(data_preprocessed['Date'] >= last_five_weeks_start) & (data_preprocessed['Date'] <= last_five_weeks_end)]
    
    # Plot the 'Weekly_Sales' column for the last 5 weeks and forecast for the next 2 weeks
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data['Date'], historical_data['Weekly_Sales'], label='Historical Sales (Last 5 Weeks)', marker='o')
    plt.plot(forecast_df.index, forecast_df['Weekly_Sales'], label='Forecasted Sales (Next 2 Weeks)', linestyle='--', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales Forecast for the Last 5 Weeks and Next 2 Weeks')
    plt.legend()
    plt.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.show()
    
    # Streamlit web app
    st.title('Multi Variate Forecasting Web app')
    st.write("Data Overview:")
    st.write(data.head())
    
    st.write("ADF Statistic:", adf_statistic)
    st.write("p-value:", p_value)
    
    st.write("Decomposed Components:")
    st.write("Trend:")
    st.line_chart(trend)
    st.write("Seasonal:")
    st.line_chart(seasonal)
    st.write("Residual:")
    st.line_chart(residual)
    
    st.write("Last 5 Weeks of Historical Data:")

    # Displays line chart with 'Date' column as index and last 35 (5 weeks) data points.
    st.line_chart(data.set_index('Date').tail(35))
    
    st.write("Forecast for the Next 2 Weeks:")

    # Display line chart for forcasted values using 'forcast_df' DataFrame object.
    st.line_chart(forecast_df)
    
    # You can add more content or interactivity as needed
    
    # To run the app, use the following command in your terminal:
    # streamlit run sales_forecast_app.py
