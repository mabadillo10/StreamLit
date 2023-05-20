import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Function to preprocess the data for LSTM and SARIMA models
def preprocess_data(df):
    # Convert to hourly interval
    df_hourly = df.resample('1H').mean().interpolate()

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_hourly)

    # Prepare the data for LSTM model
    # User input for forecasting steps
    lookback = 2160
    X_lstm = []
    y_lstm = []
    for i in range(lookback, len(scaled_data)):
        X_lstm.append(scaled_data[i-lookback:i])
        y_lstm.append(scaled_data[i])

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    # Reshape X for LSTM input shape (samples, time steps, features)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

    # Prepare the data for SARIMA model
    lookback_sarima = 2160
    X_sarima = []
    y_sarima = []
    for i in range(lookback_sarima, len(scaled_data)):
        X_sarima.append(scaled_data[i-lookback_sarima:i])
        y_sarima.append(scaled_data[i])

    X_sarima = np.array(X_sarima)
    y_sarima = np.array(y_sarima)

    return X_lstm, y_lstm, scaler, X_sarima, y_sarima

# Function to build and train the LSTM model
def build_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=15, batch_size=32)
    return model

# Function to build and train the SARIMA model
def build_sarima_model(X, y):
    model = SARIMAX(y, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

# Function to forecast data using the LSTM model
def forecast_lstm_data(model, last_x, scaler, steps):
    future_data = []

    for _ in range(steps):
        prediction = model.predict(np.array([last_x]))
        future_data.append(prediction[0])
        last_x = np.concatenate((last_x[1:], prediction), axis=0)

    future_data = np.array(future_data)
    future_data = scaler.inverse_transform(future_data)
    return future_data

# Function to forecast data using the SARIMA model
def forecast_sarima_data(model, last_x, scaler, steps):
    future_data = model.forecast(steps)
    future_data = np.array(future_data)
    future_data = future_data.reshape(future_data.shape[0], 1)
    future_data = scaler.inverse_transform(future_data)
    return future_data

# Streamlit app
def main():
    st.title('Data Forecasting')

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['time_interval'] = pd.to_datetime(df['time_interval'])
        df.set_index('time_interval', inplace=True)

        X_lstm, y_lstm, scaler_lstm, X_sarima, y_sarima = preprocess_data(df)
        model_lstm = build_lstm_model(X_lstm, y_lstm)
        model_sarima = build_sarima_model(X_sarima, y_sarima)

        # Forecast data for 1 day
        last_x_lstm = X_lstm[-1]
        last_x_sarima = X_sarima[-1]
        forecast_steps = 24  # Number of steps to forecast

        future_data_lstm = forecast_lstm_data(model_lstm, last_x_lstm, scaler_lstm, forecast_steps)
        future_data_sarima = forecast_sarima_data(model_sarima, last_x_sarima, scaler_lstm, forecast_steps)

        forecast_timestamps = pd.date_range(start=df.index[-1], periods=len(future_data_lstm) + 1, freq='H')[1:]

        # Create DataFrame for forecasted data
        forecast_df_lstm = pd.DataFrame({'Delivery Interval': forecast_timestamps, 'Forecasted Value (LSTM)': future_data_lstm[:, 0]})
        forecast_df_sarima = pd.DataFrame({'Delivery Interval': forecast_timestamps, 'Forecasted Value (SARIMA)': future_data_sarima[:, 0]})
        forecast_df_lstm.set_index('Delivery Interval', inplace=True)
        forecast_df_sarima.set_index('Delivery Interval', inplace=True)
        merged_df = pd.merge(forecast_df_lstm, forecast_df_sarima, on='Delivery Interval')

        # Display forecasted data
        st.subheader('LSTM and SARIMAX Forecasted Data')
        st.write(merged_df)


        # Plot forecasted data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=future_data_lstm[:, 0], name='Forecasted Data (LSTM)'))
        fig.add_trace(go.Scatter(x=forecast_timestamps, y=future_data_sarima[:, 0], name='Forecasted Data (SARIMA)'))
        fig.update_layout(title='1-Day Forecast using LSTM and SARIMA', xaxis_title='Delivery Interval', yaxis_title='Average LMP')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
