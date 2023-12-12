from datetime import date
import yfinance as yf
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import webbrowser
from sklearn import preprocessing
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import seaborn as sns
import streamlit as st
import joblib

# fetch stock data of AAPL from ticker Apple corporation
st.header('Stock price and Trend prediction')

start_date = "2020-01-05"
end_date = "2023-08-14"

stock_input=st.text_input('Enter stock ticker', 'AAPL')
ticker_symbol=stock_input.upper()

# download the AAPL stock data from yahoofinance to sd dataframe
sd = yf.download(ticker_symbol, start=start_date, end=end_date)

df = pd.DataFrame(sd)
df['date'] = pd.to_datetime(df.index)

st.subheader(f'data for {ticker_symbol} from 2010-2023')
st.write(sd.describe())

loaded_model = joblib.load('my_best_rf_model.pkl')


# visualize fluctuation
fig = go.Figure(data=[go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close']
                                     )])

fig.update_layout(
    title='Stock price chart of AAPL',
    yaxis_title='price ($)',
    xaxis_title='date',
    xaxis=dict(
        rangeslider=dict(
            visible=True
        )
    )
)

# fig.write_html("candlestick_chart.html")

# Open the HTML file in the default web browser
# webbrowser.open("candlestick_chart.html")
# plt.show()

# drop irrelavant columns
df2 = df.copy()

df2.plot.line(y="Close", use_index=True)

st.subheader('closing price vs time chart with 100MA')
ma100=df2.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

# model training
X = df2[['Open', 'High', 'Low', 'Close', 'Adj Close']]
y = df2['Close']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# parameter
param_g = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# model build
rf = RandomForestRegressor(n_estimators=100, random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_g, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

# evaluate the model on training data
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# evaluate the model on testing data
y_test_pred_best = best_rf.predict(X_test)
mae_best = mean_absolute_error(y_test, y_test_pred_best)
mse_best = mean_squared_error(y_test, y_test_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_test_pred_best)

print("Best Parameters:", best_params)
print("\nTesting Mean Absolute Error (Best): {:.2f}".format(mae_best))
print("Testing Mean Squared Error (Best): {:.2f}".format(mse_best))
print("Testing Root Mean Squared Error (Best): {:.2f}".format(rmse_best))
print("Testing R-squared (Best): {:.2f}".format(r2_best))

print("\nComparison of Train and Test Performance:")
print("Train MAE - Test MAE:", train_mae - mae_best)
print("Train MSE - Test MSE:", train_mse - mse_best)
print("Train R-squared - Test R-squared:", train_r2 - r2_best)

# visualize predicted values vs. actual values for testing data
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='crimson')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Prices (Testing Data)")
plt.show()

# put sample values to make predictions
new_sd = np.array([[173.850006, 174.589996, 172.169998, 173.750000, 173.750000]])

predicted_price = best_rf.predict(new_sd)
print("Predicted stock price (Best):", predicted_price[0])


#test model
loaded_model = joblib.load('my_best_rf_model.pkl')
test_d=np.array([[174.9560, 175.30, 173.908, 176, 176]])
test_predictions = loaded_model.predict(test_d)
print("Test Predictions:", test_predictions)


actual_price = test_d[0][-1]
predicted_price = test_predictions[0]
difference = actual_price - predicted_price

print("Actual Price:", actual_price)
print("Predicted Price:", predicted_price)
print("Difference:", difference)


st.sidebar.header("Enter Stock Data")
open_price = st.sidebar.number_input("Open Price", value=0.0)
high_price = st.sidebar.number_input("High Price", value=0.0)
low_price = st.sidebar.number_input("Low Price", value=0.0)
close_price = st.sidebar.number_input("Close Price", value=0.0)
adj_close = st.sidebar.number_input("Adj Close Price", value=0.0)

# Predict button
if st.sidebar.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([[open_price, high_price, low_price, close_price, adj_close]])

    # Use the loaded model to make predictions
    predicted_price = rf.predict(input_data)[0]

    # Display the prediction result
    st.write(f"Predicted Stock Price: {predicted_price:.2f}")

