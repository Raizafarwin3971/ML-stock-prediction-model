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
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import seaborn as sns
import joblib

# fetch stock data of AAPL from ticker Apple corporation

ticker = "AAPL"
start_date = "2020-01-05"
end_date = "2023-08-14"

# download the AAPL stock data from yahoofinance to sd dataframe
sd = yf.download(ticker, start=start_date, end=end_date)

df = pd.DataFrame(sd)
df['date'] = pd.to_datetime(df.index)

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

corr = df.corr()
# print(corr)

correlation_matrix = df.corr()
plt.figure(figsize=(10, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
# plt.show()

# drop irrelavant columns
df2 = df.copy()
df2 = df.drop(['date', 'Volume'], axis='columns')
print(df2.shape)

df2.reset_index(drop=True, inplace=True)

df2.plot.line(y="Close", use_index=True)

ma100=df2.Close.rolling(100).mean()
print(ma100)

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
# dropping index column
df2.reset_index(drop=True, inplace=True)

# checking null values
print(df2.isnull().sum())

# model training
X = df2[['Open', 'High', 'Low', 'Close', 'Adj Close']]
y = df2['Close']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# parameter
#random f
param_g = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
#linear
# model build
rf = RandomForestRegressor(n_estimators=100, random_state=42)
lreg = LinearRegression()

#gridsearch for random for
grid_search = GridSearchCV(estimator=rf, param_grid=param_g, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


#fit linear reg
lreg.fit(X_train, y_train)

best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_
best_lreg = lreg
#linear gridsearch

voting_regressor = VotingRegressor(estimators=[('rf', best_rf), ('lr', best_lreg)])
voting_regressor.fit(X_train, y_train)


y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

y_pred = grid_search.predict(X_test)
# evaluate the rf model on train data
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

mse = mean_squared_error(y_test, y_pred)

# evaluate the rf model on test data
y_test_pred_best = best_rf.predict(X_test)
mae_best = mean_absolute_error(y_test, y_test_pred_best)
mse_best = mean_squared_error(y_test, y_test_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_test_pred_best)

#evaluate the lr model on test data
y_pred = best_lreg.predict(X_test)
r2_lreg = r2_score(y_test, y_pred)


#voting regressor-evaluating on testing data
y_test_pred_voting = voting_regressor.predict(X_test)
mae_voting = mean_absolute_error(y_test, y_test_pred_voting)
mse_voting = mean_squared_error(y_test, y_test_pred_voting)
rmse_voting = np.sqrt(mse_voting)
r2_voting = r2_score(y_test, y_test_pred_voting)
print("\nVoting Regressor:")
print("Testing Mean Absolute Error: {:.2f}".format(mae_voting))
print("Testing Mean Squared Error: {:.2f}".format(mse_voting))
print("Testing Root Mean Squared Error: {:.2f}".format(rmse_voting))
print("Testing R-squared: {:.2f}".format(r2_voting))

print("Best Parameters:", best_params)
print("\nTesting Mean Absolute Error (Best): {:.2f}".format(mae_best))
print("Testing Mean Squared Error (Best): {:.2f}".format(mse_best))
print("Testing Root Mean Squared Error (Best): {:.2f}".format(rmse_best))
print("Testing R-squared (Best): {:.2f}".format(r2_best))

print("\nComparison of Train and Test Performance:")
print("Train MAE - Test MAE:", train_mae - mae_best)
print("Train MSE - Test MSE:", train_mse - mse_best)
print("Train R-squared - Test R-squared:", train_r2 - r2_best)

print("\nModel Comparison:")
print("Random Forest - Testing R-squared:", r2_best)
print("Linear Regression - Testing R-squared:", r2_score(y_test, best_lreg.predict(X_test)))
print("Voting Regressor - Testing R-squared:", r2_voting)

# visualize rf predicted values vs. actual values for testing data
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='crimson')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Prices (Testing Data)")
plt.show()

# Visualize Lr predicted values vs. actual values for testing data
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='crimson')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price (Linear Regression)")
plt.title("Actual vs. Predicted Prices (Testing Data) - Linear Regression")
plt.show()

# Visualize Voting Regressor predicted values vs. actual values for testing data
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_test_pred_voting, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='crimson')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price (Voting Regressor)")
plt.title("Actual vs. Predicted Prices (Testing Data) - Voting Regressor")
plt.show()

# saving model for future training for other companies
model_filename = 'my_best_rf_model.pkl'
joblib.dump(best_rf, model_filename)


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
