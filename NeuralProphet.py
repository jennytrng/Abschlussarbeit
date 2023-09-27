import pandas as pd
from neuralprophet import NeuralProphet, set_log_level
from sklearn.metrics import mean_squared_error
import numpy as np

set_log_level("ERROR")

# Load and preprocess the dataset (assuming it's a CSV file)
data = pd.read_csv('AAPL.csv')
data = data.rename(columns={"Date": "ds", "Close": "y"})
data = data.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1)

# Initialize a NeuralProphet model with specified configuration
np_model = NeuralProphet(
    growth="linear",
    changepoints_range=0.95,
    trend_reg=0,
    trend_reg_threshold=False,
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    seasonality_mode="additive",
    seasonality_reg=0,
    n_lags=7,
    batch_size=32,
    n_changepoints=2,
    learning_rate=None,
    epochs=100,
    loss_func="Huber",
    normalize="auto",
    impute_missing=True
)

# Split the dataset into training and testing sets (80% training, 20% testing)
data_train, data_test = np_model.split_df(data, valid_p=0.2)

# Fit the NeuralProphet model to the training data
metrics = np_model.fit(data, freq="D", epochs=1000)

# Forecasting
future = np_model.make_future_dataframe(data, periods=365,
                                        n_historic_predictions=True)  # Use the trained model to make predictions for the future
forecast = np_model.predict(future)

# Plots the model predictions
fig = np_model.plot(forecast, ylabel="Price")
fig.show()
actual_values = data_test['y'].values  # Extract the overlapping period from the original data
predicted_values = forecast['yhat1'].tail(
    len(data_test)).values  # Extract the predicted values from the forecast for the same overlapping period

# Calculate MSE
mse = mean_squared_error(actual_values, predicted_values)
print(f'MSE: {mse}')


# Calculate SMAPE
def calculate_smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


smape_res = calculate_smape(actual_values, predicted_values)

# Print the calculated SMAPE
print(f'SMAPE: {smape_res:.2f}%')
