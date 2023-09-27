import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.dates as mdates

data = pd.read_csv('TSLA.csv')  # Read data from the CSV file ('TSLA.csv', 'GOOGL.csv', 'AAPL.csv')
data = data[['Date', 'Close']]


# Converts a 'YYYY-MM-DD' date string to a datetime object.
def datestr_to_datetime(date):
    date_parts = date.split('-')
    year, month, day = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
    return datetime.datetime(year=year, month=month, day=day)  # Create a datetime object using the extracted components


data['Date'] = data['Date'].apply(datestr_to_datetime)
data.index = data.pop('Date')

# Visualization for 'Closing Prices Over Time'
plt.plot(data.index, data['Close'])
plt.xlabel('Index')
plt.ylabel('Close')
plt.title('Closing Prices Over Time')
plt.ylim(0, 430)
plt.show()


# A function to generate a windowed DataFrame from a given DataFrame, using a specified date range and window size.
def generate_windowed_dataframe(data, fst_date_str, lt_date_str, n=3):
    # Convert input date strings to datetime objects
    first_date = datestr_to_datetime(fst_date_str)
    last_date = datestr_to_datetime(lt_date_str)

    # Initialize variables for tracking window data
    target_date = first_date
    date_records = []
    X, Y = [], []

    last_iteration = False
    while True:
        # Create a subset DataFrame up to the target date
        data_subset = data.loc[:target_date].tail(n + 1)

        # Check if the subset size is valid for the window size
        if len(data_subset) != n + 1:
            print(f'Error: Window size is too big.')
            return

        # Extract feature (X) values and target (Y) value
        values = data_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        # Store data in respective lists
        date_records.append(target_date)
        X.append(x)
        Y.append(y)

        # Calculate the date for the next window
        following_week = data.loc[target_date:target_date + datetime.timedelta(days=7)]
        following_dt_str = str(following_week.head(2).tail(1).index.values[0])
        following_dt_str = following_dt_str.split('T')[0]
        year_month_day = following_dt_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        # Check if it's the last iteration
        if last_iteration:
            break

        target_date = next_date  # Update target date

        # Check if the target date matches the last date
        if target_date == last_date:
            last_iteration = True

    # Create a new DataFrame to store windowed data
    new_data = pd.DataFrame({})
    new_data['Target Date'] = date_records

    # Populate the windowed DataFrame with features (X) and target (Y) columns
    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        new_data[f'Target-{n - i}'] = X[:, i]

    new_data['Target'] = Y

    return new_data


# Generate a windowed DataFrame
wd_data = generate_windowed_dataframe(data,
                                      '2022-05-31',
                                      '2023-05-31',
                                      n=3)


# A function to convert a windowed DataFrame to input features (X) and target values (y) suitable for training a model.
def convert_to_model_input(wd_dataframe):
    data_as_np = wd_dataframe.to_numpy()  # Convert the windowed DataFrame to a NumPy array
    date_column = data_as_np[:, 0]  # Extract dates from the first column of the array
    feature_matrix = data_as_np[:, 1:-1]  # Extract input feature matrix (excluding the first and last columns)

    # Reshape the input feature matrix to be compatible with model input
    X = feature_matrix.reshape((len(date_column), feature_matrix.shape[1], 1))

    Y = data_as_np[:, -1]  # Extract the target values from the last column of the array

    return date_column, X.astype(np.float32), Y.astype(
        np.float32)  # Convert data types to np.float32 for compatibility with deep learning frameworks


dates, X, y = convert_to_model_input(
    wd_data)  # Convert the windowed DataFrame to input features (X) and target values (y)

# Splitting the data into training, validation, and test sets based on proportions of the dataset.
q_80 = int(len(dates) * .8)  # 80% of the data for training
q_90 = int(len(dates) * .9)  # 10% of the data for validation and 10% for testing

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[
                                                        :q_80]  # Splitting the data and corresponding input features and target values into training set
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[
                                                          q_80:q_90]  # Splitting the data and corresponding input features and target values into validation set
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[
                                                     q_90:]  # Splitting the data and corresponding input features and target values into test set

plt.plot(dates_train, y_train)  # Plotting training set
plt.plot(dates_val, y_val)  # Plotting validation set
plt.plot(dates_test, y_test)  # Plotting test set

# Adding legend to differentiate between the sets on the plot
plt.legend(['Train', 'Validation', 'Test'])
plt.show()

# Visualizing the target values for each split using line plots
plt.plot(dates_train, y_train, label='Train', color='blue')  # Plotting training set
plt.plot(dates_val, y_val, label='Validation', color='orange')  # Plotting validation set
plt.plot(dates_test, y_test, label='Test', color='green')  # Plotting test set

# Adding legend to differentiate between the sets on the plot
plt.legend()
plt.xlabel('Date')
plt.ylabel('Target Value')
plt.title('Target Value Over Time')
plt.ylim([0, 400])
plt.show()

# Define the model architecture with specific layers
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input((3, 1)),
    # Input layer with a shape of (3, 1) representing sequences of length 3 and 1 feature
    tf.keras.layers.LSTM(64),  # LSTM layer with 64 units (neurons) for sequence processing
    tf.keras.layers.Dense(32, activation='relu'),  # Fully connected layer with 32 units and ReLU activation
    tf.keras.layers.Dense(32, activation='relu'),  # Another fully connected layer with 32 units and ReLU activation
    tf.keras.layers.Dense(1)  # Output layer with 1 unit for regression prediction
])

# Compile the model with a MSE function and MAE as a metric for evaluation during training
lstm_model.compile(loss='mse',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Train the model on the training data
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000)

# Visualization for training set
train_predictions = lstm_model.predict(X_train).flatten()
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.legend(['Training Predictions', 'Training Observations'])
plt.ylim([0, 400])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.show()

# Visualization for validation set
val_predictions = lstm_model.predict(X_val).flatten()
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.legend(['Validation Predictions', 'Validation Observations'])
plt.ylim([0, 220])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.show()

# Visualization for test set
test_predictions = lstm_model.predict(X_test).flatten()
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Testing Predictions', 'Testing Observations'])
plt.ylim([0, 400])
plt.xticks(rotation=45)
plt.show()

# Visualization for all sets
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Training Predictions',
            'Training Observations',
            'Validation Predictions',
            'Validation Observations',
            'Testing Predictions',
            'Testing Observations'])
plt.ylim([0, 350])
plt.show()

val_pred = lstm_model.predict(X_val).flatten()  # Make predictions on the validation set
mse = mean_squared_error(y_val, val_pred)  # Calculate Mean Squared Error (MSE)
mae = mean_absolute_error(y_val, val_pred)  # Calculate Mean Absolute Error (MAE)
rmse = sqrt(mse)  # Calculate Root Mean Squared Error (RMSE)


# Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
def calculate_smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


smape_res = calculate_smape(y_val, val_pred)

# Print the calculated metrics
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'SMAPE: {smape_res:.2f}%')
