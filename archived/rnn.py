import pandas as pd
data_raw = pd.read_excel("AAPL.xlsx")
data_raw = data_raw.iloc[2:, :] # remove "NOTES" and "Average Check for Errors"
data_raw = data_raw.reset_index(drop = True)

# Select columns to use as features
data = data_raw[["Date", "Wiki Traffic- 1 Day Lag", "Wiki 5day disparity", "Wiki MA3 Move", "Wiki MA5 Move", "Wiki EMA5 Move", "Goog ROC", "Goog MA3", "Goog MA5", "Goog EMA5", "Goog 3day Disparity", "Goog RSI (14 days)", "Price RSI (14 days)", "Target"]]

# Remove all NaN rows
data = data.dropna(axis = 0)
data = data.reset_index(drop = True)

# Split dataset
X = data.iloc[:, 1:-1]
Y = data.iloc[:, -1]

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size = 0.2)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Reshape X data
import numpy as np
X_train_3D = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_3D = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
model = Sequential() # initialize the RNN
model.add(LSTM(units = 20, return_sequences = True, input_shape = (X_train.shape[1], 1))) # input shape: (number of features, 1)
model.add(LSTM(units = 20, return_sequences = True))
model.add(LSTM(units = 20, return_sequences = True))
model.add(LSTM(units = 20))
model.add(Dense(units = 1))
model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])

# Train the neural network
model.fit(X_train_3D, Y_train, epochs = 50, batch_size = 5, shuffle = False)

# Make predictions
Y_test_pred_prob = model.predict(X_test_3D)
Y_train_pred_prob = model.predict(X_train_3D)

Y_test_pred = (Y_test_pred_prob >= 0.5).astype(int)
Y_train_pred = (Y_train_pred_prob >= 0.5).astype(int)

# Performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
metrics = pd.DataFrame()
metrics["Metric"] = ["Accuracy", "Precision", "Recall", "F1"]
metrics["Train"] = [accuracy_score(Y_train, Y_train_pred),
                   precision_score(Y_train, Y_train_pred),
                   recall_score(Y_train, Y_train_pred),
                   f1_score(Y_train, Y_train_pred)]
metrics["Test"] = [accuracy_score(Y_test, Y_test_pred),
                   precision_score(Y_test, Y_test_pred),
                   recall_score(Y_test, Y_test_pred),
                   f1_score(Y_test, Y_test_pred)]
metrics = metrics.set_index("Metric")
print(metrics)
print("")

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_test_pred))