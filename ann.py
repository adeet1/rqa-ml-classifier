# Set random seed
from numpy.random import seed
seed(1)

import pandas as pd

# Import raw data set
df0 = pd.read_excel("aapl.xlsx").iloc[2:, :].reset_index(drop = True)

# Remove the first 14 rows and the last row (we don't have future data in the present)
df0 = df0.iloc[14:-1, :].reset_index(drop = True)
df0["Wiki Move"] = df0["Wiki Move"].astype(int)
df0["Goog ROC"] = df0["Goog ROC"].astype(float)

# Select columns from data set
df = df0[["Open", "Close", "High", "Low", "RS", "Wiki Traffic- 1 Day Lag", "Wiki 5day disparity", "Wiki Move", "Wiki MA3 Move", "Wiki MA5 Move", "Wiki EMA5 Move", "Goog RS", "Goog MA3", "Goog MA5", "Goog EMA5 Move", "Goog 3day Disparity Move", "Goog ROC Move", "Goog RSI Move", "Wiki 3day Disparity", "Price RSI Move", "Google_Move", "Target"]]

# Split data set into independent and dependent variables
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

# Split data set into training/test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns = X_test.columns)

# Feature selection (remove highly correlated features)
from feature_selector import FeatureSelector
n = len(X_train.T)
fs = FeatureSelector(data = X_train)
fs.identify_collinear(correlation_threshold = 0.7) # select features from training set
corr = fs.ops['collinear']
X_train = fs.remove(methods = ['collinear']) # remove selected features from training set
to_remove = pd.unique(fs.record_collinear['drop_feature']) # features to remove
X_test = X_test.drop(columns = to_remove) # remove selected features from test set

# Create the artificial neural network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

num_input_nodes = len(X_train.T)
num_output_nodes = 1
num_hidden_nodes = int((num_input_nodes + num_output_nodes) / 2) # a typical value

# Add layers
classifier = Sequential()
classifier.add(Dense(output_dim = num_hidden_nodes, init = "uniform", activation = "sigmoid",
                     input_dim = num_input_nodes))
classifier.add(Dense(output_dim = num_hidden_nodes, init = "uniform", activation = "sigmoid"))

# Use sigmoid activation function for the output layer because we're predicting
# a probability that the stock price will go up
classifier.add(Dense(output_dim = num_output_nodes, init = "uniform", activation = "sigmoid"))

# Compile and train the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 25)

# Make predictions
Y_test_pred_prob = classifier.predict(X_test)
Y_test_pred = (Y_test_pred_prob >= 0.5)
Y_train_pred_prob = classifier.predict(X_train)
Y_train_pred = (Y_train_pred_prob >= 0.5)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def metrics(actual, pred):
    print("Accuracy:    ", round(accuracy_score(actual, pred) * 100, 2), "%")
    print("Precision:   ", round(precision_score(actual, pred) * 100, 2), "%")
    print("Recall:      ", round(recall_score(actual, pred) * 100, 2), "%")
    print("F1 Score:    ", round(f1_score(actual, pred) * 100, 2), "%")

print("Test set =============")
metrics(Y_test, Y_test_pred)
print("")
print("Train set ============")
metrics(Y_train, Y_train_pred)