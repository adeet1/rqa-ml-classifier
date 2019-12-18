import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import raw data set
df0 = pd.read_excel("aapl.xlsx").iloc[2:, :].reset_index(drop = True)

# Remove the first 14 rows and the last row (we don't have future data in the present)
df0 = df0.iloc[14:-1, :].reset_index(drop = True)
df0["Wiki Move"] = df0["Wiki Move"].astype(int)
df0["Goog ROC"] = df0["Goog ROC"].astype(float)

# Select columns from data set
df = df0[["Open", "Close", "High", "Low", "RS", "Wiki Traffic- 1 Day Lag", "Wiki 5day disparity", "Wiki Move", "Wiki MA3 Move", "Wiki MA5 Move", "Wiki EMA5 Move", "Goog RS", "Goog MA3", "Goog MA5", "Goog EMA5 Move", "Goog 3day Disparity Move", "Goog ROC Move", "Goog RSI Move", "Wiki 3day Disparity", "Price RSI Move", "Google_Move", "Target"]]

# List of features
features = ["Wiki Traffic- 1 Day Lag", "Wiki 5day disparity", "Wiki Move", "Wiki MA3 Move", "Wiki MA5 Move", "Wiki EMA5 Move", "Goog MA3", "Target"]

# Select features
df = df[features]

# Perform forward chaining cross validation
def cross_validation(df, k):
    # k = number of folds
    split = np.split(df, k)

    train = pd.DataFrame()
    for i in range(k - 1):
        train = pd.concat([train, pd.DataFrame(split[i])])
        print("Train:", min(train.index), "to", max(train.index))
        X_train = train.iloc[:, :-1]
        Y_train = train.iloc[:, -1]

        test = pd.DataFrame(split[i + 1])
        print("Test:", min(test.index), "to", max(test.index))
        X_test = test.iloc[:, :-1]
        Y_test = test.iloc[:, -1]

        # Fit the model
        model = RandomForestClassifier(n_estimators=200)
        model.fit(X_train, Y_train)

        # Make predictions
        Y_test_pred = pd.Series(model.predict(X_test)).astype(int)
        Y_train_pred = pd.Series(model.predict(X_train)).astype(int)

        # Results
        
        labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
        test_scores = [accuracy_score(Y_test, Y_test_pred),
                       precision_score(Y_test, Y_test_pred),
                       recall_score(Y_test, Y_test_pred),
                       f1_score(Y_test, Y_test_pred)]
        train_scores = [accuracy_score(Y_train, Y_train_pred),
                        precision_score(Y_train, Y_train_pred),
                        recall_score(Y_train, Y_train_pred),
                        f1_score(Y_train, Y_train_pred)]
        metrics = pd.DataFrame()
        metrics["Index"] = labels
        metrics["Test Score"] = test_scores
        metrics["Train Score"] = train_scores
        metrics = metrics.set_index("Index")
        print(metrics)

#########################################

cross_validation(df, 5)
