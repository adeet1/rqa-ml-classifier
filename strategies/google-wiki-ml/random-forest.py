import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from feature_selector import FeatureSelector

# Import raw data set
df0 = pd.read_excel("aapl.xlsx").iloc[2:, :].reset_index(drop = True)

# Remove the first 14 rows and the last row (we don't have future data in the present)
df0 = df0.iloc[14:-1, :].reset_index(drop = True)
df0["Wiki Move"] = df0["Wiki Move"].astype(int)
df0["Goog ROC"] = df0["Goog ROC"].astype(float)

# Select columns from data set
df = df0[["Open", "Close", "High", "Low", "RS", "Wiki Traffic- 1 Day Lag", "Wiki 5day disparity", "Wiki Move", "Wiki MA3 Move", "Wiki MA5 Move", "Wiki EMA5 Move", "Goog RS", "Goog MA3", "Goog MA5", "Goog EMA5 Move", "Goog 3day Disparity Move", "Goog ROC Move", "Goog RSI Move", "Wiki 3day Disparity", "Price RSI Move", "Google_Move", "Target"]]

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

# Perform forward chaining cross validation
def cross_validation(df, k):
    # k = number of folds
    split = np.split(df, k)

    train = pd.DataFrame()

    results = np.zeros(8)
    # Loop ==================
    for i in range(k - 1):
        train = pd.concat([train, pd.DataFrame(split[i])])
        X_train = train.iloc[:, :-1]
        Y_train = train.iloc[:, -1]

        test = pd.DataFrame(split[i + 1])
        X_test = test.iloc[:, :-1]
        Y_test = test.iloc[:, -1]

        # Feature scaling
        sc = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(sc.transform(X_train), columns = X_train.columns)
        X_test = pd.DataFrame(sc.transform(X_test), columns = X_test.columns)
        
        # Feature selection (remove highly correlated features)
        n = len(X_train.columns)
        fs = FeatureSelector(data = X_train, labels = X_train.columns)
        fs.identify_collinear(correlation_threshold = 0.7) # select features from training set
        corr = fs.ops['collinear']
        X_train = fs.remove(methods = ['collinear']) # remove selected features from training set
        to_remove = pd.unique(fs.record_collinear['drop_feature']) # features to remove
        X_test = X_test.drop(columns = to_remove) # remove selected features from test set
        print("Data has", n, "features, but using", len(X_train.columns))

        # Principal component analysis (PCA)
        pca = PCA(n_components = 5, random_state = None)
        X_train = pd.DataFrame(pca.fit_transform(X_train))
        X_test = pd.DataFrame(pca.transform(X_test))
        
        # Fit the model
        model = RandomForestClassifier(n_estimators=200, max_depth=4)
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

        # Concatenate the lists of test and train scores
        current_scores = train_scores + test_scores

        # Add them to the results
        results = np.add(results, current_scores)

        # Compute average train/test scores
        # Display progress
        print("Train: rows", min(train.index), "to", max(train.index),
              "\t", np.round(np.array(results) / (i + 1), 3)[0:4],
              "\t\tTest: rows", min(test.index), "to", max(test.index),
              "\t", np.round(np.array(results) / (i + 1), 3)[4:])

    # Feature importances
    feature_imp = {}
    for i in range(len(X_train.columns)):
        col = X_train.columns[i]
        try:
            val = model.feature_importances_[i]
        except AttributeError:
            val = -1 # if feature importances are undefined, use -1 as a sentinel value
        feature_imp[col] = [val]
        
    feature_imp = pd.DataFrame(feature_imp).T
    feature_imp.columns = ["Importance"]
    feature_imp = feature_imp.sort_values(by = "Importance", ascending = False)
    print(feature_imp)
    
    # End of loop ==================

#########################################

cross_validation(df, 5)
