# Import data
import pandas as pd
df = pd.read_excel("aapl.xlsx").iloc[2:, :].reset_index(drop = True)

# Remove the first 14 rows and the last row (we don't have future data in the present)
df = df.iloc[14:-1, :].reset_index(drop = True)
df["Wiki Move"] = df["Wiki Move"].astype(int)
df["Goog ROC"] = df["Goog ROC"].astype(float)

# Remove the columns that are used to calculate features
df = df.drop(columns = ["Open", "Close", "High", "Low"]) # prices reflected in other features
df = df.drop(columns = ["Gain", "Loss"]) # reflected in "Change in Close"
df = df.drop(columns = ["Average Gain", "Average Loss"]) # reflected in RS
df = df.drop(columns = ["RS"]) # reflected in 14-day Price RSI
df = df.drop(columns = ["Wiki Traffic"]) # reflected in Wiki Traffic- 1 Day Lag
df = df.drop(columns = ["Wiki Traffic- 1 Day Lag"]) # reflected in Wiki features
df = df.drop(columns = ["PE Ratio"]) # ignore this feature entirely
df = df.drop(columns = ["Wiki 5day disparity"]) # reflected in Wiki 5day Disparity Move
df = df.drop(columns = ["Wiki MA3 Move"]) # reflected in Wiki 3day Disparity
df = df.drop(columns = ["Wiki MA5 Move"]) # reflected in Wiki EMA5 Move
df = df.drop(columns = ["Goog Total"]) # reflected in Goog features
df = df.drop(columns = ["Goog Gain", "Goog Loss"]) # reflected in "Change in Goog"
df = df.drop(columns = ["Goog Avg. Gain", "Goog Avg. Loss"]) # reflected in RS
df = df.drop(columns = ["Goog RS"]) # reflected in "Change in Goog"
df = df.drop(columns = ["Goog ROC"]) # reflected in Goog ROC Move
df = df.drop(columns = ["Goog MA3"]) # reflected in Goog 3day Disparity
df = df.drop(columns = ["Goog EMA5"]) # reflected in Goog EMA5 Move
df = df.drop(columns = ["Goog 3day Disparity"]) # reflected in Goog 3day Disparity Move
df = df.drop(columns = ["Goog RSI (14 days)"]) # reflected in Goog RSI Move
df = df.drop(columns = ["Price RSI (14 days)"]) # reflected in Price RSI Move

# Fix column data format
df["Stochastic Oscillator (14 days)"] = df["Stochastic Oscillator (14 days)"].astype(float)

# Train/test split
X = df.iloc[:, 1:-1]
Y = df.iloc[:, -1]

def dataset_split(X, Y, test_size):
    ind = int((1 - test_size) * len(X))
    X_train = X.iloc[:ind, :]
    X_test = X.iloc[ind:, :]
    Y_train = Y.iloc[:ind]
    Y_test = Y.iloc[ind:]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = dataset_split(X, Y, test_size = 0.2)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_train)
X_train = pd.DataFrame(sc.transform(X_train), columns = X.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns = X.columns)

# Each feature's correlation with every other feature
corr_matrix = X_train.corr()

# Each feature's correlation with the target variable
feature_target_corr = X_train.corrwith(Y_train, method = "pearson")
feature_target_corr = feature_target_corr.sort_values(ascending = False)

# Each feature's correlation with every other feature
corr_matrix = X_train.corr()

# Select features based on correlation
all_features = X.columns.to_list()
selected_features = all_features[:]
for i in range(0, len(corr_matrix)):
    for j in range(i, len(corr_matrix)):
        # Process every pair of features
        corr = corr_matrix.iloc[i, j] 
        if abs(corr) > 0.5 and corr != 1:
            feature1 = all_features[i]
            feature2 = all_features[j]
            
            # Out of the two features in the current pair, remove the feature that
            # is less correlated with the target variable
            corr1 = feature_target_corr[feature1]
            corr2 = feature_target_corr[feature2]
            try:
                if abs(corr1) < abs(corr2):
                    selected_features.remove(feature1)
                    print("Removed from list of features:", feature1)
                else:
                    selected_features.remove(feature2)
                    print("Removed from list of features:", feature2)
            except ValueError:
                # Catch the error just in case we're trying to remove a feature
                # that's already been removed
                pass

X_train = X_train[selected_features]
X_test = X_test[selected_features]

corr_matrix_selected_features = X_train.corr()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200, max_depth = 4, random_state = 1)
model.fit(X_train, Y_train)

# Make predictions
Y_test_pred = pd.Series(model.predict(X_test)).astype(int)
Y_train_pred = pd.Series(model.predict(X_train)).astype(int)

# Evaluate model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
train_scores = np.array([accuracy_score(Y_train, Y_train_pred),
                         precision_score(Y_train, Y_train_pred),
                         recall_score(Y_train, Y_train_pred),
                         f1_score(Y_train, Y_train_pred)]) * 100
    
test_scores = np.array([accuracy_score(Y_test, Y_test_pred),
                        precision_score(Y_test, Y_test_pred),
                        recall_score(Y_test, Y_test_pred),
                        f1_score(Y_test, Y_test_pred)]) * 100

train_scores = np.round(train_scores, 1)
test_scores = np.round(test_scores, 1)

metrics = pd.DataFrame()
metrics["Training Set"] = train_scores
metrics["Test Set"] = test_scores
metrics = metrics.set_index(np.array(["Accuracy", "Precision", "Recall", "F1 Score"]))
print(metrics)