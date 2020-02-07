import pandas as pd

# Import data
df = pd.read_excel("aapl.xlsx").iloc[2:, :].reset_index(drop = True)

# Remove the first 14 rows and the last row (we don't have future data in the present)
df = df.iloc[14:-1, :].reset_index(drop = True)
df["Wiki Move"] = df["Wiki Move"].astype(int)
df["Goog ROC"] = df["Goog ROC"].astype(float)

# Remove redundant columns
df = df.drop(columns = ["Open", "Close", "High", "Low"]) # price info already reflected in features
df = df.drop(columns = ["Gain", "Loss"]) # already reflected in "Change in Close"
df = df.drop(columns = ["PE Ratio"]) # ignore this feature altogether
df = df.drop(columns = ["Goog Gain", "Goog Loss"]) # already reflected in "Change in Goog"
df = df.drop(columns = ["Wiki Traffic"]) # already reflected in "Wiki Traffic- 1 Day Lag"

# Train/test split
X = df.iloc[:, 1:-1]
Y = df.iloc[:, -1]

def train_test_split(X, Y, test_size):
    ind = int((1 - test_size) * len(X))
    X_train = X.iloc[:ind, :]
    X_test = X.iloc[ind:, :]
    Y_train = Y.iloc[:ind]
    Y_test = Y.iloc[ind:]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_train)
X_train = sc.transform(X_train)
X_train = pd.DataFrame(X_train, columns = X.columns)

# Calculate feature correlations
feature_correlations = X_train.corrwith(Y_train, method = "pearson")
feature_correlations = feature_correlations.sort_values(ascending = False)

# Feature selection
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance
import seaborn as sb
import matplotlib.pyplot as plt

def plot_clustermap(corr_matrix):
    # https://alphascientist.com/feature_selection.html
    corr_array = np.asarray(corr_matrix)
    
    linkage = hierarchy.linkage(distance.pdist(corr_array), method = "average")
    
    g = sb.clustermap(corr_matrix, row_linkage = linkage, col_linkage = linkage, row_cluster = True, col_cluster = True, figsize = (10, 10), cmap = "Greens")
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)
    plt.show()

corr_matrix = X_train.corr()
plot_clustermap(corr_matrix)

correlated_features = feature_correlations[np.abs(feature_correlations) >= 0.05].index.tolist()
corr_matrix_filt = X_train[correlated_features].corr()
plot_clustermap(corr_matrix_filt)
feature_correlations_filt = feature_correlations[np.abs(feature_correlations) >= 0.05].sort_values(ascending = False)

X_train = X_train[correlated_features]
X_test = X_test[correlated_features]

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