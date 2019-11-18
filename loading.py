import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
#iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
column_names = iris.feature_names

#print(iris_df)

y = iris.target
X = iris.data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

f1_scores = []
error_rate = []

for i in range(1, 98):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    f1_scores.append(f1_score(y_test, y_predicted, average="macro"))
    error_rate.append(np.mean(y_predicted != y_test))

# Plotting results
plt.plot(f1_scores, color='green', label='f1 score', linestyle='--')
plt.plot(error_rate, color='red', label='error rate', linestyle='--')
plt.xlabel('n neighbors parameter')
plt.ylabel('f1_score/error_rate')
plt.legend()