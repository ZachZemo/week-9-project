import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor

os.makedirs('plots/iris_cluster', exist_ok=True)


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

plt.plot(f1_scores, color='green', label='f1 score', linestyle='--')
plt.plot(error_rate, color='red', label='error rate', linestyle='--')
plt.xlabel('n neighbors parameter')
plt.ylabel('f1_score/error_rate')
plt.legend()
plt.savefig('plots/iris_cluster/f1_scores-errors.png')
plt.close()


plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
plt.xlabel('Spea1 Length', fontsize=15)
plt.ylabel('Sepal Width', fontsize=15)
#plt.show()


km = KMeans(n_clusters = 3, n_jobs = 4, random_state=19)
km1 = km.fit(X)
centers = km.cluster_centers_
new_labels = km.labels_
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='flag',edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
plt.savefig('plots/iris_cluster/actual-predicted')
plt.close()




