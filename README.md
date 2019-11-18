# week-9-project
Here are some of the ideas we discussed as a trio in our last class and that I've expanded on. 
We were tasked to identify how we would go about identifying which class a new point would belong to in a 'clustering' graph such as Iris.
What Anthony, Ozge and I thought, is that it would be good to identify our parameters.
We identified what are our attributes, and what are our targets. 
(at this moment the class was stopped to discuss a problem we hadn't foreseen but a necessary step, the way to get new points, is to split the dataset into a 'test' and 'train' subset)

What we discussed were two possible options. First, calculating each attributes mean, and doing the [mean - the new points data] to find out the distance between them and the shortest distance would group the point to that attribute. Second, was a regression line, that would be plotted for each class (setosa, virginica, versicolor), the distance of each new point to the regression line would be calculated and the shortest would determine the class. 

We did a little bit of coding as a group, where we came up with how to calculate the distance between the points for the first method described above 

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.datasets import load_iris
iris = load_iris()
print(dir(iris))
print(iris.DESCR)
print(iris.data)
print(iris.feature_names)
print(iris.target_names)
print(iris.target)
print(iris.day[:,2])
np.where(iris.target == 0)
iris.data[np.where(iris.target == 0), 2]
petal_length_setosa = iris.data[np.where(iris.target == 0), 2]
print(petal_length_setosa.mean(),petal_length_setosa.max(),petal_length_setosa.min())
#
#
# for flower_location, flower in enumerate(iris.data):
#    if iris.target[flower_location] == 0:
#        print(flower[2])
# flower_feature_location = 2
# flower_type = 0
# iris.data[np.where(iris.target == flower_type), flower_feature_location]


I've decided to use the libraries available to do the rest of the project.
