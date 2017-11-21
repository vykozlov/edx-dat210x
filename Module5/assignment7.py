##############
# EXPLANAtion FROM EDX course stuff:
# This lab takes a lot of time, unless you automate the tuning of hyper-parameters using for-loops to traverse your search space. The way we coded it up was by first writing a for-loop to iterate over an array containing: 'uniform' and 'distance', and fed that into weights parameter of the KNeighborsClassifier() class constructor. Then we created another for-loop that iterated from 1-15, and fed that in as the n_neighbors parameter. We then ran the whole thing, altering which pre-processor was used, along with turning on/off Isomap / PCA. Here were our best results:
# No Preprocessing: 0.968571428571
# MaxAbsScaler(): 0.968571428571
# MinMaxScaler(): 0.968571428571
# StandardScaler(): 0.962857142857
# RobustScaler(): 0.962857142857
# Normalizer(): 0.857142857143
#
# Main areas to keep in mind here are:
# * Load your data correctly. The sample number, or id-field should not be added into your dataset as a feature, because it will destroy the integrity of your results.
# * After dropping features from your dataset, double check your dataset to make sure the feature was actually dropped. Many students forget to set the axis, and if pandas can't find the desired column to drop, it doesn't complain.
# * When pre-processing, fit_transform against training, and then transform testing.
# * Same goes for transforming with PCA / Isomap.
###############  

import numpy as np
import pandas as pd

from sklearn import preprocessing

# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = False


def plotDecisionBoundary(model, X, y):
  #print "Plotting..."
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()


# 
# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.

X = pd.read_csv('Datasets/breast-cancer-wisconsin.data', names=['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status'])
print X.head(5)
print X.describe()
print X.dtypes
X.nuclei = pd.to_numeric(X.nuclei, errors='coerce')
print X.dtypes
print("NaNs:",pd.isnull(X).sum())
print X[pd.isnull(X.nuclei)]
X['nuclei'] = X['nuclei'].fillna(X.nuclei.mean())
print "NaNs of nuclei after fillna:",pd.isnull(X.nuclei).sum()
        
# TODO: Copy out the status column into a slice, then drop it from the main
# dataframe. You can also drop the sample column, since that doesn't provide
# us with any machine learning power.

y = X['status'].copy()
X.drop(labels=['sample','status'], inplace=True, axis=1)
print X.head(5),"\n", y.head(5)


#
# TODO: With the labels safely extracted from the dataset, replace any nan values
# with the mean feature / column value

#already done above!

#
# TODO: Do train_test_split. Use the same variable names as on the EdX platform in
# the reading material, but set the random_state=7 for reproduceability, and keep
# the test_size at 0.5 (50%).

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=7)


# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation.
#
# .. your code here ..
#-from sklearn.preprocessing import MaxAbsScaler
#-scale = MaxAbsScaler()

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()

#-from sklearn.preprocessing import StandardScaler
#-scale = StandardScaler()

#-from sklearn.preprocessing import Normalizer
#-scale = Normalizer()

#from sklearn.preprocessing import RobustScaler
#scale = RobustScaler()

scale.fit(X_train)
XT_train = scale.transform(X_train)
XT_test = scale.transform(X_test)

#-XT_train = X_train
#-XT_test = X_test

#
# PCA and Isomap are your new best friends
model = None
if Test_PCA:
  print "Computing 2D Principle Components"
  #
  # TODO: Implement PCA here. save your model into the variable 'model'.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..
  #from sklearn.decomposition import RandomizedPCA
  #model = RandomizedPCA(n_components=2)  
  from sklearn.decomposition import PCA
  model = PCA(svd_solver='randomized',n_components=2)  

else:
  print "Computing 2D Isomap Manifold"
  #
  # TODO: Implement Isomap here. save your model into the variable 'model'
  # Experiment with K values from 5-10.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..
  from sklearn import manifold
  model = manifold.Isomap(n_neighbors=9, n_components=2)

# TODO: Train your model against data_train, then transform both
# data_train and data_test using your model. You can save the results right
# back into the variables themselves.

model.fit(XT_train)
X_train = model.transform(XT_train)
X_test = model.transform(XT_test)

# TODO: Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
weights=['uniform','distance']
for w in weights:
    print w
    for neighbors in range(1,16):
        knmodel = KNeighborsClassifier(weights=w,n_neighbors=neighbors)
        knmodel.fit(X_train, y_train)
# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the UDF
# weights don't give you any class information, the only way to introduce this
# data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.
#
# TODO: Calculate + Print the accuracy of the testing set
        predictions = knmodel.predict(X_test)
        print "kn=",neighbors,"acc=",accuracy_score(y_test, predictions)
        plotDecisionBoundary(knmodel, X_test, y_test)
