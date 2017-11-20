import pandas as pd
import time

# Grab the DLA HAR dataset from:
# http://groupware.les.inf.puc-rio.br/har
# http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip


# TODO: Load up the dataset into dataframe 'X'
X = pd.read_csv('Datasets/dataset-har-PUC-Rio-ugulino.csv',sep=';', decimal=",")

print X.head(5)
#print X.describe()
print X.dtypes
print("NaNs:",pd.isnull(X).sum())

# TODO: Encode the gender column, 0 as male, 1 as female
X.gender = X.gender.map({'Woman':1, 'Man':0})


# TODO: Clean up any column with commas in it
# so that they're properly represented as decimals instead

# INFO: Check data types
print X.dtypes
print X.head(5)
print("NaNs:",pd.isnull(X).sum())

# TODO: Convert any column that needs to be converted into numeric
# use errors='raise'. This will alert you if something ends up being
# problematic

X.how_tall_in_meters = pd.to_numeric(X.how_tall_in_meters, errors='coerce')
X.body_mass_index = pd.to_numeric(X.body_mass_index, errors='coerce')
X.z4 = pd.to_numeric(X.z4, errors='coerce')

# INFO: If you find any problematic records, drop them before calling the
# to_numeric methods above...
X = X.dropna(axis=0).reset_index(drop=True)


# TODO: Encode your 'y' value as a dummies version of your dataset's "class" column
#
#-X['class'] = X['class'].map({'sitting':0, 'sittingdown':1, 'standing':2, 'standingup':3, 'walking':4})
#-y = X['class'].copy()
y=pd.get_dummies(X['class'])

# TODO: Get rid of the user and class columns

X.drop(labels=['class','user'], inplace=True, axis=1)
print X.head(5),"\n", y.head(5)
print X.describe()
print X.dtypes


# INFO: An easy way to show which rows have nans in them
print X[pd.isnull(X).any(axis=1)]

# TODO: Create an RForest classifier 'model' and set n_estimators=30,
# the max_depth to 10, and oob_score=True, and random_state=0

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=30, max_depth=10, oob_score=True, random_state=0)

# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=7)


print "Fitting..."
s = time.time()

# TODO: train your model on your training set
model.fit(X_train, y_train)
print "Fitting completed in: ", time.time() - s

# INFO: Display the OOB Score of your data
score = model.oob_score_
print "OOB Score: ", round(score*100, 3)


print "Scoring..."
s = time.time()

# TODO: score your model on your test set
score=model.score(X_test, y_test)
print "Score: ", round(score*100, 3)
print "Scoring completed in: ", time.time() - s


# TODO: Answer the lab questions, then come back to experiment more
# TODO: Try playing around with the gender column
# Encode it as Male:1, Female:0
# Try encoding it to pandas dummies
# Also try dropping it. See how it affects the score
# This will be a key on how features affect your overall scoring
# and why it's important to choose good ones.

# TODO: After that, try messing with 'y'. Right now its encoded with
# dummies try other encoding methods to experiment with the effect.

