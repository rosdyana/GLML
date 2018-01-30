from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

# Load the iris dataset from seaborn.
iris = load_iris()

# Use the first 4 variables to predict the species.
X, y = iris.data[:, :4], iris.target

# Split both independent and dependent variables in half
# for cross-validation
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=0)

## Machine Learning
# # Train a scikit-learn log-regression model
# lr = LogisticRegressionCV()
# lr.fit(train_X, train_y)
#
# # Test the model. Print the accuracy on the test data
# pred_y = lr.predict(test_X)
# print("Accuracy is {:.2f}".format(lr.score(test_X, test_y))) # Accuracy is 0.83

## Deep Learning
# Build the keras model

model = Sequential()
# 4 features in the input layer (the four flower measurements)
# 16 hidden units
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))
# 32 hidden units
model.add(Dense(32))
model.add(Activation('sigmoid'))
# 3 classes in the ouput layer (corresponding to the 3 species)
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the keras model
model.fit(train_X, train_y, verbose=1, batch_size=1, epochs=100)

# Test the model. Print the accuracy on the test data
loss, accuracy = model.evaluate(test_X, test_y, verbose=0)
print("Accuracy is {:.2f} %".format(accuracy*100))
