# Complex Region Classification Problem

import numpy as np
from random import uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Let's create the C1 and C2 regions, (red and blue):
def isinC1(x, y):
    """ test whether a point lies within C1 """

    # anular region or half donut:
    # inside a circle of radius 2
    anular_max_bound = (x**2 + y**2 <= 4)
    # inside a circle of radius 1
    anular_min_bound = (x**2 + y**2 >= 1)
    # on the 2nd and 3rd quadrants of the Cartesian plane
    anular_x = (x <= 0)

    # semicircular region:
    # this region lies on the 4th quadrant
    semi_inside = (x**2 + (y + 1)**2 <= 1)
    semi_x = (x > 0)

    # evaluate boolean conditions:
    if (anular_max_bound and anular_min_bound and anular_x) or (semi_inside and semi_x):
        return True
    else:
        return False

def isinC2(x, y):
    """ test whether a point lies within C2 """

    # anular region or half donut:
    # inside a circle of radius 2 with origin in y = -1
    anular_max_bound = (x**2 + (y + 1)**2 <= 4)
    # inside a circle of radius 1 with origin in y = -1
    anular_min_bound = (x**2 + (y + 1)**2 >= 1)
    # on the 1st and 4th quadrants of the Cartesian plane
    anular_x = (x >= 0)

    # semicircular region:
    # this region lies on the 2nd quadrant
    semi_inside = (x**2 + y**2 <= 1)
    semi_x = (x < 0)

    # evaluate boolean conditions:
    if (anular_max_bound and anular_min_bound and anular_x) or (semi_inside and semi_x):
        return True
    else:
        return False

# Now let's fill the regions with points:
c1list = []     # empty list of [x, y, 0], 0 is the red region tag or label
c2list = []     # empty list of [x, y, 1], 1 is the blue region tag or label

# let's make a "fill" function that will be used to fill each list separately:
def fill_region(lst, fun, region):
    """ throw random coordinates until a 1000 points are collected """

    while True:
        x = uniform(-2, 2)  # x range of an enclosing square
        y = uniform(-3, 2)  # y range of an enclosing square

        # append a pair of points if they lie within the region:
        # fun is either isinC1 or isinC2
        if fun(x, y):
            lst.append([x, y, region])

        # break if the length of the list is 1000:
        if len(lst) == 1000:
            break

# fill the lists c1list and c2list using the above function:
fill_region(c1list, isinC1, 0)
fill_region(c2list, isinC2, 1)

# convert lists into numpy arrays:
c1arr = np.asarray(c1list)
c2arr = np.asarray(c2list)

# concatenate arrays vertically, dimension will be (2000, 3):
# argument must be a tuple!
arr = np.vstack((c1arr, c2arr))

# shuffle the elements:
index = np.arange(0, 2000)
np.random.shuffle(index)
arr = arr[index, :]


# -----------------------------------------------------------------------------
# Setting the MLPClassifier
# -----------------------------------------------------------------------------

# X will be our features and y will be our labels (0 or 1):
# first two columns of arr (x and y):
X = arr[:, 0:2]
# last column of arr (label)
y = arr[:, -1]

# create a train set and a test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

# we have to scale the features to features with mean zero and variance 1,
# here it's not so important, but if you have much more features with each
# one having different ranges, then it can be a problem, e.g.
# a feature may range from -500 to 1500
# another feature may range from 0 to 0.01!

scaler = StandardScaler()
scaler.fit(X_train)

# now transform the train and test sets:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# let's create an instance of the MLPClassifer class:

# hidden_layer_sizes= 8 layers with 24 neurons each
# activation='relu' (good for classification problems)
# solver='sgd' (stochastic gradient descend)
# batch size is 100, remember there are 2000 samples!
# verbose prints to screen all details
# leave the rest as default 'cause I don't really use them

mlp = MLPClassifier(hidden_layer_sizes=(24, 8),
                    activation='relu',
                    solver='sgd',
                    alpha=0.0001,
                    batch_size=100,
                    learning_rate='adaptive',
                    learning_rate_init=0.005,
                    power_t=0.5,
                    max_iter=1000,
                    shuffle=False,
                    random_state=None,
                    tol=0.0001,
                    verbose=False,
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.99,
                    epsilon=1e-8,
                    n_iter_no_change=10)

# y_train must be a column vector!
mlp.fit(X_train, y_train.ravel())

# make the predictions:

# the mlp object has a predict function to which you pass the test set
predictions = mlp.predict(X_test)

# you'll see the confusion matrix when we run the code!
# in this case the confusion matrix is 2X2, in a perfect prediction, it would be:
# |400  0|
# |0  400|
# supposing the X_test has 800 features, 400 belong to region red and 400 to region blue.

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# -----------------------------------------------------------------------------
# Plotting our predictions
# -----------------------------------------------------------------------------

# first let's transform back the X_test set:
X_plot = scaler.inverse_transform(X_test)

# we wanna know what indices belong to class 0 in the y_test set:
# this returns a numpy arrays of booleans
idx0 = (y_test == 0)

# same goes for class 1:
idx1 = (y_test == 1)

# we also want to know what indices of y_test do not match our predictions:
# (misclassified labels)
midx = (y_test != predictions)

# PREDICTION?
# YES, PREDICTION.
# PAIN!

# now let's find out what labels belong to class 0 by passing them the indices:
X_0 = X_plot[idx0]
X_1 = X_plot[idx1]

# so these are the labels belonging to class 0 and class 1 respectively.

# the misclassified labels are:
M = X_plot[midx]

#  let's create a figure with matplotlib:
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# we use scatter plots

# the [:, 0] means all rows from column 0 (the x coordinate)
# the [:, 1] means all rows from column 1 (the y coordinate)
ax.scatter(X_0[:, 0], X_0[:, 1], c='red', marker='.')
ax.scatter(X_1[:, 0], X_1[:, 1], c='blue', marker='.')
ax.scatter(M[:, 0], M[:, 1], c='green', marker='X')

# and last, we have to show the plot:
plt.show()

# That's it. If I made a mistake here, it will surely be corrected
# in the file located in the repository. Excuse my bad typing skills!
# I'm an engineer, not a secretary.
