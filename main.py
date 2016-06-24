import numpy as np
import sklearn
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()



# Generate a dataset and plot it
np.random.seed(0) # for init random generator
X,y =  datasets.make_moons(200,noise=0.20)
plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.Spectral)
plt.show()

# LOGISTIC REGRESSION FOR EXAMPLE
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X,y)

# Plot the decision boundary
plot_decision_boundary(lambda x: clf.predict(x),X,y)
plt.title("Logistic Regression")


def NeuralNetwork(X,y,hidden_layer = 1,number_of_features,output_layer = 2):
    weights = []

    shape = X.shape

    for i in range(hidden_layer):
        x = shape[1]
        y = number_of_features[i]
        w = np.zeros((x,y))
        shape = (x,y)
        weights.append(w)

    x = shape[1]
    y = output_layer
    w = np.zeros((x,y))

    for i in weights:
        print i.shape


NeuralNetwork(X,y)
