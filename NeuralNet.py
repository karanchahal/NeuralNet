import numpy as np
import sklearn
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def NeuralNetwork(X,y,hidden_layer = 1,number_of_features= [4],output_layer = 2):

    global weights,biases

    weights = []

    shape = X.shape

    for i in range(hidden_layer):
        x = shape[1]
        y = number_of_features[i]
        w = np.zeros((x,y))
        biases.append(np.zeros((1,y)))

        shape = (x,y)
        weights.append(w)

    x = shape[1]
    y = output_layer
    w = np.zeros((x,y))
    biases.append(np.zeros((1,y)))
    weights.append(w)





np.random.seed(0) # for init random generator
X,y =  datasets.make_moons(200,noise=0.20)
plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.Spectral)
#plt.show()
NeuralNetwork(X,y)
