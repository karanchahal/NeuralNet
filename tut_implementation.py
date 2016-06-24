
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



# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(n), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./n * data_loss

def predict(model,X):
    W1,b1,W2,b2 = model['W1'], model['b1'],model['W2'],model['b2']
    # Forward Prop
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    #Softmax
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores,axis = 1,keepdims=True)
    #returns index of max
    return np.argmax(probs,axis=1)


def build_model(nn_hdim, epochs = 200,print_loss =False):

    #Initialize theparamters to random values between -1 and 1
    np.random.seed(0)
    W1 = np.random.randn(input_dim,hidden_dim) / np.sqrt(input_dim)
    b1 = np.zeros((1,hidden_dim))
    W2 = np.random.randn(hidden_dim,output_dim)/np.sqrt(hidden_dim)
    b2 = np.zeros((1,output_dim))

    #This is what we have to return
    model = {}

    # Gradient Descent
    for i in xrange(0, epochs):

        #Forward Propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)

        probs = exp_scores / np.sum(exp_scores,axis = 1,keepdims =True)


        # BackPropagation
        delta3 = probs
        delta3[range(n),y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3,axis=0,keepdims =True)
        delta2 = delta3.dot(W2.T)*(1 - np.power(a1,2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add Regularization terms
        dW2 += reg_lambda*W2
        dW1 += reg_lambda*W1

        # Gradient Descent parameter update
        W1 += -epsilon*dW1
        b1 += -epsilon*db1
        W2 += -epsilon*dW2
        b2 += -epsilon*db2

        model = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" %(i, calculate_loss(model))

    return model

np.random.seed(0) # for init random generator
X,y =  datasets.make_moons(200,noise=0.20)
plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.Spectral)
#plt.show()

n = len(X)
input_dim = 2
output_dim = 2

hidden_dim = 4

epsilon = 0.01 #learning rate
reg_lambda = 0.01 #lambda for regularization strength

model = build_model(3,print_loss=True)


# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x),X,y)
plt.title("Decision Boundary for hidden layer size 3")
