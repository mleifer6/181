import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp
import random

from sklearn import linear_model

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __softmax(self, W, x):
            xs = map(lambda w: np.dot(w,x), W)
            e_x = np.exp(xs - np.max(xs))

            mx = max(e_x / e_x.sum(axis=0))
            return e_x / e_x.sum(axis=0)

    def __gradient(self, W, X, C, l, eta, order):
        grad = [[0 for i in range(len(X[0]))] for j in range(len(W))]
        n = len(X)
        
        for k in range(3):
            for i in range(n):
                grad[k] += eta * np.dot(self.__softmax(W, X[order[i]])[k] - int(C[order[i]] == k), X[order[i]])
            grad[k] += map(lambda x: x*l, W[k])
        return grad 

    def __loss(self, W, X, C, l):
        loss = 0
        regularization_penalty = 0
        n = len(X)
        classes = 3
        for i in range(n):
            for k in range(classes):
                if (C[i] == k):
                    loss += np.log(self.__softmax(W,X[i])[k])
        loss *= (- 1.0 / n)
        for k in range(3):
            for j in range(len(W[0])):
                regularization_penalty += W[k][j] ** 2
        regularization_penalty *= (0.5 * l)
        return loss + regularization_penalty


    # Fits 3 classes
    def fit(self, X, C):
        self.X = X
        self.C = C
       
        
        # Randomly Generated Weights
        ws_current =[[52, 83], [62, 7], [73, 7]]
        print ws_current
        n = len(X)

        count = 0

        order = [i for i in range(n)]
        """
        lambdas = [10 ** -i for i in range(2,6)]
        etas = [10 ** -i for i in range(2,6)]
        for l in lambdas: 
            for eta in etas:
                losses = []
                count = 0
                ws_current =[[52, 83], [62, 7], [73, 7]]
                while count < 5000: 
                    change = self.__gradient(ws_current, X, C, l, eta, order)
                    for k in range(3):
                        ws_current[k] -= change[k]

                    count += 1
                    #losses.append(self.__loss(ws_current, X, C, l))
                    #print self.__loss(ws_current, X, C, l)
                    loss = self.__loss(ws_current, X, C, l)
                print str(eta) + ", " + str(l) + ", " + str(loss)
        """

        l = self.lambda_parameter
        eta = self.eta
        losses = []
        count = 0
        ws_current =[[52, 83], [62, 7], [73, 7]]
        while count < 10000: 
            change = self.__gradient(ws_current, X, C, l, eta, order)
            for k in range(3):
                ws_current[k] -= change[k]

            count += 1
            losses.append(self.__loss(ws_current, X, C, l))
        print losses[len(losses) - 1]
        self.ws = ws_current
        fig = plt.figure()
        plt.plot([i for i in range(count)], losses, label = "Loss")
        fig.suptitle("Lambda = " + str(self.lambda_parameter) + "; Eta = " + str(self.eta))
        fig.savefig("Logistic Regression Loss-1.png")
        
 
        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        Y = []
        W = self.ws
        print W

        for x in X_to_predict:
            ps =  list(self.__softmax(W, x))
            mle_p = max(ps)
            mle_k = ps.index(max(ps))
            mle_k = mle_k 
            Y.append(mle_k)
        #print Y
        return np.array(Y)

    def visualize(self, output_file, width=2, show_charts=True):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))
        
        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
