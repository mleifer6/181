from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import math

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        classes = 3
        n = len(X)

        # Calculate Means
        means = np.asmatrix(np.zeros((classes,len(X[0]))))
        class_counts = [0 for i in range(classes)]
        for i in range(n):
            k = Y[i]
            means[k] += X[i]
            class_counts[k] += 1.0
        for k in range(classes):
            means[k] /= class_counts[k]

        self.means = means
        self.class_counts = class_counts
        total = sum(class_counts)

        # Calculate Covariances
        covariances = [np.asmatrix(np.zeros((len(X[0]),len(X[0])))) for i in range(classes)]
        for i in range(n):
            k = Y[i]
            covariances[k] += (np.dot((X[i] - means[k]).T, (X[i] - means[k])) / class_counts[k])

        if self.isSharedCovariance:
            shared = np.asmatrix(np.zeros((len(X[0]),len(X[0]))))
            for s_i in covariances:
                shared += s_i * float(class_counts[k]) / n
            self.covariances = [shared]
        else:
            self.covariances = covariances

        class_probs = map(lambda x : x / sum(class_counts), class_counts)
        self.priors = class_counts
        #print means
        # Calculate likelihood
        likelihood = 0
        if self.isSharedCovariance:
            for i in range(n):
                tmp = [means[Y[i]][0,0], means[Y[i]][0,1]]
                likelihood -= np.log(class_probs[Y[i]] * multivariate_normal.pdf(X[i], mean = tmp, cov = covariances[0]))
        else:
            for i in range(n):
                tmp = [means[Y[i]][0,0], means[Y[i]][0,1]]
                likelihood -= np.log(class_probs[Y[i]] * multivariate_normal.pdf(X[i], mean = tmp, cov = covariances[Y[i]]))
        if self.isSharedCovariance:
            print "Shared Cov Negative Log Likelihood: %f" % likelihood
        else:
            print "Not Shared Cov Negative Log Likelihood: %f" % likelihood
        return likelihood

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        Y = []
        means = self.means
        covariances = self.covariances
        for x in X_to_predict:
            mle_k = -5
            mle_prob = -5
            for k in range(3):
                cov_key = 0 if self.isSharedCovariance else k
                tmp = [means[k][0,0], means[k][0,1]]
                p = self.priors[k] * multivariate_normal.pdf(x, mean = tmp, cov = covariances[cov_key])
                if p > mle_prob:
                    mle_prob = p
                    mle_k = k
            Y.append(mle_k)

        return np.array(Y)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min,
            y_max, .01)) 

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
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
