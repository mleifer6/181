# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import random
import numpy as np
from sklearn.svm import SVC
from Perceptron import Perceptron
import itertools
import time


def kernel(K, x_t, x_i):
	return K(x_t, x_i)

def trivial(x_t, x_i):
	return np.dot(x_t, x_i)

def gaussian(x_t, x_i, sigma = 0.05):
	dist = 0
	for i in range(len(x_t)):
		dist += (x_t[i] - x_i[i]) ** 2
	dist /= 2 * sigma
	dist *= -1
	return np.exp(dist)

# Implement this class
class KernelPerceptron(Perceptron):
	def __init__(self, numsamples):
		self.numsamples = numsamples
	
	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		S = {}
		b = 0
		order = [random.randint(0,len(X) - 1) for i in range(self.numsamples)]
		for i in range(self.numsamples):
			t = order[i]
			x_train = X[t]
			y_train = Y[t]
			y_hat = b
			for k, alpha in S.items():
				y_hat += alpha * kernel(trivial, x_train, X[k])
			if y_train * y_hat <= 0:
				S[t] = y_train
		self.S = S
		self.b = b
		print "# of Support Vectors = %d" % len(S)

	# Implement this!
	def predict(self, X):
		Y = []
		for x in X:
			y_hat = self.b
			for i, alpha in self.S.items():
				y_hat += alpha * kernel(trivial, x, self.X[i])
			prediction = 1 if y_hat > 0 else -1

			Y.append(prediction)
		return np.asarray(Y)
	
# Implement this class
class BudgetKernelPerceptron(Perceptron):
	def __init__(self, beta, N, numsamples):
		self.beta = beta
		self.N = N
		self.numsamples = numsamples
	
	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		S = {}
		y_hats = {}
		b = 0
		order = [random.randint(0,len(X) - 1) for i in range(self.numsamples)]
		for k in range(self.numsamples):
			t = order[k]
			x_t = X[t]
			y_t = Y[t]
			y_hat = b
			max_seen = - 2 ** 32
			max_i = -2 ** 32
			for i, alpha in S.items():
				y_hat += alpha * kernel(trivial, x_t, X[i])
				if Y[i] * (y_hats[i] - S[i] * kernel(trivial, X[i], X[i])) > max_seen:
					max_seen = Y[i] * (y_hats[i] - S[i] * kernel(trivial, X[i], X[i])) > max_seen
					max_i = i
			y_hats[t] = y_hat
			if y_t * y_hat <= self.beta:
				S[t] = y_t
				if len(S) > self.N:
					S.pop(i)
		self.S = S
		self.b = b

	# Implement this!
	def predict(self, X):
		Y = []
		for x in X:
			y_hat = self.b
			for i, alpha in self.S.items():
				y_hat += alpha * kernel(trivial, x, self.X[i])
			prediction = 1 if y_hat > 0 else -1

			Y.append(prediction)

		return np.asarray(Y)

class SMO(Perceptron):
	def __init__(self, tau, C):
		self.tau = tau
		self.C = C

	def _grad(self, alpha, i):
		total = 0
		xi = self.X[i]
		n = len(self.X)
		for j in range(n):
			total += alpha[j] * kernel(trivial, xi, self.X[j])
		return self.Y[i] - total + self.b

	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		n = len(X)
		b = 0
		self.b = b
		grad = [Y[i] + b for i in range(n)]
		A = [min(0, self.C * Y[i]) for i in range(n)]
		B = [max(0, self.C * Y[i]) for i in range(n)]
		# i is added to alpha iff it's nonzero
		alpha = [0 for i in range(n)]
		violating = True
		print "got here"
		while(violating):
			# Violating pair i,j
			vi_i = -10
			vi_j = -10
			for i, j in itertools.product(range(n),range(n)):
				# is violating pair
				gi = self._grad(alpha, i)
				gj = self._grad(alpha, j)
				if alpha[i] < B[i] or alpha[j] > A[j] or (gi - gj) > self.tau:
					vi_i = i
					vi_j = j
					break
			if(vi_i == -10 or vi_j == -10):
				violating = False
				break
			print vi_i, vi_j, trivial(X[vi_i], X[vi_i]) +  trivial(X[vi_j], X[vi_j]) - 2 * trivial(X[vi_i], X[vi_j])
			lamb = min((gi - gj) / (trivial(X[vi_i], X[vi_i]) +  trivial(X[vi_j], X[vi_j]) - 2 * trivial(X[vi_i], X[vi_j]) ), B[vi_i] - alpha[vi_i], alpha[vi_j] - A[vi_j])
			alpha[vi_i] += lamb
			alpha[vi_j] -= lamb
			for s in range(n):
				grad[s] -= lamb * (trivial(X[vi_i], X[s]) - trivial(X[vi_j], X[s]))

			self.alpha = alpha

	def predict(self, X):
		Y = []
		for x in X:
			y_hat = self.b
			n = len(alphas)
			for i in range(n):
				y_hat += self.alpha[i] * trivial(x, self.X[i])
			prediction = 1 if y_hat > 0 else -1

			Y.append(prediction)

		return np.asarray(Y)


# Inherit for the visualize method
class LASVM(Perceptron):
	def __init__(self, iterations, seed_size, M, tau, C):
		self.seed_size = seed_size
		self.iterations = iterations
		self.M = M
		self.S = {}
		self.tau = tau
		self.C = C
		self.delta = tau + 1.0


	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		n = len(X)
		for i in range(self.seed_size):
			self.S[i] = {"alpha" : 0, "g" : 0}

		self.A = [min(0, self.C * Y[i]) for i in range(n)]
		self.B = [max(0, self.C * Y[i]) for i in range(n)]
		for time in range(self.iterations):
			examples = np.random.choice(n, self.M, replace = False)
			min_i = 2 ** 32
			min_y_hat = 2 ** 32
			for s_i in examples:
				y_hat = 0
				for s in self.S:
					y_hat += self.S[s]["alpha"] * trivial(X[s_i], X[s])
				if abs(y_hat) < min_y_hat:
					min_i = s_i
					min_y_hat = y_hat
			self.process(min_i, min_y_hat)
			self.reprocess()

		while self.delta > self.tau:
			self.reprocess()
	
	def is_violating(self,i,j):
		return self.S[i]["alpha"] < self.B[i] and self.S[j]["alpha"] > self.A[j] and self.S[i]['g'] - self.S[j]['g'] > self.tau

	def process(self, min_i, min_y_hat):
		if min_i in self.S:
			return 
		self.S[min_i] = {}
		self.S[min_i]["alpha"] = 0
		self.S[min_i]["g"] = self.Y[min_i] - min_y_hat
		A = self.A
		B = self.B
		
		if Y[min_i] == 1:
			i = min_i
			min_grad =  2 ** 32
			j = -10
			for s in self.S:
				if self.S[s]["alpha"] > A[s] and self.S[s]["g"] < min_grad:
					j = s
					min_grad = self.S[s]["g"]

		else:
			j = min_i
			max_grad = - 2 ** 32
			i = -10
			for s in self.S:
				if self.S[s]["alpha"] < B[s] and self.S[s]["g"] > max_grad:
					i = s
					max_grad = self.S[s]["g"]

		if not self.is_violating(i,j):
			return 

		gi = self.S[i]["g"]
		gj = self.S[j]["g"]
		lamb = min((gi - gj) / (trivial(X[i], X[i]) +  trivial(X[j], X[j]) - 2 * trivial(X[i], X[j]) ), self.B[i] - self.S[i]["alpha"], self.S[j]["alpha"] - self.A[j])
		self.S[i]["alpha"] += lamb
		self.S[j]["alpha"] -= lamb
		for s in self.S:
			self.S[s]['g'] -= lamb * (trivial(X[i], X[s]) - trivial(X[j], X[s]))

	def reprocess(self):
		min_grad = 2 ** 32
		j = -10
		max_grad = - 2 ** 32
		i = -10
		for s in self.S:
			if self.S[s]["alpha"] < self.B[s] and self.S[s]["g"] > max_grad:
				max_grad = self.S[s]["g"]
				i = s
			if self.S[s]["alpha"] > self.A[s] and self.S[s]["g"] < min_grad:
				min_grad = self.S[s]["g"]
				j = s

		if not self.is_violating(i,j):
			return 

		gi = self.S[i]["g"]
		gj = self.S[j]["g"]
		lamb = min((gi - gj) / (trivial(X[i], X[i]) +  trivial(X[j], X[j]) - 2 * trivial(X[i], X[j]) ), self.B[i] - self.S[i]["alpha"], self.S[j]["alpha"] - self.A[j])
		self.S[i]["alpha"] += lamb
		self.S[j]["alpha"] -= lamb
		for s in self.S:
			self.S[s]['g'] -= lamb * (trivial(X[i], X[s]) - trivial(X[j], X[s]))

		min_grad = 2 ** 32
		j = -10
		max_grad = - 2 ** 32
		i = -10
		for s in self.S:
			if self.S[s]["alpha"] < self.B[s] and self.S[s]["g"] > max_grad:
				max_grad = self.S[s]["g"]
				i = s
			if self.S[s]["alpha"] > self.A[s] and self.S[s]["g"] < min_grad:
				min_grad = self.S[s]["g"]
				j = s

		# Not allowed by python to delete an entry from a dict during a for loop
		to_delete = []
		for s in self.S:
			if self.S[s]["alpha"] == 0:
				if Y[s] == -1 and self.S[s]["g"] >= self.S[i]["g"]:
					to_delete.append(s)
				elif Y[s] == 1 and self.S[s]["g"] <= self.S[j]["g"]:
					to_delete.append(s)

		for s in to_delete:
			del(self.S[s])

		self.b = 0.5 * (self.S[i]["g"] + self.S[j]["g"])
		self.delta = self.S[i]["g"] - self.S[j]["g"]
		
	def predict(self, X):
		Y = []
		for x in X:
			y_hat = self.b
			for s in self.S:
				y_hat += self.S[s]["alpha"] * trivial(x, self.X[s])
			prediction = 1 if y_hat > 0 else -1

			Y.append(prediction)

		return np.asarray(Y)

def visualize(self, X, Y, output_file, width=3, show_charts=False, save_fig=True, include_points=True):
	self.X = X
	self.Y = Y
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
	if include_points:
		plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
	if save_fig:
		plt.savefig(output_file)
	if show_charts:
		plt.show()

def accuracy(Y_pred, Y_true):
	total = len(Y_pred)
	correct = 0.0
	for i in range(len(Y_pred)):
		if Y_pred[i] == Y_true[i]:
			correct += 1
	print "Correct: %d" % (correct)
	print "Incorrect: %d" % (total - correct)
	return correct / total


# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]
validation = np.loadtxt("val.csv", delimiter=',')
X_val = validation[:, :2]
Y_val = validation[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0.01
N = 100
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'
sk_learn_file_name = 'sklearn_SVC.png'
smo_file_name = 'smo.png'
lasvm_file_name = 'lasvm.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
############################## Kernel Perceptron ##############################
"""
k = KernelPerceptron(numsamples)
start = time.time()
k.fit(X,Y)
elapsed = time.time() - start
print "Kernel Perceptron Training time = %s sec" % (elapsed)

y_train_pred = k.predict(X)
print "Kernel Perceptron Training Accuracy = %s\n" % accuracy(y_train_pred, Y)
y_pred = k.predict(X_val)
print "Kernel Perceptron Validation Accuracy = %s\n" % accuracy(y_pred, Y_val)
k.visualize(kernel_file_name, width=0, show_charts=False, save_fig=True, include_points=False) 

print "#" * 80
#"""
########################## Budget Kernel Perceptron ###########################
"""
bk = BudgetKernelPerceptron(beta, N, numsamples)

start = time.time()
bk.fit(X, Y)
elapsed = time.time() - start
print "Budget Kernel Perceptron Training time = %s sec" % (elapsed)
print "Beta = %.3f\nN = %d" % (beta, N)

y_train_pred = bk.predict(X)
print "Budget Kernel Perceptron Training Accuracy = %s\n" % accuracy(y_train_pred, Y)
y_pred = bk.predict(X_val)
print "Budget Kernel Perceptron Validation Accuracy = %s\n" % accuracy(y_pred, Y_val)
bk.visualize(budget_kernel_file_name, width=0, show_charts=False, save_fig=True, include_points=False)

print "#" * 80
#"""
################################ SK-Learn SVM #################################
"""
sk = SVC(kernel = 'linear')
start = time.time()
sk.fit(X,Y)
elapsed = time.time() - start
print "SK-Learn SVM Training time = %s sec" % (elapsed)

y_train_pred = sk.predict(X)
print "SK Learn SVM Training Accuracy = %s\n" % accuracy(y_train_pred, Y)
y_pred = sk.predict(X_val)
print "SK-Learn SVM Validation Accuracy = %s\n" % accuracy(y_pred, Y_val)
visualize(sk, X, Y, sk_learn_file_name, width = 0, show_charts = False, save_fig =True, include_points=False)
#"""


"""
smo = SMO(1, 10)
smo.fit(X,Y)
smo.visualize(smo_file_name, width=0, show_charts=False, save_fig=True, include_points=False)
"""

lasvm = LASVM(3, 250, 250, 0.01, 1000)
start = time.time()
lasvm.fit(X,Y)
elapsed = time.time() - start
print "LASVM Training time = %s sec" % (elapsed)

y_train_pred = lasvm.predict(X)
print "LASVM Training Accuracy = %s\n" % accuracy(y_train_pred, Y)
y_pred = lasvm.predict(X_val)
print "LASVM  Validation Accuracy = %s\n" % accuracy(y_pred, Y_val)

lasvm.visualize(lasvm_file_name, width=0, show_charts=False, save_fig=True, include_points=False)

