from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
# Part 1
def MLE(n_0,n_1):
	return n_1 / (n_0 + n_1)

def MAP(a,b,n_0,n_1):
	return (a + n_1 - 1) / (a + b + n_0 + n_1 - 2)

def post_pred(a, b, n_0, n_1):
	return (a + n_1) / (a + b + n_0 + n_1)


D = [0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,0]
a = 4
b = 2


n_0 = 0.0
n_1 = 0.0
MLEs = []
MAPs = []
post_preds = []

for i in range(len(D)):
	if D[i]:
		n_1 += 1
	else:
		n_0 += 1
	MLEs.append(MLE(n_0, n_1))
	MAPs.append(MAP(a,b,n_0,n_1))
	post_preds.append(post_pred(a,b,n_0,n_1))

fig = plt.figure()
xs = [i for i in range(1,1+len(D))]
plt.plot(xs, MLEs, label = 'MLE')
plt.plot(xs, MAPs, label = 'MAP')
plt.plot(xs, post_preds, label = 'Posterior Predictive')
plt.legend(loc = 'upper right')
fig.suptitle("MLE, MAP, Posterior Predictive Comparison")
plt.show()
fig.savefig("MLE, MAP, Posterior Predictive Comparison.png")

# Part 2
fig = plt.figure()
x = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), 1000)
prior = plt.plot(x, beta.pdf(x, a, b), label = str(a) + "," + str(b))

for i in range(len(D)):
	if D[i]:
		a += 1
	else: 
		b += 1
	if i % 4 == 3:
		x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.99, a, b), 100)
		res = plt.plot(x, beta.pdf(x, a, b), '-', label = str(a) + "," + str(b))
plt.legend(loc = 'upper right')
fig.suptitle("Theta Distribution after Updating")
plt.show()
fig.savefig("Theta Distribution after Updating")