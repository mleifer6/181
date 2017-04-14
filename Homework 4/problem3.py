# CS 181, Spring 2016
# Homework 4: Clustering
# Name: Matthew Leifer
# Email: matthewleifer@college.harvard.edu

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats
import sys

inf = float("inf")

class KMeans(object):
	# K is the K in KMeans
	# useKMeansPP is a boolean. If True, you should initialize using KMeans++
	def __init__(self, K, useKMeansPP):
		self.K = K
		print "K = %d" % (self.K)
		self.useKMeansPP = useKMeansPP

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	def fit(self, X):
		n = len(X)
		self.losses = []
		self.X = X
		if self.useKMeansPP:
			print "Using K-Means++"
			means = []
			means.append(np.random.randint(256, size = (28, 28)))
			for i in range(self.K - 1):
				probs = []
				for j in range(n):
					closest_dist = inf
					for k in range(len(means)):
						dist = np.linalg.norm(np.subtract(X[j], means[k])) #self.l2_dist(X[j], means[k])
						if dist < closest_dist:
							closest_dist = dist
					probs.append(closest_dist ** 2)
				s = sum(probs)
				probs = map(lambda x: x / float(s), probs)
				custom = stats.rv_discrete(name = 'custom', values = (np.arange(n), probs))
				new_mean = custom.rvs(size = 1)
				new_mean = new_mean[0]
				means.append(X[new_mean])
			# 	sys.stdout.write(str(i) + " ")
			# 	sys.stdout.flush()
			# print ""
		else:
			print "Not Using K-Means++"
			means = [np.random.randint(256, size = (28, 28)) for i in range(self.K)]
		
		r = {}
		for i in range(n):
			r[i] = 0
		
		changes_made = True
		while(changes_made):
			changes_made = False
			old_r = r
			r = {}
			# Assign Images to Clusters
			for i in range(n):
				closest_cluster = inf
				closest_dist = inf
				for k in range(self.K):
					dist = np.linalg.norm(np.subtract(X[i], means[k])) #self.l2_dist(X[i], means[k])
					if dist < closest_dist: 
						closest_dist = dist
						closest_cluster = k
				r[i] = closest_cluster

			# Update Means
			new_means = [np.zeros((28,28)) for i in range(self.K)]
			cluster_counts = [0 for i in range(self.K)]
			for i in range(n):
				k = r[i]
				new_means[k] += X[i]
				cluster_counts[k] += 1.0
			
			# Normalize Means
			for k in range(self.K):
				if cluster_counts[k] != 0:
					new_means[k] /= cluster_counts[k]
			means = new_means

			# See if anything changed
			#print old_r
			#print r
			count = 0
			loss = 0
			for i in range(n):
				if r[i] != old_r[i]:
					changes_made = True
					count += 1
				loss += np.linalg.norm(np.subtract(X[i], means[r[i]])) #self.l2_dist(X[i], means[r[i]])
			self.losses.append(loss)
			print "Changed = %d" % count
			#print "Loss = %f" % loss

		#print r
		#print self.losses
		print "Iterations: %d" % len(self.losses)
		self.means = means
		self.r = r

		# for i in range(len(self.losses)):
		# 	if i < len(self.losses) - 2:
		# 		if self.losses[i] < self.losses[i+1]:
		# 			print i, self.losses[i], self.losses[i+1], (self.losses[i+1] - self.losses[i]) / self.losses[i]


	# x1 and x2 are 28x28 arrays
	def l2_dist(self, x1, x2):
		total = 0
		for i in range(28):
			for j in range(28):
				total += (x1[i][j] - x2[i][j]) ** 2
		return total ** 0.5

	def get_r(self):
		return self.r

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		return self.means

	# This should return the arrays for D images from each cluster that are representative of the clusters.
	def get_representative_images(self, D):
		images = [[] for i in range(self.K)]
		n = len(self.X)
		#order = np.random.choice(n,n, replace = False) #range(n)
		X = self.X
		r = self.r
		means = self.means
		for i in range(n):
			if len(images[r[i]]) < D:
				cur_dist = np.linalg.norm(np.subtract(X[i], means[r[i]]))
				images[r[i]].append((cur_dist, i, X[i]))
			elif len(images[r[i]]) == D:
				cur_dist = np.linalg.norm(np.subtract(X[i], means[r[i]]))
				max_dist = np.linalg.norm(np.subtract(images[r[i]][D - 1][2] , means[r[i]]))
				if cur_dist < max_dist:
					del(images[r[i]][D - 1])
					images[r[i]].append((cur_dist, i, X[i]))

			images[r[i]].sort()
		return images

	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	def create_image_from_array(self, img_array, img_number = None, cluster = None, show = False, is_mean = False):
		if cluster == None:
			cluster = self.r[img_number]
		title = ""
		if is_mean:
			title += "Mean_for_"
		title += "Cluster_=_" + str(cluster)
		if not is_mean:
			title += ";_Img_" + str(img_number)
		 
		plt.figure()
		plt.imshow(img_array, cmap='Greys_r')
		plt.title(title)
		if show:
			plt.show()
		plt.savefig(title+".png")
		return

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 10
D = 4
KMC = KMeans(K, useKMeansPP=False)
KMC.fit(pics)

means = KMC.get_mean_images()
for i in range(len(means)):
	KMC.create_image_from_array(means[i], cluster = i, is_mean = True)

samples = KMC.get_representative_images(D)
# for k in range(len(samples)):
#  	for (dist, i, image) in samples[k]:
#  		KMC.create_image_from_array(image, img_number = i)


for k in range(len(samples)):
	f, axarr = plt.subplots(2,2)
	samples[0][0][2]
	axarr[0,0].imshow(samples[k][0][2], cmap='Greys_r')
	axarr[0,1].imshow(samples[k][1][2], cmap='Greys_r')
	axarr[1,0].imshow(samples[k][2][2], cmap='Greys_r')
	axarr[1,1].imshow(samples[k][3][2], cmap='Greys_r')
	title = "Cluster_%d_Images.png" % k
	plt.savefig(title)

plt.figure()
plt.scatter(range(len(KMC.losses)), KMC.losses)
plt.savefig("Loss_for_K_=_" + str(K) + ".png")
