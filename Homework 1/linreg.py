#####################
# CS 181, Spring 2016
# Homework 1, Problem 3
#
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'congress-ages.csv'
times  = []
ages = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:
        # Store the data.
        times.append(float(row[0]))
        ages.append(float(row[1]))

# Turn the data into numpy arrays.
times = np.array(times)
ages = np.array(ages)

# Plot the data.
plt.plot(times, ages, 'o')
plt.xlabel("Congress age (nth Congress)")
plt.ylabel("Average age")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(times.shape), times)).T

# Nothing fancy for outputs.
Y = ages

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_times!!!!!
grid_times = np.linspace(75, 120, 200)
grid_X = np.vstack((np.ones(grid_times.shape), grid_times))
grid_Yhat  = np.dot(grid_X.T, w)

# Plot the data and the regression line.
plt.plot(times, ages, 'o', grid_times, grid_Yhat, '-')
plt.xlabel("Congress age (nth Congress)")
plt.ylabel("Average age")
plt.show()

def design_matrix(data, basis, max_j):
	n = len(data)
	result = np.empty((n, max_j + 1))
	for i in range(n):
		for j in range(max_j + 1):
			if j == 0:
				result[i,j] = 1
			else:
				result[i,j] = basis(data[i], j)
	return result

def poly(x,j):
	return x ** j

def sin(x, j):
	return np.sin(x / j)

count = 97
def make_graph(max_j, basis, xs = times, ys = ages):
	grid_times = np.linspace(75, 120, 200)
	n = len(grid_times)
	phi = design_matrix(xs, basis, max_j)
	w = np.linalg.solve(np.dot(phi.T,phi), np.dot(phi.T, ys))
	print w
	print "#"*80
	# Fill the grid_X matrix
	grid_X = np.empty((n, max_j + 1))
	for i in range(n):
		for j in range(max_j + 1):
			if j == 0:
				grid_X[i,j] = 1
			else:
				grid_X[i,j] = basis(grid_times[i], j)

	grid_Yhat = np.dot(grid_X, w)
	
	global count
	fig = plt.figure()
	b_name = "polynomial" if basis(1,1) == 1 else "sine"
	name = "Part " + chr(count) + ", j = " + str(max_j) + ", basis = " + b_name + ".png"
	fig.suptitle(name, fontsize = 25)
	plt.plot(xs, ys, 'o', grid_times, grid_Yhat, '-')
	plt.xlabel("Congress age (nth Congress)")
	plt.ylabel("Average age")
	#plt.show()
	fig.savefig("Graphs/" + name)
	count += 1


make_graph(6, poly)
make_graph(4, poly)
make_graph(6, sin)
make_graph(10, sin)
make_graph(22, sin)
