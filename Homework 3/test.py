import csv
import matplotlib.pyplot as plt


pos_xs = []
pos_ys = []
neg_xs = []
neg_ys = []
with open("data.csv", 'r') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		row = map(float, row)
		if row[2] == 1:
			pos_xs.append(row[0])
			pos_ys.append(row[1])
		else:
			neg_xs.append(row[0])
			neg_ys.append(row[1])
fig = plt.figure()
plt.scatter(pos_xs, pos_ys, color='green')
plt.scatter(neg_xs, neg_ys, color='red')
plt.show()
