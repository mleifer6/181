import csv
import numpy as np
import sys
import time

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file = 'factorization_results.csv'

labmda = 10 ** -5
gamma = 10 ** -5

ARTISTS = 2000
USERS = 233286

Artists = {}
Users = {}
dim = 40
threshold = 10 ** -7
count = 0

start = time.time()

with open(train_file, 'rb') as train_fh:
	train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
	next(train_csv, None)

	for row in train_csv:
		user   = row[0]
		artist = row[1]
		plays  = int(row[2])
	
		if user not in Users:
			Users[user] = np.random.randint(1, 10, size = dim)
		if artist not in Artists:
			Artists[artist] = np.random.randint(1, 10, size = dim)

		e = plays - np.dot(Artists[artist], Users[user])
		#while (np.linalg.norm(gamma * (e * Users[user] - labmda * Artists[artist])) > threshold and
		#	   np.linalg.norm(gamma * (e * Artists[artist] - labmda * Users[user])) > threshold):
		Artists[artist] = Artists[artist] + gamma * (e * Users[user] - labmda * Artists[artist])
		Users[user] = Users[user] + gamma * (e * Artists[artist] - labmda * Users[user])
	
		#e = e - np.dot(Artists[artist], Users[user])
		count += 1
		sys.stdout.write(str(count) + " ")
		sys.stdout.flush()

end = time.time()
print "Time: %f\n" % (end - start)

test_error = 0
with open(train_file, 'rb') as train_fh:
	train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
	next(train_csv, None)
	for row in train_csv:
		user   = row[0]
		artist = row[1]
		plays  = int(row[2])

		prediction = np.dot(Users[user], Artists[artist])
		#print plays, prediction, abs(plays - prediction)
		test_error += abs(plays - prediction)


print "Test error = %f" % test_error

with open(test_file) as test_fh:
	test_csv = csv.reader(test_fh, delimiter = ',', quotechar='"')
	next(test_csv, None)

	with open(soln_file, 'w') as soln_fh:
		soln_csv = csv.writer(soln_fh,
							  delimiter=',',
							  quotechar='"',
							  quoting=csv.QUOTE_MINIMAL)
		soln_csv.writerow(['Id', 'plays'])

		for row in test_csv:
			Id = row[0]
			user   = row[1]
			artist = row[2]
			if np.dot(Users[user], Artists[artist]) > 0:
				soln_csv.writerow([Id, round(np.dot(Users[user], Artists[artist]), 4)])
			else:
				soln_csv.writerow([Id, 118])



def compare():
	total_error = 0
	with open("user_median.csv", 'rb') as median_fh:
		with open("factorization_results.csv", 'rb') as fact_fh:
			medians_csv = csv.reader(median_fh)
			fact_csv = csv.reader(fact_fh)
			med_rows = [row for row in medians_csv]
			fact_rows = [row for row in fact_csv]
			print len(med_rows)
			print len(fact_rows)
			for i in range(1, len(fact_rows)):
				total_error += abs(float(med_rows[i][1]) - float(fact_rows[i][1]))
	print total_error

compare()
"""
count = 0
total = 0
with open("factorization_results.csv", 'rb') as fact_fh:
	fact_csv = csv.reader(fact_fh)
	next(fact_csv, None)
	for row in fact_csv:
		if float(row[1]) == 118.0:
			count += 1
		total += 1
print count, total
"""