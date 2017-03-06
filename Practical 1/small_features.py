import csv

count = 0
with open("new_features_1024.csv",'rb') as read_in:
	with open("new_features_256.csv", 'wb') as write_to:
		read_in = csv.reader(read_in)
		write_to = csv.writer(write_to)
		for row in read_in:
			write_to.writerow(row[:257])
			if count % 10000 == 0:
				print count
			count += 1