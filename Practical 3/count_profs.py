import csv
from collections import Counter
train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'user_median.csv'

def artist_stats():
	count = 0
	train_data = {}
	with open(train_file, 'rb') as train_fh:
		train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
		next(train_csv, None)
		for row in train_csv:
			user   = row[0]
			artist = row[1]
			plays  = row[2]
		
			if not user in train_data:
				train_data[user] = {}

			train_data[user][artist] = int(plays)

	print count
	min_artists = 10 ** 10
	avg_artists = 0
	count = 0
	for user, user_data in train_data.iteritems():
		avg_artists += len(user_data)
		if len(user_data) < min_artists:
			min_artists = len(user_data)
		count += 1.0
	print avg_artists / count
	print min_artists

def missing():
	missing_count = 0
	missed = 0
	total = 0
	with open("profiles.csv", 'rb') as profs_fh:
		prof_csv = csv.reader(profs_fh, delimiter = ',', quotechar = '"')
		next(prof_csv, None)
		for row in prof_csv:
			count = 0
			for d in row:
				if d != "":
					count += 1
			if count != 4:
				missing_count += 4 - count
				missed += 1
			total += 1
	# print missing_count
	# print missed
	# print missing_count / float(missed)
	print total

def largest_plays():
	max_plays = 0
	max_user, max_artist = "", ""
	largest = []
	total = 0.0
	count = 0.0
	with open(train_file, 'rb') as train_fh:
			train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
			next(train_csv, None)
			for row in train_csv:
				user   = row[0]
				artist = row[1]
				plays  = int(row[2])
				total += plays
				count += 1
				if len(largest) < 1000:
					largest.append((plays, user, artist))
				elif plays > largest[0][0]:
					largest[0] = (plays, user, artist)
				largest.sort()

	#print max_user, max_artist, max_plays
	d = {}
	for l in largest:
		if l[1] in d:
			d[l[1]] += 1
		else:
			d[l[1]] = 1
	print d.values()
	print total / count


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



def country_count():
	countries = Counter()
	with open("profiles.csv", 'rb') as profs_fh:
		prof_csv = csv.reader(profs_fh, delimiter = ',', quotechar = '"')
		next(prof_csv, None)
		for row in prof_csv:
			country = row[3]
			countries.update([country])
	print len(dict(countries).keys())


def play_counts():
	with open(train_file, 'rb') as train_fh:
		train_csv = csv.reader(train_fh, delimiter = ',', quotechar = '"')
		next(train_csv, None)
		u1 = "fa40b43298ba3f8aa52e8e8863faf2e2171e0b5d"
		u1_plays = []
		u2 = "0938eb3d1b449b480c4e2431c457f6ead7063a34"
		u2_plays = []
		for row in train_csv:
			if row[0] == u1:
				u1_plays.append(int(row[2]))
			elif row[0] == u2:
				u2_plays.append(int(row[2]))
		print u1_plays
		print len(u1_plays)
		print u2_plays
		print len(u2_plays)

play_counts()

