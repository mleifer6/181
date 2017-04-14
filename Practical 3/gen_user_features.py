import csv
import numpy as np

"""
Calculate the median, mean, min_plays, max_plays, std_dev_of_plays,
total_plays for every user and every artist
"""

users = {}
artists = {}
print "Artist and User play counts are not sorted"
with open("train.csv", 'rb') as train_fh:
	train_csv = csv.reader(train_fh, delimiter = ',', quotechar = '"')
	next(train_csv, None)
	for row in train_csv:
		user = row[0]
		artist = row[1]
		plays = int(row[2])

		if user not in users:
			users[user] = []

		if artist not in artists:
			artists[artist] = []

		users[user].append((int(plays), artist))
		#users[user].sort(reverse = True)

		artists[artist].append((int(plays), user))
		#artists[artist].sort(reverse = True)

print "Adding features to User Profiles"

countries = {}
country_counter = 0
# default for missing country names
countries[""] = -1
with open("extended_profs.csv", 'w') as extended_fh:
	extended_csv = csv.writer(extended_fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	extended_csv.writerow(['user', 'sex','age','country'] + ['median', 'mean', 'min_plays', 'max_plays', 'std_dev_of_plays', 'total_plays'])
	with open("profiles.csv", 'rb') as profs_fh:
		profs_csv = csv.reader(profs_fh, delimiter = ',', quotechar = '"')
		next(profs_csv, None)
		for row in profs_csv:
			user = row[0]
			if row[1] != "":
				if row[1] == "f":
					sex = 1
				else:
					sex = 2
			else:
				sex = 1.5
			if row[2] != "":
				age = int(row[2])
			else:
				# The Average Age of all the Users
				age = 24.5

			country = row[3]
			if not country in countries:
				countries[country] = country_counter
				country_counter += 1

			country = countries[country]

			tmp = np.asarray(users[user])
			#print tmp
			plays = []
			for i in range(len(tmp)):
				plays.append(int(tmp[i][0]))
			plays = np.asarray(plays)
			#print plays
			median = np.median(plays)
			mean = round(np.mean(plays),4)
			min_plays = min(plays)
			max_plays = max(plays)
			std_dev_of_plays = round(np.std(plays), 4)
			total_plays = sum(plays)

			new_features = [user, sex, age, country, median, mean, min_plays, max_plays, std_dev_of_plays, total_plays]
			extended_csv.writerow(new_features)

print "Adding features to Artists"

with open("extended_artists.csv", 'w') as extended_fh:
	extended_csv = csv.writer(extended_fh, delimiter = ',', quotechar = '"')
	extended_csv.writerow(['id','artist','group','person','US','begin'] + ['median', 'mean', 'min_plays', 'max_plays', 'std_dev_of_plays', 'total_plays'])
	with open("artist_data.csv", 'rb') as artists_fh:
		artists_csv = csv.reader(artists_fh, delimiter = ',', quotechar = '"')
		next(artists_csv, None)
		for row in artists_csv:
			artist = row[1]
			tmp = np.asarray(artists[artist])
			#print tmp
			plays = []
			for i in range(len(tmp)):
				plays.append(int(tmp[i][0]))
			plays = np.asarray(plays)
			#print plays
			median = np.median(plays)
			mean = round(np.mean(plays),4)
			min_plays = min(plays)
			max_plays = max(plays)
			std_dev_of_plays = round(np.std(plays), 4)
			total_plays = sum(plays)

			new_features = row + [median, mean, min_plays, max_plays, std_dev_of_plays, total_plays]
			extended_csv.writerow(new_features)
