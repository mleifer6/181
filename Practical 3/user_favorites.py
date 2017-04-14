import csv

def artists_to_nums():
	artist_count = 0
	artists = {}
	with open("artists.csv", 'rb') as artists_fh:
		artists_csv = csv.reader(artists_fh, delimiter = ',', quotechar = '"')
		next(artists_csv, None)
		for row in artists_csv:
			artists[row[0]] = artist_count
			artist_count += 1
	return artists, artist_count

def favorite_artists(n):
	artists, artist_count = artists_to_nums()
	users = {}
	with open("train.csv", 'rb') as train_fh:
		train_csv = csv.reader(train_fh, delimiter = ',', quotechar = '"')
		next(train_csv, None)
		for row in train_csv:
			user = row[0]
			artist = row[1]
			plays = int(row[2])

			if user not in users:
				users[user] = []

			users[user].append((plays, artist))
			users[user].sort(reverse = True)
			
	with open('user_favorites.csv', 'w') as fav_file:
		fav_csv = csv.writer(fav_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		fav_csv.writerow(['user', 'sex','age','country'] + [i for i in range(artist_count)])
		with open("profiles.csv", 'rb') as prof_fh:
			prof_csv = csv.reader(prof_fh, delimiter = ',', quotechar = '"')
			next(prof_csv, None)
			for row in prof_csv:
				favorites = [0 for i in range(artist_count)]
				user = row[0]
				for j in range(n):
					if len(users[user]) > j:
						artist_name = users[user][j][1]
						artist_num = artists[artist_name]
						favorites[artist_num] = 1
				fav_csv.writerow(row + favorites)


favorite_artists(10)
