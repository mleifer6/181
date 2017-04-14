import csv

genres = {}
artist_genres = {}
genre_count = 0
with open("genres.csv", 'rb') as genres_fh:
	genres_csv = csv.reader(genres_fh, delimiter = ',', quotechar = '"')
	next(genres_csv, None)
	for row in genres_csv:
		artist = row[0]
		genre = row[1]
		if genre not in genres:
			genres[genre] = genre_count
			print genre_count, genre
			genre_count += 1
		artist_genres[artist] = genres[genre]

print genres

with open("temp_extended_artists.csv", 'wb') as tmp_fh:
	tmp_csv = csv.writer(tmp_fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	tmp_csv.writerow(['id','artist','group','person','US','begin','median', 'mean', 'min_plays', 'max_plays', 'std_dev_of_plays', 'total_plays', 'genre'])
	with open("extended_artists.csv", 'rb') as artists_fh:
		artists_csv = csv.reader(artists_fh, delimiter = ',', quotechar = '"')
		next(artists_csv, None)
		for row in artists_csv:
			artist = row[1]
			genre = artist_genres[artist]
			tmp_csv.writerow(row + [genre])
 