import csv
import numpy as np
from sklearn.model_selection import train_test_split
def get_medians(train_file):
	user_plays = {}
	artist_plays = {}
	triples = []
	plays = []
	with open(train_file, 'rb') as train_fh:
		train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
		next(train_csv, None)
		for row in train_csv:
			user = row[0]
			artist = row[1]
			plays = int(row[2])

			if user not in user_plays:
				user_plays[user] = []
			user_plays[user].append(plays)

			if artist not in artist_plays:
				artist_plays[artist] = []
			artist_plays[artist].append(plays)

			triples.append((user, artist, plays))


	artist_medians = {}
	for artist in artist_plays:
		artist_medians[artist] = np.median(artist_plays[artist])

	user_medians = {}
	for user in user_plays:
		user_medians[user] = np.median(user_plays[user])

	X = []
	Y = []
	for user, artist, play in triples:
		X.append((user_medians[user], artist_medians[artist]))
		Y.append(play)

	global_median = np.median(Y)

	return X, Y, global_median

X, Y, global_median = get_medians("train.csv")

predictions = []

global_median = float(global_median)

exp = 0.22

for user_med, artist_med in X:
	predictions.append(user_med * (artist_med / global_median) ** exp)

n = float(len(Y))
error = 0
for i in range(n):
	error += 1.0 / n * abs(Y[i] - predictions[i])

print error

with open('simple_predictions_0.22.csv') as sp_fh:
	sp_csv = csv.writer(sp_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL))
        soln_csv.writerow(['Id', 'plays'])

