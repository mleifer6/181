import numpy as np
import csv
import musicbrainzngs as mb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import itertools
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


users = {}
print "Reading User Profile Data..."
with open("extended_profs.csv", 'rb') as profs_fh:
	profs_csv = csv.reader(profs_fh, delimiter = ',', quotechar = '"')
	next(profs_csv, None)
	for row in profs_csv:
		user = row[0]
		users[user] = map(float, row[1:])
		#print users[user]		

print "Reading Artist Data"
artists = {}
with open("extended_artists.csv") as artists_fh:
	artists_csv = csv.reader(artists_fh, delimiter = ',', quotechar = '"')
	next(artists_csv, None)
	for row in artists_csv:
		artist = row[1]
		artists[artist] = map(float, row[2:])


print "Reading Train Data..."
X_train = []
Y_train = []
num_examples = 50000
print "Using %d data points" % num_examples
with open("train.csv", 'r') as train_fh:
	train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
	next(train_csv, None)
	for row in itertools.islice(train_csv, 0, num_examples):
		user = row[0]
		artist = row[1]
		plays = int(row[2])
		X_train.append(users[user] + artists[artist])
		Y_train.append(plays)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_training, X_validating, Y_training, Y_validating = train_test_split(
	X_train, Y_train, test_size=0.33, random_state=42)

print("learning...")
model = RandomForestRegressor(n_estimators=20)
#model =  MLPRegressor()
#model = GradientBoostingRegressor()
model.fit(X_training, Y_training)
print("done learning")
print("learning score: ", cross_val_score(model, X_training, Y_training, cv=5))
print("mean absolute error: ", mean_absolute_error(Y_training, model.predict(X_training)))

print("predicting...")
preds = model.predict(X_validating)
print("mean absolute error: ", mean_absolute_error(Y_validating, preds))


soln_file = "few_features_results.csv"
with open("test.csv", 'r') as test_fh:
	test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
	next(test_csv, None)

	with open(soln_file, 'w') as soln_fh:
		soln_csv = csv.writer(soln_fh,
							  delimiter=',',
							  quotechar='"',
							  quoting=csv.QUOTE_MINIMAL)
		soln_csv.writerow(['Id', 'plays'])

		for row in test_csv:
			id     = row[0]
			user   = row[1]
			artist = row[2]

			data = np.asarray(users[user] + artists[artist]).reshape(1,-1) #
			predicted_plays = round(model.predict(data)[0],4)
			if predicted_plays < 0:
				predicted_plays = 1 #users[user][3]
			soln_csv.writerow([id, predicted_plays])


