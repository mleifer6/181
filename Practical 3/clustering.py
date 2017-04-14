from sklearn.cluster import KMeans
import numpy as np
import csv

X = []
count = 0
examples = {}
with open('artist_features.csv', 'rb') as features:
		features_csv = csv.reader(features, delimiter=',', quotechar='"')
		next(features_csv, None)
		for row in features_csv:
			key = row[0]
			row = map(lambda x: int(x), row[1:61])
			X.append(row)
			examples[key] = np.asarray(row).reshape(1, -1), count
			count += 1


KMC = KMeans(n_clusters=20)
X = np.asarray(X)
print X.shape

KMC.fit(X)


"""
"a3cb23fc-acd3-4ce0-8f36-1e5aa6a18432",U2
"84eac621-1c5a-49a1-9500-555099c6e184",Spoon
"b071f9fa-14b0-4217-8e97-eb41da73f598",The Rolling Stones
"650e7db6-b795-4eb5-a702-5ea2fc46c848",Lady Gaga
"94b0fb9d-a066-4823-b2ec-af1d324bcfcf",The Velvet Underground

"1f9df192-a621-4f54-8850-2c5373b7eac9",Ludwig van Beethoven
"b972f589-fb0e-474e-b64a-803b0364fa75",Wolfgang Amadeus Mozart
"27870d47-bb98-42d1-bf2b-c7e972e6befc",George Frideric Handel
"""

print KMC.predict(examples["1f9df192-a621-4f54-8850-2c5373b7eac9"][0])
print KMC.predict(examples["b972f589-fb0e-474e-b64a-803b0364fa75"][0])
print KMC.predict(examples["27870d47-bb98-42d1-bf2b-c7e972e6befc"][0])
print ""
print KMC.predict(examples["a3cb23fc-acd3-4ce0-8f36-1e5aa6a18432"][0]) # U2
print KMC.predict(examples["94b0fb9d-a066-4823-b2ec-af1d324bcfcf"][0]) # Velvet Underground
print ""
print KMC.predict(examples["84eac621-1c5a-49a1-9500-555099c6e184"][0])
print KMC.predict(examples["b071f9fa-14b0-4217-8e97-eb41da73f598"][0])
print KMC.predict(examples["650e7db6-b795-4eb5-a702-5ea2fc46c848"][0])
print KMC.predict(examples["94b0fb9d-a066-4823-b2ec-af1d324bcfcf"][0])


