import rdkit
import csv
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sys

FEATURES = 1024
myfile = open("new_features_" + str(FEATURES) + ".csv", 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
header = ["feat_" + str(i) for i in range(256,256 + FEATURES + 1)]
header[0] = "smiles"
wr.writerow(header)

print "here"
df = pd.read_csv("train.csv")
molecules = df.smiles
print molecules.shape

count = 0
for molecule in molecules:
	if count % 10000 == 0:
		sys.stdout.write(str(count) + "\n")
	m1 = Chem.MolFromSmiles(molecule)
	bits = AllChem.GetMorganFingerprintAsBitVect(m1,2,nBits=FEATURES)
	lst = [molecule]
	for i in bits:
		lst.append(i)
	wr.writerow(lst)
	count += 1

df = pd.read_csv("test.csv")
molecules = df.smiles
print molecules.shape

for molecule in molecules:
	if count % 10000 == 0:
		sys.stdout.write(str(count) + "\n")
	m1 = Chem.MolFromSmiles(molecule)
	bits = AllChem.GetMorganFingerprintAsBitVect(m1,2,nBits=FEATURES)
	lst = [molecule]
	for i in bits:
		lst.append(i)
	wr.writerow(lst)
	count += 1
