
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


# In[2]:

"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("train.csv")
print "checkpoint 1"
df_test = pd.read_csv("test.csv")
print "checkpoint 2"
df_new_feats = pd.read_csv("new_features.csv")
print "checkpoint 3"


# In[3]:

print df_train.head()


# In[4]:

print df_test.head()


# In[5]:

print df_new_feats.head()


# In[6]:

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)
print "checkpoint 4"


# In[7]:

#delete 'smiles' column from new feature
#df_new_feats = df_new_feats.drop(['smiles'], axis=1)
#delete 'Id' column
#df_new_feats = df_new_feats.drop(['Id'], axis=1)
print "checkpoint 5"


# In[8]:

print df_new_feats.head()
print df_new_feats.tail()


# In[ ]:

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
print "checkpoint 6"
df_all = pd.merge(df_all, df_new_feats, on = 'smiles')
print "checkpoint 7"

print df_all.head()


# In[ ]:

print df_all.tail()


# In[9]:

#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
print "checkpoint 8"
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print "Train features:", X_train.shape
print "Train gap:", Y_train.shape
print "Test features:", X_test.shape


# In[10]:

LR = LinearRegression()
LR.fit(X_train, Y_train)
LR_pred = LR.predict(X_test)


# In[11]:

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)


# In[12]:

print LR_pred
print RF_pred


# In[13]:

tests = 6
# l1 penalty
A = [float(10**i) for i in range(tests)]
#l2 penalty
B = [float(10**i) for i in range(tests)]
all_errors = {i :{} for i in range(tests)}


# In[26]:

# # Try different penalty values
# for i in range(tests):
#     for j in range(tests):
#         EN = ElasticNet(alpha = 20.0, l1_ratio = 10.0)
#         print "checkpoint 1"
#         #A[i] + B[j] and A[i] /(A[i] + B[j])
#         EN.fit(X_train, Y_train)
#         #print "checkpoint 2"
#         errors =  cross_val_score(EN, X_train, Y_train, cv=2, scoring="neg_mean_squared_error")
#         print errors
# #         all_errors[i][j] = errors
#         #print(A[i] + B[j])
# print all_errors


# In[29]:

print "RIDGE ERRORS:"
clf = Ridge(alpha=10.0)
clf.fit(X_train, Y_train)
errors =  cross_val_score(clf, X_train, Y_train, cv=5, scoring="neg_mean_squared_error")
print errors


# In[31]:

# for i in range(1, 10):
#     clf = Ridge(alpha = 0.1 ** i)
#     clf.fit(X_train, Y_train)
#     errors =  cross_val_score(clf, X_train, Y_train, cv=2, scoring="neg_mean_squared_error")
#     print errors


# In[ ]:

def avg_rmse(lst):
    total = 0
    for mse in lst:
        mse = abs(mse) ** 0.5
        total += mse
    return mse / len(lst)


# In[ ]:

# import sys
# min_rmse = sys.maxsize
# A,B = -1,-1
# for i in range(tests):
#     for j in range(tests):
#         e = avg_rmse(all_errors[i][j])
#         if e < min_rmse:
#             min_rmse = e
#             A,B = i,j

# print min_rmse, i, j


def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


# write_to_file("sample1.csv", LR_pred)
# write_to_file("sample2.csv", RF_pred)

