
#Import libraries

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time
import datetime

#Read data

df = pd.read_csv('/path/Asteroids_data2.csv')


#Select parts of the datasets and concatenate them 

df_Apollo = df[1:1000]
df_Aten = df[8500:9200]
datasets = [df_Apollo, df_Aten]
df = pd.concat(datasets)

#Remove spaces from columns 

df2 = df.rename(columns = lambda x: x.strip())

#Separate in train and test sets 

df2['is_training'] = np.random.uniform(0, 1, len(df2)) <= .75
train, test = df2[df2['is_training']==True], df2[df2['is_training']==False]

#See the length of train and test sets 

len(train)
len(test)

#Remove spaces from values in columns

test['Type'].str.replace('\s+', '')

#Select features 

features = df2.columns[1:12]

#Convert labels into array of numbers 

y = pd.factorize(y2)[0]

#Call the classifier and fit it in with the data 

clf = RandomForestClassifier(n_jobs=2)
clf.fit(train[features], y)

#Evauate classifier 

clf.predict(test[features])
clf.predict_proba(test[features])[0:10]
test['Type'].head()

#List the importance of features

list(zip(train[features], clf.feature_importances_))



