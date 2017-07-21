
#Import libraries

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time
import datetime

#Read data

df = pd.read_csv('/home/gparreno/Deep-Asteroid/V3/Asteroids_data2.csv')


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
