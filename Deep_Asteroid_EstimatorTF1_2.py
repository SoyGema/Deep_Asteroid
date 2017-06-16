from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import numpy as np


#Headers
names = [
    'Designation(name)',
    'Prov. Desc.', #provisional designation
    'q', #Perihelion distance
    'Q', #Aphelion distance   
    'Emoid', #minium distance between earth orbit and minor planet 
    'H', #absolute visual magnitude   
    'Epoch',
    'M',
    'Peri', #1 Argument of perihelion
    'Node', #1 longitude of the ascending node
    'Incl', #1 inclination
    'e', #orbital eccentricity
    'a', #Semimajor axis 
    'Opps', #number of oppositions 
    'Ref',
    'Designation',
    'Discovery date',
    'site',
    'discoverers',
    'Class', #Aten or Apollo. Classification 
]

dtypes = {
        'Designation(name)': dict,
    'Prov.Desc.': str,
    'q': np.float32,
    'Q': np.float32,    
    'Emoid': np.float32,
    'H': np.float32,    
    'Epoch' : np.int32,
    'M': np.float32,
    'Peri': np.float32,
    'Node': np.float32,
    'Incl': np.float32,
    'e': np.float32,
    'a': np.float32,
    'Opps': np.int32,
    'Ref': str,
    'Designation': str,
    'Discovery date': str,
    'site': str,
    'discoverers':str,
    'Class': str,
    'Epoch': datetime?,

    'Discovery date': datetime,
}

#Read file
df = pd.read_csv('dataset.csv', names=names, dtype=dtypes, na_values='?') #Path

#Fill missing values in continuous columns with zeros instead of NaN
float_columns = [i for i,x in dtypes.items() if x == np.float32]
df[float_columns] = df[float_columns].fillna(value=0., axis='columns')
string_columns = [i for i,x in dtypes.items() if x == str]
df[string_columns] = df[string_columns].fillna(value='', axis='columns')

#Split the data into a training set and a eval set in a 90 / 10 split : Total dataset 9290 entries
training_data = df[:8361]
eval_data = df[8361:]

#Separate input features from labels
training_label = training_data.pop('Class')
eval_label = eval_data.pop('Class')

import tensorflow as tf
#Make input function for training
# num_epochs --> cycles
# shuffle= True  --> randomize order of input data

training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_data, y= training_label, batch_size=64, shuffle=True, num_epochs=None)

# Make input function for evaluation:
# shuffle=False -> do not randomize input data 

eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=eval_data, y=eval_label, batch_size= 64, suffle=False)

# Describe how the model should interpret the inputs. The names of the feature columns have to match
# the names of the series in the dataframe

Emoid = tf.feature_column.numeric_column('Emoid')
.....
Class = tf.feature_column.categorical_column_with_vocabulary_list('Class', ['Aten', 'Apollo'])

#Make a list of the linear features
linear_features = [q, Q, EMoid, H, Epoch, M,
                   Peri, Node, Incl, e, a, Opps]

#Set the regressor
regressor = tf.contrib.learn.LinearRegressor(feature_columns=linear_features)

#Set the fit of the regressor 
regressor.fit(input_fn=training_input_fn, steps=10000)

regressor.evaluate(input_fn=eval_input_fn)
