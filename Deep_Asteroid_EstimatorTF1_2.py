from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import numpy as np


#Headers
names = [
    'Emoid',
    'Epoch',
    'H',
    'q',
    'Q',
    'Class',
    'Discovery date',
]

dtypes = {
    'Emoid': np.float32,
    'Epoch': datetime?,
    'H': np.float32,
    'q': np.float32,
    'Q': np.float32,
    'Class': str,
    'Discovery date': datetime,
}

#Read file
df = pd.read_csv('dataset.csv', names=names, dtype=dtypes, na_values='?') #Path

#Fill missing values in continuous columns with zeros instead of NaN
float_columns = [i for i,x in dtypes.items() if x == np.float32]
df[float_columns] = df[float_columns].fillna(value=0., axis='columns')
string_columns = [i for i,x in dtypes.items() if x == str]
df[string_columns] = df[string_columns].fillna(value='', axis='columns')

#Split the data into a training set and a eval set
training_data = df[:80]
eval_data = df[80:]

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
