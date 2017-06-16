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
df = pd.read_csv('All.csv', names=names, dtype=dtypes, na_values='?') #Path

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


Designation(name) = 
Prov. Desc =
q = tf.feature_column.numeric_column('q')
Q = tf.feature_column.numeric_column('Q')
Emoid = tf.feature_column.numeric_column('Emoid')
H = tf.feature_column.numeric_column('H')
Epoch = 
M = tf.feature_column.numeric_column('M')
Peri = tf.feature_column.numeric_column('Peri')
Node = tf.feature_column.numeric_column('Node')
Incl = tf.feature_column.numeric_column('Incl')
e = tf.feature_column.numeric_column('e')
a = tf.feature_column.numeric_column('a')
Opps = 
Ref = tf.feature_column.categorical_column_with_vocabulary_list('Ref', ['MPCs', 'MPOs'])
Designation = 
Discovery Date = 
Site = tf.feature_column.categorical_column_with_vocabulary_file(key='Site', vocabulary_file = '/us/site.txt', vocabulary_size = 50, num_oov_buckets=5)
Disoverers = tf.feature_column.categorical_column_with_vocabulary_file(key='discoverers', vocabulary_file = '/us/discovers.txt', vocabulary_size = 50, num_oov_buckets=5)

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

dnn_features = [
    #numerical features
    q, Q, EMoid, H, Epoch, M, Peri, Node,
    Incl, e, a, Opps,    
    # densify categorical features:
    tf.feature_column.indicator_column(q),
    tf.feature_column.indicator_column(Q),
    tf.feature_column.indicator_column(Emoid),
    tf.feature_column.indicator_column(e),
    tf.feature_column.indicator_column(a),
]

dnnregressor = tf.contrib.learn.DNNRegressor(feature_columns=dnn_features, hidden_units=[50, 30, 10])
dnnregressor.fit(input_fn=training_input_fn, steps=10000)

dnnregressor.evaluate(input_fn=eval_input_fn)



def experiment_fn(run_config, params):
  # This function makes an Experiment, containing an Estimator and inputs for training and evaluation.
  # You can use params and config here to customize the Estimator depending on the cluster or to use
  # hyperparameter tuning.

  # Collect information for training
  return tf.contrib.learn.Experiment(estimator=tf.contrib.learn.LinearRegressor(
                                     feature_columns=linear_features, config=run_config),
                                     train_input_fn=training_input_fn,
                                     train_steps=10000,
                                     eval_input_fn=eval_input_fn)


import shutil
shutil.rmtree("/tmp/output_dir", ignore_errors=True)
tf.contrib.learn.learn_runner.run(experiment_fn, run_config=tf.contrib.learn.RunConfig(model_dir="/tmp/output_dir"))
