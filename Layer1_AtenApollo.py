# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import csv
import numpy 
import pandas as pd

Ast = pd.read_csv('.../Classification_Aten_Apollo Dataset.csv') #Path
Ast.head(10)

#TODO : Transform datetime // regular expresion or remove column

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf 


def main(_):
    #Import data 
    Ast = pd.read_csv('/home/gparreno/DataScience/Deep_Asteroid/Classification_Aten_Apollo Dataset.csv')
    
    #Create the model
    x = tf.placeholder(tf.float32, [None, 7])
    W = tf.Variable(tf.zeros([7, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.matmul(x,W) + b
    
    #Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run
    
    ## Train
    #for _ in range(100):
        #batch_xs, batch_ys = Ast[1:10]
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
    #Test trained model 
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.int32))
    print(sess.run(accuracy, feed_dict={x: Ast[1:20],
                                        y_: Ast[20:40]}))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/Deep_Asteroid',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
