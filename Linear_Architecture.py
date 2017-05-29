import tempfile
import urllib
train_file = '/home/gparreno/DataScience/Deep_Asteroid/testing_dataset.csv'
test_file = '/home/gparreno/DataScience/Deep_Asteroid/testing_dataset2.csv'
##MAybe need to rename the variables for something easier or just load it in pandas
import pandas as pd
COLUMNS = ["EMoid", "Epoch", "H", "q-Perihelion", "Q-Aphelion",
           "Name", "Class"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

#Binary classification problem, label whose value is 1 if the classification is Aten, 0 if it is Apollo
LABEL_COLUMN = "Class"
df_train[LABEL_COLUMN] = (df_train["Class"].apply(lambda x: "Aten" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["Class"].apply(lambda  x: "Aten" in x)).astype(int)


#define Categorical and continuous
CATEGORICAL_COLUMNS = ["Class"]
CONTINUOUS_COLUMNS = ["EMoid","Epoch","H","q-Perhelion","Q-Aphelion"]






import tensorflow as tf
def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

classifica = tf.contrib.layers.sparse_column_with_keys(
  column_name="class", keys=["Aten", "Apolo"])

EMoid = tf.contrib.layers.real_valued_column("EMoid")
Epoch = tf.contrib.layers.real_valued_column("Epoch")
H = tf.contrib.layers.real_valued_column("H")
qPerhelion = tf.contrib.layers.real_valued_column("q-Perhelion")
QAphelion = tf.contrib.layers.real_valued_column("a-Aphelion")

#Intersecting Multiple Columns with CrossedColumn
qPerhelion_x_QAphelion = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))

#Defining the Logistic Regression Model
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
    EMoid, Epoch, H, qPerhelion, QAphelion, Name, classifica
    qPerhelion_x_QAphelion],
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=1.0,
        l2_regularization_strength=1.0),
    model_dir=model_dir)

#Training and Evaluating our model
m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])
