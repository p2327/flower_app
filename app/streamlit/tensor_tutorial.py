from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import tensorflow as tf

import streamlit as st
import pandas as pd
import time
import numpy as np

st.title("TensorFlow Tutorial")

st.write(f"TensorFlow version: {tf.__version__}")
st.write(f"Eager execution: {tf.executing_eagerly()}")

st.markdown("""The Iris genus entails about 300 species, 
but our program will only classify the following three:
 - *Iris setosa*
 - *Iris virginica*
 - *Iris versicolor*""")


train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))
#head -n5 {'C:\Users\pietr\.keras\datasets\iris_training.csv'}

# look at the first 5 lines of the dataset in terminal
with open(train_dataset_fp) as f:
    for x in range(5):
        head = next(f).strip()
        print(head)


# columns order in the csv file
column_names = ['sepal_length', 
                 'sepal_width', 
                 'petal_length', 
                 'petal_width', 
                 'species']

# using pandas
train_df = pd.read_csv(train_dataset_fp, 
                       names=column_names,
                       header=0)
st.write(train_df.head())

# no need to use st.markdown with python3
st.markdown("""Each label is associated with a string name (for example, *setosa*), 
but machine learning typically relies on numeric values (**0, 1, 2** in this case)""")

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
# number of examples in a batch (one iteration of model training)
batch_size = 32
feature_names = column_names[:-1]
label_name = column_names[-1]


'''
---
### Prepare the data for TensorFlow
Calling ```make_cv_dataset``` to create a **tf.data.Dataset**
'''

# prepare dataset for tf
train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_fp,
                                                      batch_size,
                                                      column_names=column_names,
                                                      label_name=label_name,
                                                      num_epochs=1)

# no need to call st.write() with Python3
train_dataset                                                


st.markdown("The ```make_csv_dataset``` function returns a **tf.data.Dataset** of (features, label) pairs, \
    where features is a dictionary: {'feature_name': value}")

features, labels = next(iter(train_dataset))
features


'''
Explore data with a scatter plot:
'''

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
#plt.show()
st.pyplot()


''' 
Pack the features in a single arrays with ```pack_features_vector```
'''

st.code('''def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels''')

# takes values from a list of tensors and creates a combined tensor at the specified dimension
def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

# pack the features of each (features,label) pair into the training datase
train_dataset = train_dataset.map(pack_features_vector)

# The features element of the Dataset are now arrays with shape (batch_size, num_features)
features, labels = next(iter(train_dataset))
features[:5]


'''
---
### Create a model using Keras
'''

st.code('''model = tf.keras.Sequential([
  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), 
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])''')

model = tf.keras.Sequential([
  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])


'''
The activation function determines the output shape of each node in the layer. 
The ideal number of hidden layers and neurons depends on the problem and the dataset.
> **Rule of thumb**: increasing the number of hidden layers and neurons creates a more powerful model, 
> which requires more data to train effectively.
'''


'''
---
### Train the model                
**Define the loss and gradient function**    
Both training and evaluation stages need to calculate the model's loss. 
This measures how off a model's predictions are from the desired label, in other words, how bad the model is performing. 
We want to minimize, or optimize, this value.    

Our model will calculate its loss using the ```tf.keras.losses.SparseCategoricalCrossentropy``` function 
which takes the model's class probability predictions and the desired label, 
and returns the average loss across the examples.
'''

st.code('''loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
  y_ = model(x)
  return loss_object(y_true=y, y_pred=y_)

crossentropy_loss = loss(model, features, labels)
print(f"Loss test: {crossentropy_loss:.2f}")
''')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
  y_ = model(x)
  return loss_object(y_true=y, y_pred=y_)

crossentropy_loss = loss(model, features, labels)
#print(f"Loss test: {crossentropy_loss}")
#st.write("Loss test: %.2f" % crossentropy_loss)
st.write(f"Loss test:  {crossentropy_loss:.2f}")



























#st.button("Re-run")