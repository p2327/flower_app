'''
Rendering the TensorFlow Iris tutorial as a Streamlit app.
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from utils import pack_features_vector, loss, grad

import os
import matplotlib.pyplot as plt
import tensorflow as tf

import streamlit as st
import pandas as pd


st.title("TensorFlow Tutorial")

st.write(f"TensorFlow version: {tf.__version__}\n\
Eager execution: {tf.executing_eagerly()}")

st.markdown("""The Iris genus entails about 300 species,
but our program will only classify the following three:
 - *Iris setosa*
 - *Iris virginica*
 - *Iris versicolor*  

Load train and test set and look at the first rows of data with pandas:
""")
st.code('''
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url),
    origin=train_dataset_url
    )
test_fp = tf.keras.utils.get_file(
    fname=os.path.basename(test_url),
    origin=test_url
    )

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
train_df.head()
''')

# load train and test data
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url),
    origin=train_dataset_url
    )
test_fp = tf.keras.utils.get_file(
    fname=os.path.basename(test_url),
    origin=test_url
    )

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
st.markdown("""
Each label is associated with a string name (for example, *setosa*),
but machine learning typically relies on numeric values
(**0, 1, 2** in this case)
""")

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
# number of examples in a 'batch' (one iteration of model training)
batch_size = 32
feature_names = column_names[:-1]
label_name = column_names[-1]


'''
---
### Prepare the data for TensorFlow
Calling ```make_csv_dataset``` to create a **tf.data.Dataset**
'''

# prepare dataset for tf
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
    )

st.markdown("The ```make_csv_dataset``` function returns a **tf.data.Dataset** of (features, label) pairs, \
    where ```features``` is a dictionary: {'feature_name': value}")
# no need to call st.write() with Python3 to print
train_dataset


'''
```Dataset``` objects are iterable.
Like-features are grouped together or *batched* and each example row fields are appended to the corresponding feature array.  
Changing ```batch_size``` in ```make_csv_dataset``` sets the number of examples
stored in these feature arrays.
'''
st.code('''
features, labels = next(iter(train_dataset))
features
''')
# look at the next item in the Dataset object
features, labels = next(iter(train_dataset))
features


'''
Explore data with a scatter plot. Some clusters are visible by
plotting a few features from the batch.
'''
st.code('''
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()
''')

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
# plt.show()
st.pyplot()


'''
Pack the features in a single arrays with shape ```(batch_size, num_features)``` 
using ```pack_features_vector```
'''
st.code('''
def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels
    ''')


'''
Use ```tf.data.Dataset.map``` to pack the features of each
```(features, label)```pair into the training dataset
(it applies the function to each pair).

The features element of the ```Dataset``` are now arrays with shape
```(batch_size, num_features)```.
'''
st.code('''
train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))
# the features values for the next batch of examples in train_dataset
# let's look at the first 5 examples (i.e. samples)
features[:5]
# the target labels for each example in the batch
labels[:5]
''')
train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))
# let's look at the first 5 examples (i.e. samples)
features[:5]
# the target labels for each example in the batch
labels[:5]


st.markdown('''
---
### Select the model

A model is a relationship between the features and the label.
<p align="justify">
We'll use a neural network to solve the Iris classification problem.   
Neural networks can find complex relationships between features and the label. 
<p align="justify">
It is a highly-structured graph, organized into one or more hidden layers. 
Each hidden layer consists of one or more neurons. 
There are several categories of neural networks and this program uses a dense, 
or fully-connected neural network: the neurons in one layer receive
input connections from every neuron in the previous layer.  

When the model from is trained and fed an unlabeled example, 
it yields three predictions: the likelihood that this flower is 
the given Iris species. This prediction is called **inference**.</p> 
''', unsafe_allow_html=True)
image_url = 'https://www.tensorflow.org/images/custom_estimators/full_network.png'
st.image(image_url, use_column_width=True)


'''
---
### Create a model using Keras

The ```tf.keras.Sequential``` model is a linear stack of layers.  
Its constructor takes a list of layer instances, in this case, two Dense layers with 10 nodes each,
and an output layer with 3 nodes representing our label predictions.  

The first layer's ```input_shape``` parameter corresponds to the number of features from the dataset, 
and is required.
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
> **Rule of thumb**. Increasing the number of hidden layers and neurons creates a more powerful model, 
> which requires more data to train effectively.
'''

'''
#### Using the model
Passing ```features``` in the model calculates predictions for each
example in the batch (in our case we have 32 examples per batch).  
For each example we obtain a logit for each class in the target label.
'''
st.code('''
predictions = model(features)
# take the first 5 examples
predictions[:5]
''')
predictions = model(features)
predictions

'''
Using ```softmax``` converts the logit to the predicted probability of 
each example in the batch to be of one of the classes.
'''
st.code('''
# take the first 5 examples
tf.nn.softmax(predictions[:5])
''')
tf.nn.softmax(predictions[:5])

'''
Calling ```tf.argmax``` on ```predictions``` gives the predicted class index.
As the model has not been trained yet, these won't be very good just now.
'''
st.write(f" Predictions: {tf.argmax(predictions, axis=1)}")
st.write(f"Actual label: {labels}")


'''
---
### Train the model
#### Define the loss and gradient function
Both training and evaluation stages need to calculate the model's loss.
This measures how off a model's predictions are from the desired label, or 
in other words, how bad the model is performing.
We want to minimize, or optimize, this value.

Our model will calculate its loss using the
```tf.keras.losses.SparseCategoricalCrossentropy``` function
which takes the model's class probability predictions and the desired label,
and returns the average loss across the examples.
'''
st.code('''
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)
  y_ = model(x)
  return loss_object(y_true=y, y_pred=y_)

crossentropy_loss = loss(model, features, labels)
print(f"Loss test: {crossentropy_loss:.2f}")
''')

crossentropy_loss = loss(model, features, labels)
st.write(f"Loss test:  {crossentropy_loss:.2f}")


'''
```tf.GradientTape``` calculates gradients used to optimise the model.
'''
st.code('''
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
''')


'''
#### Create an optimiser

An *optimizer* applies the computed gradients to the model's variables
to minimize the ```loss``` function.

> #### Gradient descent.
The loss function can be thinked of as a curved surface.
We want to find its lowest point by walking around.

TensorFlow ```tf.train.GradientDescentOptimizer```
implements the stochastic gradient descent (SGD) algorithm.
'''
st.code('''
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
''')

'''
The *hyperparameter* ```learning_rate``` sets the step size to take
for each iteration down the hill.
'''
# image_url2 = 'https://cs231n.github.io/assets/nn3/opt1.gif'
# st.image(image_url2, use_column_width=True)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

'''
Use ```grad``` to calculate gradients and loss value.  By iteratively
calculating the loss and gradient for each batch, we'll adjust the model
during training.  
Gradually, the model will find the best combination of weights and bias
to minimize loss.
'''
st.code('''
loss_value, gradients = grad(model, features, labels)

# print the initial loss, no optimisation
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))
# calculate a single optimisation step
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels
                                          ).numpy()))
''')
loss_value, gradients = grad(model, features, labels)

st.write(f"Step: {optimizer.iterations.numpy()} \
           \nInitial Loss: {loss_value.numpy()}")

optimizer.apply_gradients(zip(gradients, model.trainable_variables))

st.write(f"Step:  {optimizer.iterations.numpy()} \
           \nLoss: {loss(model, features, labels).numpy()}")


'''
#### Training loop
The model is ready for training.  
A *training loop* feeds the dataset examples into the ```model```
to help it make better predictions.  

The following code sets up these *training steps*:
1. Iterate each **epoch**. An epoch is one pass through the dataset.
2. Within an epoch, iterate over each example in the training ```Dataset```
grabbing its ```features``` and (actual)```label```.
3. Using the example features, make a prediction and compare it
with the label. Measure the inaccuracy and use that to calculate the gradients
(via ```grad``` internal call to ```loss```).
4. Use an ```optimizer``` to update the model variables.
5. Keep track of stats for later visualization.

```num_epochs``` is the number of times to loop over the dataset.
> Counter-intuitively, training a model longer does not
guarantee a better model.
'''
st.code('''
# Note: Rerunning this cell uses the same model variables
# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    epoch_accuracy(y, model(x))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  # print results at set number of epochs
  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
''')
# Note: Rerunning this cell uses the same model variables
# Keep results for plotting
train_loss_results = []
train_accuracy_results = []
num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        # for each batch
        # Add current batch loss 
        epoch_loss_avg(loss_value)
        # Compare predicted label to actual label
        epoch_accuracy(y, model(x))

    # At end epoch append average loss and epoch accuracy
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # print results at set number of epochs
    if epoch % 50 == 0:
        st.write(f"Epoch: {epoch:03d} \
            Loss: {epoch_loss_avg.result():.3f} \
            Accuracy: {epoch_accuracy.result():.3%}")


'''
#### Visualise the loss function over time
'''
st.code('''
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
''')
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
st.pyplot()


'''
---
### Evaluate the model on the test set
Load the test data in a a ```tf.dataset```:
'''
st.code('''
test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)
''')
test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)


'''
The model evaluates only a single epoch of test data.
'''
st.code('''
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
''')
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

st.write(f"Test set accuracy: {test_accuracy.result():.3%}")


'''
#### Use the model to make predictions
The trained model can be used to make predictions on *unlabeled examples*,
i.e examples that contain a feature but no label.
'''
st.code('''
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5],
    [5.9, 3.0, 4.2, 1.5],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
''')
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5],
    [5.9, 3.0, 4.2, 1.5],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    st.write(f"Example {i} prediction: {name} ({100*p:4.1f}%)")
