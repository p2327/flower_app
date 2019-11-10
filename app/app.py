'''
Deploy a classifier as a service for front-end consumption.

The application contains 3 parts:
    1. Show a random Iris flower
    2. Allow users to challenge tha AI
    3. Show the current scoreboard

The plan is to write a data pipelaine that:
    - Gathers and validates Iris flower data
    - Trains and validates a model capable of classifying Iris flowers
    - Deploys that model as a service for front-end consumption
'''
#from utils_app import show_random_iris
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import streamlit as st
import json

column_names = ['sepal_length',
                'sepal_width',
                'petal_length',
                'petal_width',
                'species']

test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"


test_fp = tf.keras.utils.get_file(
    fname=os.path.basename(test_url),
    origin=test_url
    )

# load data 'flower_app/data/iris_test.csv'
iris_test_data = pd.read_csv(test_fp,
                             skiprows=[0],
                             names=column_names
                             )
# encode 0, 1, 2 label classes to string names
# np.where(condition, [x, y])
iris_test_data['species'] = np.where(iris_test_data['species'] == 0,
                                     'setosa',
                                     np.where(iris_test_data['species'] == 1,
                                              'versicolor',
                                              'virginica'
                                              ))

st.write(iris_test_data.head())


def show_random_iris():
    choices = ['setosa', 'versicolor', 'virginica']
    random_flower = random.choice(choices)
    flowers = {
        'setosa': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/440px-Kosaciec_szczecinkowaty_Iris_setosa.jpg',
        'versicolor': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/440px-Iris_versicolor_3.jpg',
        'virginica': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/440px-Iris_virginica.jpg'
    }
    random_url = flowers[random_flower]
    single_type = iris_test_data[(iris_test_data['species'] == random_flower)]
    random_row = single_type.sample(n=1).drop(['species'], axis=1)
    final = random_flower, random_url, random_row
    return(final)


# GET /get_random_flower
random_result = show_random_iris()
res = {
    "name" : random_result[0],
    "url" : random_result[1],
    "data" : {"keys" : list(random_result[2]), "values" : random_result[2].iloc[0].tolist()}
}

st.write(json.dumps(res))