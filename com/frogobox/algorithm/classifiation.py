#
# Created by Faisal Amir
# FrogoBox Inc License
# -----------------------------------------
# Mini-ML-Air-BnB
# Copyright (C) 08/04/2020.      
# All rights reserved
# -----------------------------------------
# Name     : Muhammad Faisal Amir
# E-mail   : faisalamircs@gmail.com
# Github   : github.com/amirisback
# LinkedIn : linkedin.com/in/faisalamircs
# -----------------------------------------
# FrogoBox Software Industries
# 
# /

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from com.frogobox.algorithm.evaluation import evaluation
from com.frogobox.base.config import *


def create_result_classification(data_test, y_prediction):
    item_column_0 = []
    item_column_1 = []
    item_column_2 = []
    item_column_3 = []

    for column in data_test:
        item_column_0.append(column[0])
        item_column_1.append(column[1])
        item_column_2.append(column[2])
        item_column_3.append(column[3])

    data_frame = pd.DataFrame(
        {COLUMN_PRICE: item_column_0, COLUMN_MINIMUM_NIGHTS: item_column_1,
         COLUMN_NUMBER_OF_REVIEWS: item_column_2,
         COLUMN_AVAILABILITY_365: item_column_3, COLUMN_CLASS: y_prediction})

    data_frame.to_csv(FILE_NAME_RESULT_CLASSIFICATION, index=False)


def classification(path_data_result_clustering):
    dataset = pd.read_csv(path_data_result_clustering)
    # Labeling data dalam beberapa set
    x_dataset = dataset.iloc[:, :-1].values
    y_dataset = dataset.iloc[:, 4].values

    x_training, x_test, y_training, y_test = train_test_split(x_dataset, y_dataset, test_size=0.30)

    # Feature Scaling
    scaler = StandardScaler()
    scaler.fit(x_training)

    x_training = scaler.transform(x_training)
    temp_test = x_test
    x_test = scaler.transform(x_test)

    # Melatih dan Prekdiksi
    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(x_training, y_training)
    y_prediction = classifier.predict(x_test)

    create_result_classification(temp_test, y_prediction)

    evaluation(y_test, y_prediction)
