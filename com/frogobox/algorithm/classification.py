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
from com.frogobox.base.helper import cast_to_int, print_border_line


# Output hasil di console
def classification_print(data_frame, save_path_file):
    print_border_line()
    print("Result Classification : " + save_path_file)
    print_border_line()
    print(data_frame)
    print()


# Generate csv files classification
def classification_write_to_csv(data_frame, save_path_file):
    data_frame.to_csv(save_path_file, index=False)


#
def classification_create_result(column_label, data_test, y_prediction, save_path_file):
    array_item_column = [[] for x in range(len(column_label))]

    for column in data_test:
        array_item_column[0].append(cast_to_int(column[0]))
        array_item_column[1].append(cast_to_int(column[1]))
        array_item_column[2].append(cast_to_int(column[2]))
        array_item_column[3].append(cast_to_int(column[3]))

    classification_data_frame = pd.DataFrame({column_label[0]: array_item_column[0],
                                              column_label[1]: array_item_column[1],
                                              column_label[2]: array_item_column[2],
                                              column_label[3]: array_item_column[3],
                                              COLUMN_CLASS: y_prediction})

    classification_print(classification_data_frame, save_path_file)
    classification_write_to_csv(classification_data_frame, save_path_file)


def classification(path_data_result_clustering, column_label, save_path_file):
    dataset = pd.read_csv(path_data_result_clustering)

    x_dataset = dataset.iloc[:, :-1].values
    y_dataset = dataset.iloc[:, len(column_label)].values

    x_training, x_test, y_training, y_test = train_test_split(x_dataset, y_dataset, test_size=CLASSIFICATION_SIZE)

    scaler = StandardScaler()
    scaler.fit(x_training)

    x_training = scaler.transform(x_training)
    temp_test = x_test
    x_test = scaler.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=CLASSIFICATION_NEIGHBOR)
    classifier.fit(x_training, y_training)
    y_prediction = classifier.predict(x_test)

    classification_create_result(column_label, temp_test, y_prediction, save_path_file)

    evaluation(y_test, y_prediction)
