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
from com.frogobox.base.config import *

from com.frogobox.base.config import FILE_NAME_RESULT_CLUSTERING
from com.frogobox.algorithm.kmeans import Kmeans
from com.frogobox.base.helper import get_value_type


def create_result_clustering(column_label, kmeans):
    item_column_0 = []
    item_column_1 = []
    item_column_2 = []
    item_column_3 = []
    labels = []

    for classification in kmeans.classes:
        for column in kmeans.classes[classification]:
            item_column_0.append(get_value_type(column[0]))
            item_column_1.append(get_value_type(column[1]))
            item_column_2.append(get_value_type(column[2]))
            item_column_3.append(get_value_type(column[3]))
            label = classification + 1
            labels.append(COLUMN_CLASS + "_" + str(label))

    data_frame = pd.DataFrame({column_label[0]: item_column_0,
                               column_label[1]: item_column_1,
                               column_label[2]: item_column_2,
                               column_label[3]: item_column_3,
                               COLUMN_CLASS: labels})

    # Create clustering result csv
    data_frame.to_csv(FILE_NAME_RESULT_CLUSTERING, index=False)


def clustering(path_file_raw_dataset, column_label):
    fetch_raw_data = pd.read_csv(path_file_raw_dataset)
    sorted_raw_data = fetch_raw_data[column_label]
    numpy_array = sorted_raw_data.values  # kembalikan numpy array
    kmeans = Kmeans(2)
    kmeans.fit(numpy_array)
    create_result_clustering(column_label, kmeans)
