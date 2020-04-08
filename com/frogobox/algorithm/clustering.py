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


def create_result_clustering(kmeans):
    item_column_0 = []
    item_column_1 = []
    item_column_2 = []
    item_column_3 = []
    labels = []

    for classification in kmeans.classes:
        for column in kmeans.classes[classification]:
            item_column_0.append(column[0])
            item_column_1.append(column[1])
            item_column_2.append(column[2])
            item_column_3.append(column[3])
            label = classification + 1
            labels.append(COLUMN_CLASS + "_" + str(label))

    data_frame = pd.DataFrame({COLUMN_PRICE: item_column_0, COLUMN_MINIMUM_NIGHTS: item_column_1,
                               COLUMN_NUMBER_OF_REVIEWS: item_column_2,
                               COLUMN_AVAILABILITY_365: item_column_3, COLUMN_CLASS: labels})

    # Create clustering result csv
    data_frame.to_csv(FILE_NAME_RESULT_CLUSTERING, index=False)


def clustering(path_file_raw_dataset):
    fetch_raw_data = pd.read_csv(path_file_raw_dataset)
    sorted_raw_data = fetch_raw_data[
        [COLUMN_PRICE, COLUMN_MINIMUM_NIGHTS, COLUMN_NUMBER_OF_REVIEWS, COLUMN_AVAILABILITY_365]]
    numpy_array = sorted_raw_data.values  # kembalikan numpy array
    kmeans = Kmeans(2)
    kmeans.fit(numpy_array)
    create_result_clustering(kmeans)
