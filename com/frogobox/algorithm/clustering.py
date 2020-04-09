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
from pandas import DataFrame

from com.frogobox.base.config import *

from com.frogobox.algorithm.kmeans import Kmeans
from com.frogobox.base.helper import cast_to_int, print_border_line


def create_result_clustering(column_label, kmeans, save_path_file):
    item_column_0 = []
    item_column_1 = []
    item_column_2 = []
    item_column_3 = []
    labels = []

    for classification in kmeans.classes:
        for column in kmeans.classes[classification]:
            item_column_0.append(cast_to_int(column[0]))
            item_column_1.append(cast_to_int(column[1]))
            item_column_2.append(cast_to_int(column[2]))
            item_column_3.append(cast_to_int(column[3]))
            label = classification + 1
            labels.append(COLUMN_CLASS + "_" + str(label))

    clustering_data_frame = pd.DataFrame({column_label[0]: item_column_0,
                                          column_label[1]: item_column_1,
                                          column_label[2]: item_column_2,
                                          column_label[3]: item_column_3,
                                          COLUMN_CLASS: labels})

    print_border_line()
    print("Result Clustering : " + save_path_file)
    print_border_line()
    print(clustering_data_frame)
    print()

    # Create clustering result csv
    clustering_data_frame.to_csv(save_path_file, index=False)


def clustering(path_file_raw_dataset, column_label, save_path_file):
    fetch_raw_data = pd.read_csv(path_file_raw_dataset)
    sorted_raw_data = fetch_raw_data[column_label]
    numpy_array = sorted_raw_data.values
    kmeans = Kmeans(2)
    kmeans.fit(numpy_array)
    create_result_clustering(column_label, kmeans, save_path_file)
