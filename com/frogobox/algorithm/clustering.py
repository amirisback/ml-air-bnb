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

from com.frogobox.base.config import FILE_NAME_RAW_DATA_SET, FILE_NAME_RESULT_CLUSTERING
from com.frogobox.algorithm.kmeans import Kmeans


def clustering():
    fetch_raw_data = pd.read_csv(FILE_NAME_RAW_DATA_SET)
    sorted_raw_data = fetch_raw_data[
        [DATA_SET_PRICE, DATA_SET_MINIMUM_NIGHTS, DATA_SET_NUMBER_OF_REVIEWS, DATA_SET_AVAILABILITY_365]]
    numpy_array = sorted_raw_data.values  # kembalikan numpy array
    kmeans = Kmeans(2)
    kmeans.fit(numpy_array)

    item_price = []
    item_minimum_nights = []
    item_number_of_reviews = []
    item_availability_365 = []
    labels = []

    for classification in kmeans.classes:
        for features in kmeans.classes[classification]:
            item_price.append(features[0])
            item_minimum_nights.append(features[1])
            item_number_of_reviews.append(features[2])
            item_availability_365.append(features[3])
            label = classification + 1
            labels.append(DATA_SET_CLASS + "_" + str(label))

    data_frame = pd.DataFrame({DATA_SET_PRICE: item_price, DATA_SET_MINIMUM_NIGHTS: item_minimum_nights,
                               DATA_SET_NUMBER_OF_REVIEWS: item_number_of_reviews,
                               DATA_SET_AVAILABILITY_365: item_availability_365, DATA_SET_CLASS: labels})
    data_frame.to_csv(FILE_NAME_RESULT_CLUSTERING, index=False)
