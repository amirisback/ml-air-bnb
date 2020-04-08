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
from sklearn.metrics import classification_report, confusion_matrix
from com.frogobox.base.config import *

from com.frogobox.base.config import FILE_NAME_RAW_DATA_SET, FILE_NAME_RESULT_CLUSTERING
from com.frogobox.kmeans import Kmeans


def main():
    fetch_raw_data = pd.read_csv(FILE_NAME_RAW_DATA_SET)
    df = fetch_raw_data[
        [DATA_SET_PRICE, DATA_SET_MINIMUM_NIGHTS, DATA_SET_NUMBER_OF_REVIEWS, DATA_SET_AVAILABILITY_365]]
    dataset = df.astype(float).values.tolist()
    X = df.values  # kembalikan numpy array
    km = Kmeans(2)
    km.fit(X)
    item_price = []
    item_minimum_nights = []
    item_number_of_reviews = []
    item_availability_365 = []
    labels = []
    for classification in km.classes:
        for features in km.classes[classification]:
            fetch_price = features[0]
            fetch_minimum_nights = features[1]
            fetch_number_of_reviews = features[2]
            fetch_availability_365 = features[3]

            item_price.append(fetch_price)
            item_minimum_nights.append(fetch_minimum_nights)
            item_number_of_reviews.append(fetch_number_of_reviews)
            item_availability_365.append(fetch_availability_365)
            label = classification + 1
            labels.append(DATA_SET_CLASS + "_" + str(label))

    df = pd.DataFrame({DATA_SET_PRICE: item_price, DATA_SET_MINIMUM_NIGHTS: item_minimum_nights,
                       DATA_SET_NUMBER_OF_REVIEWS: item_number_of_reviews,
                       DATA_SET_AVAILABILITY_365: fetch_availability_365, DATA_SET_CLASS: labels})
    df.to_csv(FILE_NAME_RESULT_CLUSTERING, index=False)

    dataset = pd.read_csv(FILE_NAME_RESULT_CLUSTERING)
    # bagi data set kedalam nilai dan label
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Feature Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    temp_test = X_test
    X_test = scaler.transform(X_test)

    # Melatih dan Prekdiksi
    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    item_price = []
    item_minimum_nights = []
    item_number_of_reviews = []
    item_availability_365 = []
    for features in temp_test:
        fetch_price = features[0]
        fetch_minimum_nights = features[1]
        fetch_number_of_reviews = features[2]
        fetch_availability_365 = features[3]

        item_price.append(fetch_price)
        item_minimum_nights.append(fetch_minimum_nights)
        item_number_of_reviews.append(fetch_number_of_reviews)
        item_availability_365.append(fetch_availability_365)

    df = pd.DataFrame(
        {DATA_SET_PRICE: item_price, DATA_SET_MINIMUM_NIGHTS: item_minimum_nights,
         DATA_SET_NUMBER_OF_REVIEWS: item_number_of_reviews,
         DATA_SET_AVAILABILITY_365: fetch_availability_365, DATA_SET_CLASS: y_pred})
    df.to_csv(FILE_NAME_RESULT_CLASSIFICATION, index=False)

    # evaluasi
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
