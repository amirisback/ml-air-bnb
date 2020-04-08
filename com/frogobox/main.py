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
    df = fetch_raw_data[[DATA_SET_PRICE, DATA_SET_MINIMUM_NIGHTS, DATA_SET_NUMBER_OF_REVIEWS, DATA_SET_AVAILABILITY_365]]
    dataset = df.astype(float).values.tolist()
    X = df.values  # kembalikan numpy array
    km = Kmeans(2)
    km.fit(X)
    prices = []
    minimum_nightss = []
    number_of_reviewss = []
    availability_365s = []
    labels = []
    for classification in km.classes:
        for features in km.classes[classification]:
            price = features[0]
            minimum_nights = features[1]
            number_of_reviews = features[2]
            availability_365 = features[3]
            prices.append(price)
            minimum_nightss.append(minimum_nights)
            number_of_reviewss.append(number_of_reviews)
            availability_365s.append(availability_365)
            label = classification + 1
            labels.append(DATA_SET_CLASS + "_" + str(label))
    df = pd.DataFrame({DATA_SET_PRICE: prices, DATA_SET_MINIMUM_NIGHTS: minimum_nightss, DATA_SET_NUMBER_OF_REVIEWS: number_of_reviewss,
                       DATA_SET_AVAILABILITY_365: availability_365, DATA_SET_CLASS: labels})
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
    prices = []
    minimum_nightss = []
    number_of_reviewss = []
    availability_365s = []
    for features in temp_test:
        price = features[0]
        minimum_nights = features[1]
        number_of_reviews = features[2]
        availability_365 = features[3]
        prices.append(price)
        minimum_nightss.append(minimum_nights)
        number_of_reviewss.append(number_of_reviews)
        availability_365s.append(availability_365)
    df = pd.DataFrame(
        {DATA_SET_PRICE: prices, DATA_SET_MINIMUM_NIGHTS: minimum_nightss, DATA_SET_NUMBER_OF_REVIEWS: number_of_reviewss,
         DATA_SET_AVAILABILITY_365: availability_365, DATA_SET_CLASS: y_pred})
    df.to_csv(FILE_NAME_RESULT_CLASSIFICATION, index=False)

    # evaluasi
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
