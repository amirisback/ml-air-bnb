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
    df = fetch_raw_data[['wage_eur', 'value_eur', "overall", "potential"]]
    dataset = df.astype(float).values.tolist()
    X = df.values  # kembalikan numpy array
    km = Kmeans(2)
    km.fit(X)
    wage_eurs = []
    value_eurs = []
    overalls = []
    potentials = []
    labels = []
    for classification in km.classes:
        for features in km.classes[classification]:
            wage_eur = features[0]
            value_eur = features[1]
            overall = features[2]
            potential = features[3]
            wage_eurs.append(wage_eur)
            value_eurs.append(value_eur)
            overalls.append(overall)
            potentials.append(potential)
            label = classification + 1
            labels.append("Class " + str(label))
    df = pd.DataFrame(
        {"wage_eur": wage_eurs, "value_eur": value_eurs, "overall": overalls, "potential": potential, "Class": labels})
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
    wage_eurs = []
    value_eurs = []
    overalls = []
    potentials = []
    for features in temp_test:
        wage_eur = features[0]
        value_eur = features[1]
        overall = features[2]
        potential = features[3]
        wage_eurs.append(wage_eur)
        value_eurs.append(value_eur)
        overalls.append(overall)
        potentials.append(potential)
    df = pd.DataFrame(
        {"wage_eur": wage_eurs, "value_eur": value_eurs, "overall": overalls, "potential": potential, "Class": y_pred})
    df.to_csv(FILE_NAME_RESULT_CLASSIFICATION, index=False)

    # evaluasi
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
