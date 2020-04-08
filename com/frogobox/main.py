#
# Created by Faisal Amir
# FrogoBox Inc License
# -----------------------------------------
# Mini-ML-Air-BnB
# Copyright (C) 16/03/2020.      
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


import numpy as np
import matplotlib.pyplot as plt, mpld3
from com.frogobox.base.config import *
from matplotlib import style
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

style.use('ggplot')


class K_Means:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, data):

        self.centroids = {}

        # inisialasi centroid
        for i in range(self.k):
            self.centroids[i] = data[i]

        # Mulai Iterasi
        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            # find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                klasifikasi = distances.index(min(distances))
                self.classes[klasifikasi].append(features)

            previous = dict(self.centroids)

            # rata-rata datapoints cluster untuk menghitung ulang centroid
            for klasifikasi in self.classes:
                self.centroids[klasifikasi] = np.average(self.classes[klasifikasi], axis=0)

            isOptimal = True

            for centroid in self.centroids:

                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False

            # Hentikan looping jika tida terjadi perubuhan centroid
            if isOptimal:
                break

    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        klasifikasi = distances.index(min(distances))
        return klasifikasi


def main():
    df = pd.read_csv('raw/fifa20.csv')
    df = df[['wage_eur', 'value_eur']]
    dataset = df.astype(float).values.tolist()

    X = df.values  # kemablikan numpy array

    km = K_Means(3)
    km.fit(X)

    wage_eurs = []
    value_eurs = []
    labels = []
    for klasifikasi in km.classes:
        for features in km.classes[klasifikasi]:
            wage_eurs.append(features[0])
            value_eurs.append(features[1])
            label = klasifikasi + 1
            labels.append("Class " + str(label))
            print(features)
            print(klasifikasi)
    df = pd.DataFrame({"wage_eur": wage_eurs, "value_eur": value_eurs, "Class": labels})
    df.to_csv(FILE_NAME_RESULT_CLUSTERING, index=False)

    filename = FILE_NAME_RESULT_CLUSTERING
    dataset = pd.read_csv(filename)
    # bagi data set kedalam nilai dan label
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 2].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Feature Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Melatih dan Prekdiksi
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # evaluasi
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
