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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("raw/air_bnb.csv", delimiter=',')

# --- Menghilangkan Kolom Yang Tidak Perlu ---
# simpleData = data.drop([id,name,host_id,host_name,neighbourhood_group,neighbourhood,latitude,longitude,room_type,price,minimum_nights,number_of_reviews,last_review,reviews_per_month,calculated_host_listings_count,availability_365])
data = data.drop(
    ["host_id", "host_name", "neighbourhood_group", "neighbourhood", "latitude", "longitude",
     "minimum_nights", "number_of_reviews", "last_review", "reviews_per_month", "calculated_host_listings_count",
     "availability_365"], axis=1)
data.head()
print(data)

# -- Menentukan variabel yang akan di klusterkan ---
data_x = data.iloc[:, 1:3]
data_x.head()
print(data_x)

# --- Memvisualkan persebaran simpleData ---
plt.scatter(data.name, data.room_type, 10, "c", "o", 1)
plt.show()

# --- Mengubah Variabel simpleData Frame Menjadi Array ---
x_array = np.array(data_x)
print(x_array)

# --- Menstandarkan Ukuran Variabel ---
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled

# --- Menentukan dan mengkonfigurasi fungsi kmeans ---
kmeans = KMeans(n_clusters=3, random_state=123)
# --- Menentukan kluster dari data ---
kmeans.fit(x_scaled)

# --- Menampilkan pusat cluster ---
print(kmeans.cluster_centers_)

# --- Menampilkan Hasil Kluster ---
print(kmeans.labels_)
# --- Menambahkan Kolom "kluster" Dalam Data Frame Driver ---
data["kluster"] = kmeans.labels_

# --- Memvisualkan hasil kluster ---
output = plt.scatter(x_scaled[:, 0], x_scaled[:, 1], s=100, c=data.kluster, marker="o", alpha=1, )
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=1, marker="s")
plt.title("Hasil Klustering K-Means")
plt.colorbar(output)
plt.show()