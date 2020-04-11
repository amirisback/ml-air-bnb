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

# Path file data
from datetime import datetime

# Constant
FORMAT_DATE = "%Y-%m-%d"
DATE_TODAY = str(datetime.today().strftime(FORMAT_DATE))
FILE_NAME_RAW_DATA_SET = "raw/air_bnb.csv"
FILE_NAME_RESULT_CLUSTERING = "result/" + DATE_TODAY + "-airbnb-clustering-kmeans.csv"
FILE_NAME_RESULT_CLUSTERING_RANDOM = "result/" + DATE_TODAY + "-airbnb-clustering-kmeans-random.csv"
FILE_NAME_RESULT_CLASSIFICATION = "result/" + DATE_TODAY + "-airbnb-classificaction-knn.csv"
FILE_NAME_RESULT_CLASSIFICATION_RANDOM = "result/" + DATE_TODAY + "-airbnb-classificaction-knn-random.csv"
BORDER_LINE = "-----------------------------------------------------------------------"
IDENTITY_NAME = "Muhammad Faisal Amir"
IDENTITY_NIM = "1301198497"
IDENTITY_CLASS = "IFX-43-GAB"
IDENTITY_MAJORS = "S1 Informatika 2019"
IDENTITY_UNIVERSITY = "Telkom University"
IDENTITY_DATASET = "AIR BNB"

KMEANS_K = 3
KMEANS_TOLERANCE = 0.001
KMEANS_MAX_ITERARIONS = 300

COLORS_CLUSTERING = 10 * ["g", "r", "c", "b", "k"]

CLASSIFICATION_SIZE = 0.30
CLASSIFICATION_NEIGHBOR = 2

# Title column table dataset
COLUMN_ID = "id"  # int
COLUMN_NAME = "name"  # str
COLUMN_HOST_ID = "host_id"  # int
COLUMN_HOST_NAME = "host_name"  # str
COLUMN_NEIGHBOURHOOD_GROUP = "neighbourhood_group"  # str
COLUMN_NEIGHBOURHOOD = "neighbourhood"  # str
COLUMN_LATITUDE = "latitude"  # int
COLUMN_LONGITUDE = "longitude"  # int
COLUMN_ROOM_TYPE = "room_type"  # str
COLUMN_PRICE = "price"  # int
COLUMN_MINIMUM_NIGHTS = "minimum_nights"  # int
COLUMN_NUMBER_OF_REVIEWS = "number_of_reviews"  # int
COLUMN_LAST_REVIEW = "last_review"  # str
COLUMN_REVIEWS_PER_MONTH = "reviews_per_month"  # int
COLUMN_CALCULATED_HOST = "calculated_host_listings_count"  # int
COLUMN_AVAILABILITY_365 = "availability_365"  # int
COLUMN_CLASS = "room_level"

COLUMN_0 = COLUMN_ROOM_TYPE
COLUMN_1 = COLUMN_PRICE
COLUMN_2 = COLUMN_MINIMUM_NIGHTS
COLUMN_3 = COLUMN_NUMBER_OF_REVIEWS

DATA_SET_LABEL = [COLUMN_0, COLUMN_1, COLUMN_2, COLUMN_3]

LIST_INT_COLUMN = [COLUMN_ID,
                   COLUMN_HOST_ID,
                   COLUMN_LATITUDE,
                   COLUMN_LONGITUDE,
                   COLUMN_PRICE,
                   COLUMN_MINIMUM_NIGHTS,
                   COLUMN_NUMBER_OF_REVIEWS,
                   COLUMN_REVIEWS_PER_MONTH,
                   COLUMN_CALCULATED_HOST,
                   COLUMN_AVAILABILITY_365]
