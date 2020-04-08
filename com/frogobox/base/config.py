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

FORMAT_DATE = "%Y-%m-%d"
DATE_TODAY = str(datetime.today().strftime(FORMAT_DATE))

FILE_NAME_RAW_DATA_SET = "raw/air_bnb.csv"
FILE_NAME_RESULT_CLUSTERING = "result/" + DATE_TODAY + "-airbnb-clustering-kmeans.csv"
FILE_NAME_RESULT_CLASSIFICATION = "result/" + DATE_TODAY + "-airbnb-classificaction-knn.csv"

# Title column table dataset
COLUMN_ID = "id"
COLUMN_NAME = "name"
COLUMN_HOST_ID = "host_id"
COLUMN_HOST_NAME = "host_name"
COLUMN_NEIGHBOURHOOD_GROUP = "neighbourhood_group"
COLUMN_NEIGHBOURHOOD = "neighbourhood"
COLUMN_LATITUDE = "latitude"
COLUMN_LONGITUDE = "longitude"
COLUMN_ROOM_TYPE = "room_type"
COLUMN_PRICE = "price"
COLUMN_MINIMUM_NIGHTS = "minimum_nights"
COLUMN_NUMBER_OF_REVIEWS = "number_of_reviews"
COLUMN_LAST_REVIEW = "last_review"
COLUMN_REVIEWS_PER_MONTH = "reviews_per_month"
COLUMN_CALCULATED_HOST = "calculated_host_listings_count"
COLUMN_AVAILABILITY_365 = "availability_365"
COLUMN_CLASS = "class"
