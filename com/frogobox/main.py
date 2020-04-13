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
import random
from com.frogobox.algorithm.classification import classification
from com.frogobox.algorithm.clustering import clustering
from com.frogobox.base.config import *
from com.frogobox.base.helper import print_border_line, get_list_column_length, random_number


def identity():
    print("Name \t\t: " + IDENTITY_NAME)
    print("NIM \t\t: " + IDENTITY_NIM)
    print("Class \t\t: " + IDENTITY_CLASS)
    print("Major \t\t: " + IDENTITY_MAJORS)
    print("University \t: " + IDENTITY_UNIVERSITY)
    print_border_line()
    print("Dataset : " + IDENTITY_DATASET)
    print("Date : " + DATE_TODAY)


def logic():
    print_border_line()
    print(" -- Column already specified -- ")
    print_border_line()
    print(DATA_SET_FEATURES)
    clustering(FILE_NAME_RAW_DATA_SET, DATA_SET_FEATURES, FILE_NAME_RESULT_CLUSTERING)
    classification(FILE_NAME_RESULT_CLUSTERING, DATA_SET_FEATURES, FILE_NAME_RESULT_CLASSIFICATION)


def main():
    identity()
    logic()



if __name__ == "__main__":
    main()
