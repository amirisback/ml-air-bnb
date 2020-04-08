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


def logic_random():
    print_border_line()
    print(" --      Column random       -- ")
    random_index = [random_number(), random_number(), random_number(), random_number()]
    random_index.sort()
    # parameter_column = [COLUMN_PRICE, COLUMN_MINIMUM_NIGHTS, COLUMN_NUMBER_OF_REVIEWS, COLUMN_AVAILABILITY_365]
    parameter_column = [
        LIST_INT_COLUMN[random_index[0]],
        LIST_INT_COLUMN[random_index[1]],
        LIST_INT_COLUMN[random_index[2]],
        LIST_INT_COLUMN[random_index[3]]
    ]
    print_border_line()
    print(random_index)
    print(parameter_column)

    clustering(FILE_NAME_RAW_DATA_SET, parameter_column, FILE_NAME_RESULT_CLUSTERING_RANDOM)
    classification(FILE_NAME_RESULT_CLUSTERING_RANDOM, parameter_column, FILE_NAME_RESULT_CLASSIFICATION_RANDOM)


def logic():
    print_border_line()
    print(" -- Column already specified -- ")
    parameter_column = [COLUMN_PRICE, COLUMN_MINIMUM_NIGHTS, COLUMN_NUMBER_OF_REVIEWS, COLUMN_AVAILABILITY_365]
    print_border_line()
    print(parameter_column)
    clustering(FILE_NAME_RAW_DATA_SET, parameter_column, FILE_NAME_RESULT_CLUSTERING)
    classification(FILE_NAME_RESULT_CLUSTERING, parameter_column, FILE_NAME_RESULT_CLASSIFICATION)


def main():
    identity()
    logic()
    logic_random()


if __name__ == "__main__":
    main()
