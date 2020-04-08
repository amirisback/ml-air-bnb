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

from com.frogobox.algorithm.classifiation import classification
from com.frogobox.algorithm.clustering import clustering
from com.frogobox.base.config import *


def main():
    clustering(FILE_NAME_RAW_DATA_SET)
    classification(FILE_NAME_RESULT_CLUSTERING)


if __name__ == "__main__":
    main()
