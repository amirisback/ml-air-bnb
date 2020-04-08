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

from com.frogobox.base.config import *


def cast_to_int(x):
    if type(x) is str:
        return len(x)
    elif type(x) is float:
        return int(x)
    else:
        return x


def get_list_column_length():
    return len(LIST_INT_COLUMN)


def random_number():
    return random.randint(0, get_list_column_length()-1)


def print_border_line():
    print(BORDER_LINE)
