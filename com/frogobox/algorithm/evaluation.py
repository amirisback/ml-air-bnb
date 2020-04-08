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

from sklearn.metrics import classification_report, confusion_matrix


def evaluation(y_test, y_pred):
    # evaluasi
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
