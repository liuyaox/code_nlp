#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:02:18 2019
@author: liuyao8
"""


def model_eval(model, test_x, test_y, y_thresh=0.5):
	""" Ä£ÐÍÆÀ¹À   Metrics: Accuracy, Precision, Recall, F1 Score"""
    ypred = [int(x[0] > y_thresh) for x in model.predict(test_x)]
    TP, FP, TN, FN = 0, 0, 0, 0
    for ytrue, yhat in zip(test_y, ypred):
        if ytrue == 1 and yhat == 1:
            TP += 1
        elif ytrue == 0 and yhat == 1:
            FP += 1
        elif ytrue == 0 and yhat == 0:
            TN += 1
        elif ytrue == 1 and yhat == 0:
            FN += 1
    accuracy = (TP + TN) / res.shape[0]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
	f1 = precision * recall / (precision + recall)
    print(f'TP: {TP}  FP: {FP}  TN: {TN}  FN: {FN}')
    print('Accuracy:\t' + str(accuracy))
    print('Precision:\t' + str(precision))
    print('Recall:\t' + str(recall))
	print('f1 score:\t' + str(f1))
    return accuracy, precision, recall, f1
	
	
	