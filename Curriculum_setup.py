# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:50:20 2020

@author: rashe
"""

import numpy as np
import pandas as pd
import keras.backend as K
from network_builder import buildNetwork
from sklearn.datasets import load_iris
from curricula.curricula import curriculum

def loadData():
    from sklearn.model_selection import train_test_split
    X,y = load_iris(return_X_y=True)
    unique_classes = pd.DataFrame(y)[0].unique()
    train_x, val_x, train_y, val_y = train_test_split(X,y,train_size=0.6)
    val_x, test_x, val_y, test_y = train_test_split(val_x, val_y,train_size=0.5)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, unique_classes


def forwardPass(model, inputs, layer):
    if layer > len(model.layers) - 1:
        layer = len(model.layers) - 1
    
    get_hidden_layer_output = K.function([model.layers[0].input],
                               [model.layers[layer].output])
    layer_output = pd.DataFrame(get_hidden_layer_output([inputs])[0])
 
    return layer_output

def toDataframe(arg):
    return pd.DataFrame(arg)

def getRankings(data):
    c = curriculum(data)
    if isinstance(data, pd.core.frame.DataFrame):
        ma = c.getMahalanobis()
        co = c.getCosine()
        wa = c.getWasserstein()
    else:
        for z in data:
            data[z] = toDataframe(z)
        ma = c.getMahalanobis()
        co = c.getCosine()
        wa = c.getWasserstein()
    
    return ma, co, wa

def createBatches(datas, numbatches, cumulative=False, forward=True):
    k = int(len(datas[0])/numbatches)
    ma, co, wa = getRankings(datas[0])
    currics = [ma, co, wa]
    batches = dict()
    outs = dict()
    for i,curric in enumerate(currics):
        ba = dict()
        ou = dict()
        if forward:
            curric = curric.sort_values(0, ascending=False)
        else:
            curric = curric.sort_values(0)
        for batch in range(numbatches):
            if cumulative == True:
                ba[batch] = datas[0].iloc[0:(batch+1)*k]
            else:
                ba[batch] = datas[0].iloc[batch*k:(batch+1)*k]
            ou = datas[1].iloc[ba[batch].index]
        batches[i] = ba
        outs[i] = ou
    
    return batches, outs
    
data = loadData()
uniques = data[-1]
datas = dict()
for z in range(len(data)):
    datas[z] = toDataframe(data[z])

ma, co, wa = getRankings(datas[0])

batches, outs = createBatches(datas, 10, False)

bn = buildNetwork()
model = bn.build(data[0],pd.DataFrame(uniques))
model = bn.compiler(model, q_net=False, actor=False)



