# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:50:20 2020

@author: rashe
"""

import numpy as np
import pandas as pd
import keras.backend as K
from network_builder import buildNetwork
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_digits
from curricula.curricula import curriculum
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def flattenList(t):
    return [item for sublist in t for item in sublist]

def loadData():
    from sklearn.model_selection import train_test_split
    X,y = load_digits(return_X_y=True)
    unique_classes = pd.DataFrame(y)[0].unique()
    ohe = OneHotEncoder(sparse=False)
    y = ohe.fit_transform(pd.DataFrame(y))
    train_x, val_x, train_y, val_y = train_test_split(X,y,train_size=0.6)
    val_x, test_x, val_y, test_y = train_test_split(val_x, val_y,train_size=0.5)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, unique_classes

def checkBatchVariance(batches):
    import matplotlib.pyplot as plt
    res = dict()
    for i in range(len(batches)):
        re =[]
        for j in range(len(batches[i])):
            re.append(np.var(batches[i][j]))
        res[i] = pd.DataFrame(re)
        for z in range(len(res[i])):
            plt.plot(np.array(res[i][z:z+1])[0])
        plt.title("curriculum" + str(i))
        plt.show()
        
def checkBatchMean(batches):
    import matplotlib.pyplot as plt
    res = dict()
    for i in range(len(batches)):
        re =[]
        for j in range(len(batches[i])):
            re.append(np.mean(batches[i][j]))
        res[i] = pd.DataFrame(re)
        for z in range(len(res[i])):
            plt.plot(np.array(res[i][z:z+1])[0])
        plt.title("curriculum" + str(i))
        plt.show()
        
        
def forwardPass(model, inputs, layer):
    if layer > len(model.layers) - 1:
        layer = len(model.layers) - 1
    elif layer < 0:
        layer = 0
    
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
            if i == 0:
                curric = curric.sort_values(0)
            else:
                curric = curric.sort_values(0, ascending=False)
        else:
            if i == 0:
                curric = curric.sort_values(0, ascending=False)
            else:
                curric = curric.sort_values(0)

        for batch in range(numbatches):
            if cumulative == True:
                ba[batch] = datas[0].iloc[curric[0:(batch+1)*k].index]
            else:
                ba[batch] = datas[0].iloc[curric[batch*k:(batch+1)*k].index]
            ou[batch] = datas[1].iloc[ba[batch].index]
        batches[i] = ba
        outs[i] = ou
    
    return batches, outs


def getEmbeddeds(model, batches):
    embeddeds = dict()
    for i in range(len(batches)):
        emb = dict()
        for j in range(len(batches[i])):
            emb[j] = forwardPass(model, batches[i][j], len(model.layers))
        embeddeds[i] = emb
    return embeddeds

def train(model,batches,outs,val_x,val_y,curriculum,recurs=10):
    if curriculum.lower() in ('m', 'mahalanobis', 'maha'):
        curric = 0
    elif curriculum.lower() in ('c','cosine','cos'):
        curric = 1
    elif curriculum.lower() in ('w', 'wasserstein', 'wass'):
        curric = 2
    else:
        raise('Need to choose a valid curriculum')
    
    l = []
    a = []
    va_l = []
    va_a = []
    for b,o in zip(batches[curric],outs[curric]):
        for r in range(recurs):
            loss,acc = model.train_on_batch(batches[curric][b],outs[curric][o])
            l.append(loss)
            a.append(acc)
            val_loss, val_acc = model.test_on_batch(val_x, val_y)
            va_l.append(val_loss)
            va_a.append(val_acc)

    return model, l, a, va_l, va_a
    
def vanillaCurriculum(model, batches, outs, data, recursion, plot=True):
    for curric in ('m', 'c', 'w'):
        bn = buildNetwork()
        model = bn.build(data[0],pd.DataFrame(uniques))
        model = bn.compiler(model, q_net=False, actor=False)
        training_loss = []
        validation_loss = []
        training_acc = []
        validation_acc = []
        embs = dict()
        embs[0] = getEmbeddeds(model, batches)
        for r in range(repetitions):
            model, l, a, va_l, va_a = train(model, batches, outs, data[2], data[3], curriculum=curric, recurs=1000)
            embs[r+1] = getEmbeddeds(model, batches)
            training_loss.append(l)
            validation_loss.append(va_l)
            training_acc.append(a)
            validation_acc.append(va_a)
        
        validation_acc = flattenList(validation_acc)
        training_acc = flattenList(training_acc)
        training_loss = flattenList(training_loss)
        validation_loss = flattenList(validation_loss)
        
        if plot:
            plt.plot(validation_acc)
            plt.ylabel("validation accuracy")
            plt.xlabel("Iteration")
            plt.legend(['m', 'c', 'w'])
            plt.show()
            
            plt.plot(validation_loss)
            plt.ylabel("validation loss")
            plt.xlabel("Iteration")
            plt.legend(['m', 'c', 'w'])
            plt.show()
    

if __name__ == "__main__":
    
    repetitions = 1
    data = loadData()
    uniques = data[-1]
    datas = dict()
    for z in range(len(data)):
        datas[z] = toDataframe(data[z])
    
    batches, outs = createBatches(datas, 10, forward = False)
    
    checkBatchMean(batches)
    checkBatchVariance(batches)
    
    bn = buildNetwork()
    model = bn.build(data[0],pd.DataFrame(uniques))
    model = bn.compiler(model, q_net=False, actor=False)
    
    vanillaCurriculum(model, batches, outs, data, recursion=5, plot=True)