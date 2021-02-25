# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:06:34 2020

@author: rashe
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from network_builder import buildNetwork
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_digits
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy
from scipy.stats import entropy
import argparse
from utils.boolParse import str2bool
import os


def flattenList(t):
    return [item for sublist in t for item in sublist]

def toDataframe(arg):
    return pd.DataFrame(arg)

def loadData():
    from sklearn.model_selection import train_test_split
    X,y = load_digits(return_X_y=True)
    unique_classes = pd.DataFrame(y)[0].unique()
    ohe = OneHotEncoder(sparse=False)
    y = ohe.fit_transform(pd.DataFrame(y))
    train_x, val_x, train_y, val_y = train_test_split(X,y,train_size=0.6)
    val_x, test_x, val_y, test_y = train_test_split(val_x, val_y,train_size=0.5)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, unique_classes


def forwardPass(model, inputs, layer):
    if layer > len(model.layers) - 1:
        layer = len(model.layers) - 1
    elif layer < 0:
        layer = 0
    
    get_hidden_layer_output = K.function([model.layers[0].input],
                               [model.layers[layer].output])
    layer_output = pd.DataFrame(get_hidden_layer_output([inputs])[0])
 
    return layer_output


def getLosses(true, pred):
    
    product = categorical_crossentropy(true, pred)
    with tf.Session() as sess:
        loss = product.eval()

    return loss

def getLossRankings(model, data, forward=True):
    l = forwardPass(model, data[0], len(model.layers)-1)
    losses = pd.DataFrame(getLosses(data[1],l))
    losses = losses.sort_values(0, ascending=forward)

    return losses

def getGradientRankings(model, data, forward=True):
    outputTensor = model.output
    listOfVariableTensors = model.trainable_weights
    grad_mag = []
    gradients = K.gradients(outputTensor, listOfVariableTensors)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for d in range(len(data[0])):
        evaluated_gradients = sess.run(gradients,feed_dict={model.input:data[0][d:d+1]})
        x = 0 
        for i in range(len(evaluated_gradients)):
            x += np.linalg.norm(evaluated_gradients[i])
        grad_mag.append(x)
    
    grad_mag = pd.DataFrame(grad_mag)
    grad_mag = grad_mag.sort_values(0, ascending=forward)
    
    return grad_mag

def getEntropyRankings(model, data, forward=True):
    entrop=[]
    l = forwardPass(model, data[0], len(model.layers)-1)
    for d in range(len(data[0])):
        entrop.append(entropy(np.array(l[d:d+1])[0]))
    
    entrop = pd.DataFrame(entrop)
    entrop = entrop.sort_values(0, ascending=forward)
    
    return entrop

def getUncertaintyRankings(model, data, forward=True):
    embeds = []
    for i in range(10):
        bn = buildNetwork(seed=i,dropout=np.random.random(1)[0])
        m = bn.build(data[0], data[-1])
        m = bn.compiler(m,q_net=False,actor=False)
        embeds.append(forwardPass(m, data[0], len(model.layers)-1))
    
    stacked = np.stack(embeds, axis=-1)
    variance = []
    for i in range(len(stacked)):
        variance.append(np.var(stacked[:,0:1][i][0]))
    
    variance = pd.DataFrame(variance)
    variance = variance.sort_values(0, ascending=forward)
    
    return variance


def createLossBasedBatches(model, data, numbatches, cumulative=False, forward=True,metric='loss'):
    if metric.lower() == 'loss':
        losses = getLossRankings(model, data, forward)
    elif metric.lower() == "gradient":
        losses = getGradientRankings(model, data, forward)
    elif metric.lower() == 'entropy':
        losses = getEntropyRankings(model,data,forward)
    elif metric.lower() == 'uncertainty':
        losses = getUncertaintyRankings(model, data, forward)
    train_x = toDataframe(data[0])
    train_y = toDataframe(data[1])
    k = int(len(train_x)/numbatches)
    batches = dict()
    outs = dict()
    for batch in range(numbatches):
        if cumulative==False:
            batches[batch] = train_x.iloc[losses[k*batch:k*(batch+1)].index]
        else:
            batches[batch] = train_x.iloc[losses[0:k*(batch+1)].index]
        outs[batch] = train_y.iloc[batches[batch].index]
    return batches, outs

def trainCurriculum(model, batches, outs, data, epochs=1):
    train_acc=[]
    train_loss=[]
    val_acc=[]
    val_loss=[]
    for epo in range(epochs):
        for j in range(len(batches)):
           tr_l,tr_a = model.train_on_batch(batches[j],outs[j])
           train_loss.append(tr_l), train_acc.append(tr_a)
           va_l, va_a = model.test_on_batch(data[2],data[3])
           val_loss.append(va_l), val_acc.append(va_a)
    
    return model, train_acc, train_loss, val_acc, val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curric_type", type=str, help="Specify which metric to use for curriculum")
    parser.add_argument("--numbatches", type=int, help="Number of curriculum batches to create")
    parser.add_argument("--cumulative", type=str2bool, help="Make batches cumulative supersets if True")
    parser.add_argument("--forward", type=str2bool, help="Run the curriculum in reverse if False")
    parser.add_argument("--epochs", type=int, help="How many times should batches be recomputed")
    parser.add_argument("--curriculum_epochs", type=int, help="How many times should the same batches be exploited")
    parser.add_argument("--plot", type = str2bool, help="visualise training")
    args = parser.parse_args()
  
    curric_type = args.curric_type
    numbatches = args.numbatches
    cumulative = args.cumulative
    forward = args.forward
    epochs = args.epochs
    curriculum_epochs = args.curriculum_epochs
    plot = args.plot

    save_path="../Results/digits/"+ str(curric_type) 

    data = loadData()
    bn = buildNetwork()
    model = bn.build(data[0],data[-1])
    model = bn.compiler(model, q_net=False, actor=False)
    train_accs=[]
    val_accs = []
    train_loss = []
    val_loss = []
    for e in range(epochs):
        batches, outs = createLossBasedBatches(model, data, numbatches, cumulative,forward,metric=curric_type)
        model, tr_a, tr_l, va_a, va_l = trainCurriculum(model, batches, outs, data, epochs=curriculum_epochs)
        train_accs.append(tr_a)
        val_accs.append(va_a)
        train_loss.append(tr_l)
        val_loss.append(va_l)
        if plot:
            plt.plot(flattenList(train_accs))
            plt.plot(flattenList(val_accs))
            plt.xlabel("Number of batches trained on")
            plt.ylabel("Accuracy")
            plt.legend(["train", 'val'])
            plt.show()
            plt.plot(flattenList(train_loss))
            plt.plot(flattenList(val_loss))
            plt.xlabel("Number of batches trained on")
            plt.ylabel("Loss")
            plt.legend(["train", 'val'])
            plt.show()
    
    if not os.path.exists(os.path.join(save_path, str(numbatches)+"curricBatches")):
        os.mkdir(os.path.join(save_path, str(numbatches)+"curricBatches"))

    with  open(os.path.join(save_path, 
               "epochs"+str(epochs)+
               "cumulative"+str(cumulative)+
               "curricEpochs"+str(curriculum_epochs)+".txt"), "w") as output:
        output.write(str(val_accs))
    output.close()