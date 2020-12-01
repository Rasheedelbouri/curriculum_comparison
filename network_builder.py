#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:25:33 2020

@author: kebl4170
"""
import os    
os.environ['THEANO_FLAGS'] = "device=cuda0"    
#import theano
#theano.config.floatX = 'float32'
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras import regularizers
import numpy as np


class buildNetwork():
    
    def __init__(self, source='H', seed = 0, hidden_layers=2, hidden_nodes=50, temp=1, dropout=0.2,\
                 activation='relu', batchnorm=True, numepochs=100, batchsize=30 \
                 ,curriculum_batches = 10, curriculum_recursion = 1, q_net = False,
                 act_net = False, crit_net=False):
        self.source = source
        self.seed = seed
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.temp = temp
        self.dropout = dropout
        self.activation = activation
        self.batchnorm = batchnorm
        self.numepochs = numepochs
        self.batchsize = batchsize
        self.curriculum_batches = curriculum_batches
        self.curriculum_recursion = curriculum_recursion
        self.q_net = q_net
        self.act_net = act_net
        self.crit_net = crit_net
        

    def build(self, train_x, uniques):
        np.random.seed(self.seed) # setting initial seed for reproducable results
        
        model = Sequential() # begin a sequential model
        #model.add(Hadamard(input_shape=([82])))
        #model.add(Activation('softmax'))
        model.add(Dense(self.hidden_nodes, input_dim=train_x.shape[1], activation=self.activation))
        if self.batchnorm == True:
            model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

        for i in range(0, self.hidden_layers):
            model.add(Dense(self.hidden_nodes, activation= self.activation, W_regularizer = regularizers.l2(1e-3)))
            model.add(Dropout(self.dropout))
            if self.batchnorm == True:
                model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
        
        if self.source == 'M':
            model.add(Dense(len(uniques)-1, activation = 'sigmoid'))
        else:
            if self.act_net == True:
                model.add(Dense(len(uniques), activation='tanh'))
                #model.add(Dense(len(uniques), activation='sigmoid'))
            elif self.crit_net == True:
                model.add(Dense(len(uniques), activation=None))
            else:
                model.add(Dense(len(uniques), activation = 'softmax'))
                #model.add(Lambda(lambda x: x / temp))
        #model.summary()
        
        return(model)
        
    def customLoss(self, yTrue, yPred):
        
        #yPred=0
        loss = 10*tf.reduce_mean(yTrue - yPred)
        return loss
        
    def compiler(self, model, q_net, actor):
        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        if q_net == True:
            if actor == True:
                model.compile(loss=self.customLoss,
                              optimizer=sgd,
                              metrics=['accuracy'])
                
            model.compile(loss=self.customLoss,# self.customLoss, #self.my_loss, 
                  optimizer= 'adam', 
                  metrics=['accuracy'])
        else:
            if self.source == 'M':
                model.compile(loss='binary_crossentropy', #self.my_loss, 
                      optimizer= 'adam', 
                      metrics=['accuracy'])
            else:
                model.compile(loss='categorical_crossentropy', #self.my_loss, 
                          optimizer= 'adam',
                          metrics=['accuracy'])
            
        return(model)
        
