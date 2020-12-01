#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:52:28 2020

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:53:02 2020

@author: Rasheed el-Bouri
"""
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd



class curriculum():
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def getMahalanobis(self):

        dataframe = pd.DataFrame(self.dataframe)
        dataframe = dataframe.reset_index(drop=True)
        nunique = dataframe.apply(pd.Series.nunique)
        if dataframe.shape[1] >= 15:
             cols_to_drop = nunique[nunique <= 2].index
             dataframe = dataframe.drop(cols_to_drop, axis=1) 
             
        features = list(dataframe)
        means = pd.DataFrame(np.zeros(len(features)))
        covariance = np.cov(dataframe.T)
        inv_cov = np.linalg.inv(covariance)
        Mahalanobis = np.zeros(len(dataframe))
     
        for j in range(0,len(means)):
                 means[0][j] = np.mean(dataframe.iloc[:,j])
                
        means = means.reset_index(drop=True)
         
        for i in range(0,len(dataframe)):
            first = pd.DataFrame(dataframe.iloc[i,:]).reset_index(drop=True)    
            
            V = first[i]-means[0]
            Mahalanobis[i] = np.sqrt(np.abs(np.dot(np.dot(V.T,inv_cov), V)))#[0][0]
            
        
        return(pd.DataFrame(Mahalanobis))
        
            
         
    def getCosine(self):
        
        dataframe = self.dataframe.reset_index(drop=True)
        
        features = list(dataframe)
        means = pd.DataFrame(np.zeros(len(features)))
        
        for j in range(0,len(means)):
            means[0][j] = np.mean(dataframe.iloc[:,j])
    
        l=[]
        for i in range(0, len(dataframe)):
            l.append(np.arccos(np.dot(np.array(dataframe[i:i+1]),np.array(means))/(np.linalg.norm(np.array(dataframe[i:i+1]))*np.linalg.norm(np.array(means))))[0])
        
        return(pd.DataFrame(np.array(l)))
       
    
    
    def getWasserstein(self):
        
        dataframe = self.dataframe.reset_index(drop=True)
        
        uniform = (1/dataframe.shape[1])*np.ones(dataframe.shape[1])
        
        l = []
        for i in range(0,len(dataframe)):
            l.append(wasserstein_distance(uniform, np.array(dataframe[i:i+1])[0]))
        
        return(pd.DataFrame(np.array(l)))
        
        
