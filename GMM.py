import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas

class gmm:
    def __init__(self, clusters, iter, seed=0):
        self.seed = seed
        self.clusters = clusters
        self.iter = iter
        self.m None
        self.sig = None
        self.pi = None
        self.confidence = None

    def trainModel(self, trainee):
        if isinstance(trainee, pandas.DataFrame):
            trainee = trainee.values    
