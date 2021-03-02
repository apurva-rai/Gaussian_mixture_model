import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas

class gmm:
    def __init__(self, clusters, iter, randSeed=0):
        self.randSeed = randSeed
        self.clusters = clusters
        self.iter = iter
        self.u None
        self.sig = None
        self.pi = None
        self.confidence = None

    #Pass an instance of the data set to train the current instance of GMM initialization
    def trainModel(self, trainee):
        if isinstance(trainee, pandas.DataFrame):
            trainee = trainee.values

        #Calculate variable values
        np.random.seed(self.randSeed)
        randRow = np.random.choice(trainee.shape[0], self.clusters, replace = False)
        self.pi = np.ones(self.clusters)/self.clusters
        self.confidence = np.ones(trainee.shape)/self.clusters
        self.u = [trainee[i] for i in randRow]
        self.sig = [np.cov(np.transpose(trainee).astype(float)) for _ in range(self.clusters)]

        for _ in range(clusters):
            self.expected(trainee)
            self.maximizer(trainee)

        return self
   
