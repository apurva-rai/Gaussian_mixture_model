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

    #Given the covariance matrix and centroid the probability for each class class can be calculated and normalized.
    #The conditional probability is the multivariate normal distribution
    def expected(self, trainee):
       prob = np.zeros((trainee.shape[0], self.clusters))

       for x in range(self.clusters):
           prob[:,c] = multivariate_normal.pdf(trainee,self.u[x],self.sig[x])

       self.confidence = prob * self.pi / np.sum(prob * self.pi, axis = 1, keepdims = True)
       self.pi = self.confidence.mean(axis = 0)

       return self

    #The weights of a given matrix with trainee observations which come from a multivariate distribution are used.
    #There is no learning rate or gradients as the following function is already maximal.
    def maximizer(self, trainee):

       for x in range (self.clusters):
           confidence = self.confidence[:,[x]]
           totalConfidence = self.confidence[:,[x]].sum()
           self.u[x] = (trainee * confidence).sum(axis=0) / totalConfidence
           self.sig[x]] = np.cov(np.transpose(trainee).astype(float), aweights = (confidence / totalConfidence).flatten(), bias = True)

       return self

    #Calculates the prediction probailitie and returns the max along the 1st axis
    def predClass(self, trainee):
        prob = np.zeros((trainee.shape[0], self.clusters))

        for x in range(self.clusters):
            prob[:,c] = multivariate_normal.pdf(trainee, self.u[x], self.sig[x])

        self.confidence = prob * self.pi / np.sum(prob * self.pi, axis = 1, keepdims = True)

        return np.argmax(self.confidence, axis = 1)

    #Plots the data along with the contour lines. Does not account for more than 6 clusters as the data sets I picked do not need that many.
    def draw(self, trainee, u, sig, xAxis="X-axis", yAxis="Y-axis", title="Gaussian Mixture Model countor map"):
        x, y = np.meshgrid(np.sort(trainee[:,0]), np.sort([trainee[:,1]]))
        xy = np.array([x.flatten(), y.flatten()]).T
        figure = plt.figure(figsize=(10,10))

        x0 = fig.add_subplot(111)
        x0.scatter(trainee[:,0], trainee[:,1])
        x0.set_title(title)

        x0.set_xlabel(xAxis)
        x0.set_ylabel(yAxis)

        colors = ['red','black','yellow','green','magenta','cyan']

        #Does not account for more than 6 clusters. (One could implement a simple color randomizer for that)
        for i in range(self.clusters):
            x0.contour(np.sort(trainee[:,0]), np.sort(trainee[:,1]), multivariate_normal.pdf(xy, mean = u[i], cov = sig[i]).reshape(len(trainee), len(trainee)), colors='black')
            x0.scatter(u[i][0], u[i][1], c = 'grey', zorder=10, s=100)

        plt.show()

    def findInflections(trainee):
        inflections = []

        for x in range(len(trainee[0])):
            colVals = [rows[i] for rows in trainee]
            minVal = min(colVals)
            maxVal = max(colVals)

            inflections.append([minVal, maxVal])

        return inflections

    def normalizeSet(trainee):

        if isinstance(trainee. pandas.DataFrame):
            trainee = trainee.values

        inflections = gmm.findInflections(trainee)

        for row in trainee:
            for col in range(len(row)):
                row[col] = (row[col] - inflections[col][0]) / (inflections[col][1] - inflections[col][0])

        return trainee            
