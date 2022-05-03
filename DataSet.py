import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.utils import resample, shuffle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#CHANGE BATCH_SIZE in MAIN TO 7

class DataSet:
    def __init__(self):
        self.loadDataSet = fetch_covtype(as_frame=True, download_if_missing=True, data_home="H:/USB_Backup/SchoolStuff/Skovde Computer Science/Examens Arbete/Results/ALL DATASETS")
        self.cov = self.loadDataSet.frame
        self.X = self.loadDataSet.data
        self.Y = self.cov.Cover_Type
        self.standardX = preprocessing.scale(self.X)
        self.df = pd.DataFrame(self.standardX, columns=self.loadDataSet.feature_names)
       # self.normalize = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        self.df['target'] = self.cov.Cover_Type
        self.target_1 = self.df[self.df['target'] == 1]
        self.target_2 = self.df[self.df['target'] == 2]
        self.target_3 = self.df[self.df['target'] == 3]
        self.target_4 = self.df[self.df['target'] == 4]
        self.target_5 = self.df[self.df['target'] == 5]
        self.target_6 = self.df[self.df['target'] == 6]
        self.target_7 = self.df[self.df['target'] == 7]
        self.target_1_resample = resample(self.target_1, n_samples=50, random_state=0, replace=False)
        self.target_2_resample = resample(self.target_2, n_samples=50, random_state=0, replace=False)
        self.target_3_resample = resample(self.target_3, n_samples=50, random_state=0, replace=False)
        self.target_4_resample = resample(self.target_4, n_samples=50, random_state=0, replace=False)
        self.target_5_resample = resample(self.target_5, n_samples=50, random_state=0, replace=False)
        self.target_6_resample = resample(self.target_6, n_samples=50, random_state=0, replace=False)
        self.target_7_resample = resample(self.target_7, n_samples=50, random_state=0, replace=False)
        self.df_resample = pd.concat([self.target_1_resample, self.target_2_resample, self.target_3_resample,
                                      self.target_4_resample, self.target_5_resample, self.target_6_resample,
                                      self.target_7_resample])
        self.df_resample = shuffle(self.df_resample, random_state=1)

        #self.df_resample = self.df_resample[self.df_resample.columns].fillna(0)
        self.dataUnBatched = np.array(self.df_resample)
        batch_size = 50
        self.batchedData = np.vsplit(self.dataUnBatched, batch_size)
        self.train, self.test = train_test_split(self.batchedData, test_size=0.1)


    @staticmethod
    def getAllTargets():
        result = []
        for x in range(len(ds.train)):
            for i in range(len(ds.train[x])):
                targetArray = []
                for y in range(10):
                    if y == ds.train[x][i][len(ds.train[x][i]) -1]:
                        targetArray.append(1)
                    else:
                        targetArray.append(0)
                result.append(targetArray)
        return result

    @staticmethod
    def getTestInputRow(batchIndex ,rowIndex):
        rows = ds.test[batchIndex][rowIndex]
        return rows

    @staticmethod
    def getBatchTrainInputRow(batchIndex ,rowIndex):
        rows = ds.train[batchIndex][rowIndex]
        return rows

    @staticmethod
    def getTestTarget(batchIndex, rowIndex, target_classes):
        testArray = []
        for x in range(target_classes):
            if x == ds.test[batchIndex][rowIndex] [(len(ds.test[batchIndex][rowIndex]) -1)] :
                testArray.append(1)
            else:
                testArray.append(0)
        return testArray

    @staticmethod
    def getTrainBatchTarget(batchIndex, rowIndex, target_classes):
        trainArray = []
        for x in range(target_classes):
            if x == ds.train[batchIndex][rowIndex] [(len(ds.train[batchIndex][rowIndex]) -1)] :
                trainArray.append(1)
            else:
                trainArray.append(0)
        return  trainArray

    @staticmethod
    def getTestAmount():
        return len(ds.test)

    @staticmethod
    def getBatchTrainAmount():
        return len(ds.train)

    @staticmethod
    def createGraph(data,lables, yLabel, epoch_amount):
        X = []
        for x in range (epoch_amount):
            X.append(x)
        plt.xlabel("iteration")
        plt.ylabel(yLabel)
        #data contains both arrays
        plt.plot(X, data,  label = lables)
        plt.legend()
        return plt.show()

ds = DataSet()
