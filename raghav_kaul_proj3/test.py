# -*- coding: utf-8 -*-
"""
@author: kaulr
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import entropy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#Loading Dataset
my_cols = ["A", "B", "C", "D", "E", "F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","AA","AB","AC","AD","AE"]

#Meal Data
################### ENTER TEST CSV FILE #########################
MealData1 = pd.read_csv('proj3_test.csv', names = my_cols) ######
#################################################################

#Data Pre-processing 
#1.Removing everything after the 30th column 
#Meal Data
MealData1 = MealData1.loc[:, :'AD']

#Checking rows where NaN is present in MealData 
df1 = MealData1

#Dropping NaN rows from Meal Data 
MealData1 = MealData1.dropna()

#Resetting Indexes in Dataframes
#1. Meal Data
MealData1 = MealData1.reset_index(drop = True)

Mealdf = MealData1

#Scaling
scaler = MinMaxScaler(feature_range=(0,1))
MealdfScaled = scaler.fit_transform(Mealdf)
MealdfScaled = pd.DataFrame(MealdfScaled)

MealDatacopy = MealdfScaled

#Feature Extraction
#1. Moving Average 
updatedfeatureMatrix = pd.DataFrame()
#Calculting averages for discrete 30 minute intervals 
meanFrame = pd.DataFrame()
for i in range(0,30,6):
  meanFrame['Mean ' + str(i)+"-"+str(i + 6)] = MealDatacopy.iloc[:, i :i + 6].mean(axis = 1)
#meanFrame

#Inserting features in feature matrix
updatedfeatureMatrix['mean0-6']= meanFrame['Mean 0-6']
updatedfeatureMatrix['mean6-12']= meanFrame['Mean 6-12']
updatedfeatureMatrix['mean12-18']= meanFrame['Mean 12-18']
updatedfeatureMatrix['mean18-24']= meanFrame['Mean 18-24']
updatedfeatureMatrix['mean24-30']= meanFrame['Mean 24-30']

#Displaying updated feature matrix
updatedfeatureMatrix

#2.Maximum difference [30-Minute Intervals]
#Calculate maximum difference of each row 
velocityFrame = pd.DataFrame() 
for i in range(0,25):
  velocityFrame['Velocity '+ str(i+1)+"-"+ str(i+5)] = ((MealDatacopy.iloc[: , i + 5]) - (MealDatacopy.iloc[: , i]))
#velocityFrame

#Inserting features in the feature matrix
updatedfeatureMatrix['maximumVelocity']= velocityFrame.max(axis = 1)
updatedfeatureMatrix

#3.Entropy 
#Function to calculate entropy of each row
def calculateEntropy(series):
    numberofSeries = series.value_counts()
    entropyvalues = entropy(numberofSeries)  
    return entropyvalues

entropyTest = pd.DataFrame()
entropyTest['Entropy'] = MealDatacopy.apply(lambda row: calculateEntropy(row), axis=1) 
#entropyTest

#Inserting values in updated feature matrix
updatedfeatureMatrix['Entropy'] = MealDatacopy.apply(lambda row: calculateEntropy(row), axis=1) 
updatedfeatureMatrix

#4. Feature of Covariation 
feature_COV = pd.DataFrame()
feature_COV["COV"] = MealDatacopy.mean(axis= 1) / MealDatacopy.std(axis= 1)
#feature_COV

#Adding COV into updated feature matrix 
#updatedfeatureMatrix['COV'] = feature_COV
#updatedfeatureMatrix

#5. Fast Fourier Transform
#Inserting FFT values in the feature matrix
#Function to calculate top 5 fft values 
'''
def get_fft(row):
    cgmFFTValues = abs(fftpack.fft(row))
    cgmFFTValues.sort()
    return np.flip(cgmFFTValues)[0:8]

FFT = pd.DataFrame()
FFT['FFT_Top2'] = fftdf.apply(lambda row: get_fft(row), axis=1)
FFT_updated = pd.DataFrame(FFT.FFT_Top2.tolist(), columns=['FFT_1', 'FFT_2', 'FFT_3', 'FFT_4', 'FFT_5', 'FFT_6', 'FFT_7', 'FFT_8'])

#FFT_updated.head()

#updatedfeatureMatrix.join(FFT_updated)
'''
#fftvalue = abs(fftpack.fft(fftdf[1]))
'''
fftdf = MealDatacopy.values
for i in range(len(fftdf)):
    #print(i)
    fftvalue = abs(fftpack.fft(fftdf[0]))
#print(fftvalue)
    '''
################PROVIDING FEATURE MATRIX TO PCA ###########################
#Standardizing feature matrix
#updatedfeatureMatrix = StandardScaler().fit_transform(updatedfeatureMatrix)

#Taking Top 5 Components for PCA
pca = PCA(n_components = 5)
principalComponents = pca.fit(updatedfeatureMatrix)
principalComponentsTrans = pca.fit_transform(updatedfeatureMatrix)
pc5Matrix = pd.DataFrame(data = principalComponentsTrans, columns = ['Principal Component 1', 'Principal Component 2','Principal Component 3', 'Principal Component 4','Principal Component 5'])
pc5Matrix    
#print(principalComponents.components_)

#Loading Testing dataset into X
X = pc5Matrix.iloc[:, 0:5].values
#print(X)

#Load the trained model using pickle
#ENSURE TRAIN AND TEST FILES ARE IN THE SAME FOLDER!
#Load model for kmeans 
loaded_model = pickle.load(open("finalized_model1.pkl", 'rb'))
result = loaded_model.predict(X)

#Displaying cluster labels for KMeans
print(result)

X1 = pc5Matrix.iloc[:, 0:5].values

#Load model for DBSCAN
loaded_model1 = pickle.load(open("finalized_model2.pkl", 'rb'))
result1 = loaded_model1.fit(X1)
result2 = result1.labels_ 

#Displaying cluster labels for DBSCAN
print(result2)

 