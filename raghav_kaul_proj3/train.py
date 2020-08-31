# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:46:16 2020

@author: kaulr
"""

#Header Files
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import entropy
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cluster import DBSCAN
from scipy import fftpack
import pickle

#Loading Datasets
my_cols = ["A", "B", "C", "D", "E", "F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","AA","AB","AC","AD","AE"]

#Meal Data
MealData1 = pd.read_csv('mealData1.csv', names = my_cols)
MealData2 = pd.read_csv('mealData2.csv', names = my_cols)
MealData3 = pd.read_csv('mealData3.csv', names = my_cols)
MealData4 = pd.read_csv('mealData4.csv', names = my_cols)
MealData5 = pd.read_csv('mealData5.csv', names = my_cols)

#Meal Amount Data
MealAmountData1 = pd.read_csv('mealAmountData1.csv', names = my_cols)
MealAmountData2 = pd.read_csv('mealAmountData2.csv', names = my_cols)
MealAmountData3 = pd.read_csv('mealAmountData3.csv', names = my_cols)
MealAmountData4 = pd.read_csv('mealAmountData4.csv', names = my_cols)
MealAmountData5 = pd.read_csv('mealAmountData5.csv', names = my_cols)


#Plotting graph of insulin levels
''' 
for i in range(len(MealData1)):
    plt.xlabel("Time Stamp")
    plt.ylabel("Insulin Levels")
    plt.plot( MealData1.iloc[i])
    plt.show()
'''

#Data Pre-processing 
#1.Removing everything after the 30th column 
#Meal Data
MealData1 = MealData1.loc[:, :'AD']
MealData2 = MealData2.loc[:, :'AD']
MealData3 = MealData3.loc[:, :'AD']
MealData4 = MealData4.loc[:, :'AD']
MealData5 = MealData5.loc[:, :'AD']

#Meal Amount Data
MealAmountData1 = MealAmountData1.loc[:, :'A']
MealAmountData2 = MealAmountData2.loc[:, :'A']
MealAmountData3 = MealAmountData3.loc[:, :'A']
MealAmountData4 = MealAmountData4.loc[:, :'A']
MealAmountData5 = MealAmountData5.loc[:, :'A']

#Removing all rows after the 51st Row
MealAmountData1 = MealAmountData1.head(51)
MealAmountData2 = MealAmountData2.head(51)
MealAmountData3 = MealAmountData3.head(51)
MealAmountData4 = MealAmountData4.head(51)
MealAmountData5 = MealAmountData5.head(51)


#2. Reversing both datasets for chronologically accurate picture
#Meal Data
# MealData1 = MealData1.iloc[:, ::-1]
# MealData2 = MealData2.iloc[:, ::-1]
# MealData3 = MealData3.iloc[:, ::-1]
# MealData4 = MealData4.iloc[:, ::-1]
# MealData5 = MealData5.iloc[:, ::-1]

#Checking rows where NaN is present in MealData 
df1 = MealData1
df2 = MealData2
df3 = MealData3
df4 = MealData4
df5 = MealData5

nanindex1 = []
nanindex2 = []
nanindex3 = []
nanindex4 = []
nanindex5 = []

#Omitting rows where NaN and blank values are present (Glucose Levels)
df1 = df1.isnull().any(axis=1)
for i,j in df1.iteritems():
  if (j == True):
    nanindex1.append(i)
    
df2 = df2.isnull().any(axis=1)
for i,j in df2.iteritems():
  if (j == True):
    nanindex2.append(i)
    
df3 = df3.isnull().any(axis=1)
for i,j in df3.iteritems():
  if (j == True):
    nanindex3.append(i)
    
df4 = df4.isnull().any(axis=1)
for i,j in df4.iteritems():
  if (j == True):
    nanindex4.append(i)
    
df5 = df5.isnull().any(axis=1)
for i,j in df5.iteritems():
  if (j == True):
    nanindex5.append(i)

#Printing Indexes where rows have True for NaN
#print(nanindex1)
#print(nanindex2)
#print(nanindex3)
#print(nanindex4)
#print(nanindex5)

#Dropping those rows from MealData 

#Dropping NaN rows from Meal Data 
MealData1 = MealData1.dropna()
MealData2 = MealData2.dropna()
MealData3 = MealData3.dropna()
MealData4 = MealData4.dropna()
MealData5 = MealData5.dropna()

#Dropping corresponding rows in Meal Amount Data
MealAmountData1 = MealAmountData1.drop(MealAmountData1.index[nanindex1]) 
MealAmountData2 = MealAmountData2.drop(MealAmountData2.index[nanindex2]) 
MealAmountData3 = MealAmountData3.drop(MealAmountData3.index[nanindex3]) 
MealAmountData4 = MealAmountData4.drop(MealAmountData4.index[nanindex4]) 
MealAmountData5 = MealAmountData5.drop(MealAmountData5.index[nanindex5])

#Resetting Indexes in Dataframes
#1. Meal Data
MealData1 = MealData1.reset_index(drop = True)
MealData2 = MealData2.reset_index(drop = True)
MealData3 = MealData3.reset_index(drop = True)
MealData4 = MealData4.reset_index(drop = True)
MealData5 = MealData5.reset_index(drop = True)

#2. Meal Amount Data
MealAmountData1 = MealAmountData1.reset_index(drop = True)
MealAmountData2 = MealAmountData2.reset_index(drop = True)
MealAmountData3 = MealAmountData3.reset_index(drop = True)
MealAmountData4 = MealAmountData4.reset_index(drop = True)
MealAmountData5 = MealAmountData5.reset_index(drop = True)

#Concatenating Dataframe into One List
MealList = [MealData1, MealData2, MealData3, MealData4, MealData5]
Mealdf = pd.concat(MealList)

#print("Meal df shape :", Mealdf.shape)

MealAmountList = [MealAmountData1, MealAmountData2, MealAmountData3, MealAmountData4, MealAmountData5]
MealAmountdf = pd.concat(MealAmountList)

#print("Meal Amount shape :", MealAmountdf.shape)


#Discretizing Meal Amount Data into 6 Bins
MealAmountdf.loc[MealAmountdf['A'] == 0, 'Bins'] = 0
MealAmountdf.loc[MealAmountdf['A'] > 0, 'Bins'] = 1
MealAmountdf.loc[MealAmountdf['A'] > 20, 'Bins'] = 2
MealAmountdf.loc[MealAmountdf['A'] > 40, 'Bins'] = 3
MealAmountdf.loc[MealAmountdf['A'] > 60, 'Bins'] = 4
MealAmountdf.loc[MealAmountdf['A'] > 80, 'Bins'] = 5

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
updatedfeatureMatrix['COV'] = feature_COV
updatedfeatureMatrix

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

fftdf = MealDatacopy.values
for i in range(len(fftdf)):
    #print(i)
    fftvalue = abs(fftpack.fft(fftdf[0]))
#print(fftvalue)
    
################PROVIDING FEATURE MATRIX TO PCA ###########################
#Standardizing feature matrix
#updatedfeatureMatrix = StandardScaler().fit_transform(updatedfeatureMatrix)

#Taking Top 5 Components for PCA
pca = PCA(n_components = 5)
principalComponents = pca.fit(updatedfeatureMatrix)
#print(principalComponents.components_)

principalComponentsTrans = pca.fit_transform(updatedfeatureMatrix)
pc5Matrix = pd.DataFrame(data = principalComponentsTrans, columns = ['Principal Component 1', 'Principal Component 2','Principal Component 3', 'Principal Component 4','Principal Component 5'])
pc5Matrix    
    

#Applying KMeans Clustering Algorithm
km = KMeans(
    n_clusters=6, init='random',
    n_init=10, max_iter=300, 
   tol=1e-04, random_state=0
)

#km = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, tol=1e-04,
#                       precompute_distances='auto', 
#                       verbose=0, 
#                       random_state = 0 , 
#                       copy_x=True, 
#                       n_jobs=None, 
#                       algorithm='auto'
#                       )

y_km = km.fit_predict(pc5Matrix)
#print(y_km)
#print("SSE (KMeans) :", km.inertia_) 

#Setting Ground Truth for Calculation of Accuracy
groundtruth = MealAmountdf['Bins'].values
#print(groundtruth)

count = 0
for i in range(len(y_km)):
    if(y_km[i] == int(groundtruth[i])):
        count += 1

acc = count / len(y_km) * 100       

#print("Count :", count)
print("Classification Accuracy (KMeans) :", acc)

#print(groundtruth)
#Calculating accuracy for KMeans 
#for i in range(len(y_km)):
    


#Implementing KFold Cross Validation
kf = KFold(n_splits = 20, shuffle = True) 
'''
for train_index, test_index in kf.split(X):
    #print("Train:", train_index, "Validation:",test_index)
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index]

    svclassifier.fit(X_train, y_train)

    #p = svclassifier.predict(X_test)

    matrix = matrix + confusion_matrix(y_test, p)
    pScore.append(precision_score(y_test, p))
    recall.append(recall_score(y_test, p))
    F1Score.append(f1_score(y_test, p))
'''

#Calculating Accuracy from Ground Truth
#match = 0
#for i in range(len(MealAmountdf)):
#    print(MealAmountdf.loc)

#Mapping cluster labels to bins
#Applying DBSCAN Clustering Algorithm
db = DBSCAN(eps = 0.3, min_samples = 2, algorithm = "ball_tree", n_jobs = 3).fit(pc5Matrix)
y_label = db.labels_    
#print(y_label)
    
#Setting Ground Truth for Calculation of Accuracy
#groundtruth = MealAmountdf['Bins'].values
#print(groundtruth)

#Implementing KFold Cross Validation
kf = KFold(n_splits = 20, shuffle = True) 
'''
for train_index, test_index in kf.split(X):
    #print("Train:", train_index, "Validation:",test_index)
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index]

    svclassifier.fit(X_train, y_train)

    #p = svclassifier.predict(X_test)

    matrix = matrix + confusion_matrix(y_test, p)
    pScore.append(precision_score(y_test, p))
    recall.append(recall_score(y_test, p))
    F1Score.append(f1_score(y_test, p))
'''


count = 0
for i in range(len(y_label)):
    if(y_label[i] == int(groundtruth[i])):
        count += 1

acc = count / len(y_label) * 100
#print(db.intertia_)
#print("Count :", count)
print("Classification Accuracy (DBSCAN)", acc)

#Saving model in Pickle files 
filename1 = 'finalized_model1.pkl'
pickle.dump(km, open(filename1, 'wb'))

filename2 = 'finalized_model2.pkl'
pickle.dump(db, open(filename2, 'wb'))

#for i in len(range(y_km)):
#    y_km[i] =+ 1
    
#print("New KMeans : ", y_km)
    
