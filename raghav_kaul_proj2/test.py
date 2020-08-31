import pandas as pd
import numpy as np
import pickle
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy.stats import entropy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#Loading Datasets

my_cols = ["A", "B", "C", "D", "E", "F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","AA","AB","AC","AD","AE"]

###########################################################################################################
#Enter test csv in the following file path (THE TEST DATASET SHOULD BE IN THE SAME DIRECTORY AS TRAIN.PY) #
MealDatacopy = pd.read_csv('TestData.csv', names = my_cols)                                               #
###########################################################################################################

#Data Pre-processing 
#1.Removing everything after the 30th column 
MealDatacopy = MealDatacopy.loc[:, :'AD']

#2. Reversing both datasets for chronologically accurate picture
MealDatacopy = MealDatacopy.iloc[:, ::-1]

#Removing rows with NaN and Empty elements 

#Meal Data
MealDatacopy = MealDatacopy.dropna()
#print(MealDatacopy)

#Extracting Features and creating Feature Matrix
#1. Moving Average 
updatedfeatureMatrix = pd.DataFrame()
#Calculting averages for discrete 30 minute intervals 
meanFrame = pd.DataFrame()
for i in range(0,30,6):
  meanFrame['Mean ' + str(i)+"-"+str(i + 6)] = MealDatacopy.iloc[:, i :i + 6].mean(axis = 1)
#print(meanFrame)

#Inserting features in feature matrix
updatedfeatureMatrix['mean0-6']= meanFrame['Mean 0-6']
updatedfeatureMatrix['mean6-12']= meanFrame['Mean 6-12']
updatedfeatureMatrix['mean12-18']= meanFrame['Mean 12-18']
updatedfeatureMatrix['mean18-24']= meanFrame['Mean 18-24']
updatedfeatureMatrix['mean24-30']= meanFrame['Mean 24-30']

#Displaying updated feature matrix
#print(updatedfeatureMatrix)

#2.Maximum difference [30-Minute Intervals]

#Calculate maximum difference of each row 
velocityFrame = pd.DataFrame() 
for i in range(0,25):
  velocityFrame['Velocity '+ str(i+1)+"-"+ str(i+5)] = ((MealDatacopy.iloc[: , i + 5]) - (MealDatacopy.iloc[: , i]))
#print(velocityFrame)

#Inserting features in the feature matrix
updatedfeatureMatrix['maximumVelocity']= velocityFrame.max(axis = 1)
#print(updatedfeatureMatrix)

#3.Entropy 
#Function to calculate entropy of each row
def calculateEntropy(series):
    numberofSeries = series.value_counts()
    entropyvalues = entropy(numberofSeries)  
    return entropyvalues

entropyTest = pd.DataFrame()
entropyTest['Entropy'] = MealDatacopy.apply(lambda row: calculateEntropy(row), axis=1) 
#print(entropyTest)

#Inserting values in updated feature matrix
updatedfeatureMatrix['Entropy'] = MealDatacopy.apply(lambda row: calculateEntropy(row), axis=1) 
#print(updatedfeatureMatrix)

#4. Feature of Covariation 

feature_COV = pd.DataFrame()

feature_COV["COV"] = MealDatacopy.mean(axis= 1) / MealDatacopy.std(axis= 1)

#print(feature_COV)

#Adding COV into updated feature matrix 
updatedfeatureMatrix['COV'] = feature_COV
#print(updatedfeatureMatrix)

################PROVIDING FEATURE MATRIX TO PCA ###########################
#Standardizing feature matrix
updatedfeatureMatrix = StandardScaler().fit_transform(updatedfeatureMatrix)

#Taking Top 5 Components for PCA
pca = PCA(n_components = 5)
principalComponents = pca.fit(updatedfeatureMatrix)
#print(principalComponents.components_)

principalComponentsTrans = pca.fit_transform(updatedfeatureMatrix)
pc5Matrix = pd.DataFrame(data = principalComponentsTrans, columns = ['Principal Component 1', 'Principal Component 2','Principal Component 3', 'Principal Component 4','Principal Component 5'])
#print(pc5Matrix)

#Loading Testing dataset into X
X = pc5Matrix.iloc[:, 0:5].values
#print(X)

#Load the trained model using pickle
#ENSURE TRAIN AND TEST FILES ARE IN THE SAME FOLDER!
loaded_model = pickle.load(open("finalized_model.pkl", 'rb'))
result = loaded_model.predict(X)

#Displaying 1's or 0's according to Meal or Non Meal Data
print(result)
