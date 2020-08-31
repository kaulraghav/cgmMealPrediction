README
--------
 
1.	The zip file contains two python files, one for training the model (train.py) and one for testing the model (test.py) on an unknown dataset. 
2.	The model is trained in the train.py file and stored as a pickle file (finalized_model.pkl). This file is fed to test.py as a trained model and a corresponding output for the meal/no-meal (1/0) data is generated. 
3.	Ensure that both train.py and test.py are present in the same working directory along with the training and testing datasets
4.	When train.py is run, it takes in the 10 provided datasets of Meal and No Meal Data as input and trains a model and returns the corresponding accuracy, f1 score, precision and recall of the model. A pickle file (finalized_model.pkl) is also generated and stored in the same working directory. 
5.	Before running test.py, the testing dataset must be manually hardcoded in place of testdata.csv (line 20). 
6.	When test.py is run, the pre-trained model is loaded (Ensure all the files are in the same working directory) and the predicted class labels are outputted.  
