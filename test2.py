import numpy
import os
import sqlite3
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

#Obtain and clean flight data from csv file
def get_data(labels, file='flights-sample.csv',path='data',seed=1067748903,datatype=str):
	data = numpy.genfromtxt(os.path.join(path,file),delimiter=',',dtype=datatype)
	#Shuffle Data
	numpy.random.shuffle(data)
	num_x = data[:,[ 0,1,2,3,5,9, 14, 17, 20]]
	alph_x = data[:,[4,6,7,8]]
	enc = preprocessing.OrdinalEncoder()
	enc.fit(alph_x)
	alph_x = enc.transform(alph_x)
	#For simplicity in dealing with values, all features will be returned to original position
	x = numpy.hstack((num_x[:,[ 0,1,2,3]], alph_x[:,[0]], num_x[:,[4]], alph_x[:,[1,2,3]], num_x[:,[ 5,6,7,8]]))
	y = data[:,[labels]]
	#Convert str arrays to float arrays
	x = x.astype(numpy.float)
	y = y.astype(numpy.float)
	train_x = x[0:800, :]
	train_y = y[0:800, :]
	test_x = x[801:998, :]
	test_y = y[801:998, :]
	return train_x, train_y, test_x, test_y

#Returns total accurate predictions / total labels
def get_accuracy(labels, predictions):
	total = len(labels)
	if not total == len(predictions):
		return 0
	else:
		accuracy = sum(1 for (x,y) in zip(labels, predictions) if x == y)/total
		return accuracy

#Obtain results from a model with hyperparameters
def get_predictions(model, train_x, train_y, test_x):
	model.fit(train_x, train_y)
	predictions = model.predict(test_x)
	return predictions

#Print Confusion Matrix and Accuracy
def accuracy_insights(labels, predictions):
	cm = confusion_matrix(predictions, labels)
	print(cm)
	print(get_accuracy(labels, predictions))

#Grid Search over a model to obtain best hyperparameters
def grid_search(model, param_dict=None):
	if param_dict is None:
		print("Add implementation here later to support taking any model")
	else:
		clf = GridSearchCV(model, param_dict, cv=StratifiedKFold(n_splits=3, shuffle=False))
		clf.fit(train_X, train_Y)



train_x, train_y, test_x, test_y = get_data(24)
accuracy_insights(numpy.ravel(test_y), get_predictions(MLPClassifier(), train_x, numpy.ravel(train_y), test_x))



#Data By Column: Index, Numeric vs Alphabetic, Independent vs Dependent, Attribute Name
'''
0  # X Year 
1  # X Month 
2  # X Day 
3  # X Day of Week 
4  A X Airline 
5  # X Flight Number 
6  A X Tail Number 
7  A X Origin Airport 
8  A X Destination Airport 
9  # X Scheduled Departure 
10 # Y Departure Time 
11 # Y Departure Delay 
12 # Y Taxi Out 
13 # Y Wheels Off 
14 # X Planned Trip Time 
15 # Y AIR_TIME+TAXI_IN+TAXI_OUT 
16 # Y The time duration between wheels_off and wheels_on time (air time) 
17 # X Distance between two airports 
18 # Y Wheels On 
19 # Y Taxi in 
20 # X Scheduled Arrival 
21 # Y Arrival Time 
22 # Y Arrival Delay 
23 # Y Diverted 
24 # Y Cancelled 
25 A Y Cancellation Reason
26 # Y Air System Delay
27 # Y Security Delay
28 # Y Airline Delay
29 # Y Late Aircraft Delay
30 # Y Weather Delay
'''