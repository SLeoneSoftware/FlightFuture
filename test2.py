import numpy
import os
import sqlite3
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

#Obtain and clean flight data from csv file
def get_data(file='flights.csv',path='flight-delays',seed=1067748903,datatype=str):
	data = numpy.genfromtxt(os.path.join(path,file),delimiter=',',dtype=datatype)
	num_x = data[:,[ 0,1,2,3,5,9, 14, 17, 20]]
	alph_x = data[:,[4,6,7,8]]
	enc = preprocessing.OrdinalEncoder()
	enc.fit(alph_x)
	alph_x = enc.transform(alph_x)
	#For simplicity in dealing with values, all features will be returned to original position
	x = numpy.hstack((num_x[:,[ 0,1,2,3]], alph_x[:,[0]], num_x[:,[4]], alph_x[:,[1,2,3]], num_x[:,[ 5,6,7,8]]))
	#This project only needs to predict prospects of flight cancellation/delays , Arrival Time, and Departure Time
	y = data[:,[21, 23,24]]
	return x, y

x, y = get_data()
print(x)


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