

import numpy
import os
import sqlite3
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier


def get_data(file='flights.csv',path='flight-delays',seed=1067748903,datatype=str):
	data = numpy.genfromtxt(os.path.join(path,file),delimiter=',',dtype=datatype)
	x = data[:,[ 0,1,2,3,4,5,6,7,8,9,13,16,19,]]
	y = data[:,[22,23]]
	return x, y

x, y = get_data()
print(x)


#Data By Column
'''
1 Year
2 Month
3 Day
4 Day of Week
5 Airline
6 Flight Number
7 Tail Number
8 Origin Airport
9 Destination Airport
10 Scheduled Departure
11 Departure Time
12 Departure Delay
13 Taxi Out
14 Wheels Off
15 Planned Trip Time
16 AIR_TIME+TAXI_IN+TAXI_OUT
17 The time duration between wheels_off and wheels_on time (air time)
18 Distance between two airports
19 Wheels On
20 Taxi in
21 Scheduled Arrival
22 Arrival Time
23 Arrival Delay
24 Diverted
25 Cancelled
26 Cancellation Reason
27 Air System Delay
28 Security Delay
29 Airline Delay
30 Late Aircraft Delay
31 Weather Delay
'''