#This will move all data into a sqlite3 database, making it much easier to query

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
