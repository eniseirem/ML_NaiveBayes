import pandas as pd
import numpy as np

#from sklearn import preprocessing #todo: add sklearn later

#I'm adding these too avoid any type of na value.
missing_values = ["n/a", "na", "--", " ?","?"]
names=[] #todo: add names of variables
#reading the data
data = pd.read_csv('data/abalone.data', na_values=missing_values)

#checking the dataset
print(data.tail)