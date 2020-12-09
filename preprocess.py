import pandas as pd
import numpy as np

#I'm adding these too avoid any type of na value.
missing_values = ["n/a", "na", "--", " ?","?"]

    # Name		Data Type	Meas.	Description
	# ----		---------	-----	-----------
	# Sex		nominal			M, F, and I (infant)
	# Length		continuous	mm	Longest shell measurement
	# Diameter	continuous	mm	perpendicular to length
	# Height		continuous	mm	with meat in shell
	# Whole weight	continuous	grams	whole abalone
	# Shucked weight	continuous	grams	weight of meat
	# Viscera weight	continuous	grams	gut weight (after bleeding)
	# Shell weight	continuous	grams	after being dried
	# Rings		integer			+1.5 gives the age in years

names=["Sex", "Length", "Diameter", "Height", "Whole-w", "Shucked-w","Viscera-w", "Shell-w", "Rings"]
#reading the data
data = pd.read_csv('data/abalone.data', names=names,na_values=missing_values)
#data= pd.notnull(data)

#checking the dataset
#print(data.tail)
#print(data.shape)

#I'll split the data here since both of the questions needs same type of it.

from sklearn.model_selection import train_test_split

data = data.sample(frac=1, random_state=42) #mixing database for more random sample

#let's label the data with its ages according to
#output: class label
#(less than 8 in age belongs to class 1 (young), between 8 and 12 to class 2 (middle-aged), greater than 12 to class 3 (old))
#rings +1.5 gives the age so lets update the data accordingly

col_name='Rings'
conditions = [
    data[col_name] <  8- float(1.5), #adding 1,5 will give the age so doing like that will work as well
    data[col_name].between(6.5,(12-1.5)),
    12-1.5 < data[col_name]

]
labels = ["young","middle-aged","old"]

data['age']=np.select(conditions,labels)
data.drop("Rings", axis=1, inplace=True)

def tt(X,y, sample):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=sample,  random_state=1) #for calling from

    return {"X_train":X_train, "X_valid":X_valid, "y_train":y_train, "y_valid":y_valid}