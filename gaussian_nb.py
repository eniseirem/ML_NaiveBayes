#
# 1) Assume gaussian distribution for continuous features. Report the accuracies for each of the following case:
#
#   1.1) 100 samples for training, and rest for validation set
#   1.2) 1000 samples for training, and rest for validation set


import preprocess as pre

df= pre.data

#print(df.tail)

X = df.iloc[:,1:-1] #continous
#print(X)
y = df['Rings'] # +1.5 gives the age

d_1=pre.tt(X,y,100) #first case
d_2=pre.tt(X,y,1000) #second case


from naive_bayes import NaiveBayes

nb = NaiveBayes()
nb.fit(d_1["X_train"],d_1["y_train"])