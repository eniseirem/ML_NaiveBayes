#
# 1) Assume gaussian distribution for continuous features. Report the accuracies for each of the following case:
#
#   1.1) 100 samples for training, and rest for validation set
#   1.2) 1000 samples for training, and rest for validation set


import preprocess as pre

df= pre.data

#print(df.tail)

X = df.iloc[:,1:-1] #continous
y = df["age"]

#print(df.tail)
d_1=pre.tt(X,y,100) #first case
d_2=pre.tt(X,y,1000) #second case


from my_naivebayes import NaiveBayes

nb = NaiveBayes()

L = nb.predict_gau(d_1["X_valid"],d_1["X_train"],d_1["y_train"])
acc = nb.accuracy(L,d_1["y_valid"])
conf = nb.cf_matrix(d_1["y_valid"], L)

L2 = nb.predict_gau(d_2["X_valid"],d_2["X_train"],d_2["y_train"])
acc2 = nb.accuracy(L,d_2["y_valid"])
conf2 = nb.cf_matrix(d_2["y_valid"], L2)

#printing :)
print("Gaussian Distribution Naive Bayes")
print("---------------------------------")
print("Case 1 : 100 Sample Training")
print("Accuracy : ", acc)
print(conf)
print()
print("Case 2 : 1000 Sample Training")
print("Accuracy : ", acc2)
print(conf2)

#I will do the regressogram part here