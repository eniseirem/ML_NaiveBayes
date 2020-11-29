#
# 1) Assume gaussian distribution for continuous features. Report the accuracies for each of the following case:
#
#   1.1) 100 samples for training, and rest for validation set
#   1.2) 1000 samples for training, and rest for validation set


import preprocess as pre

df= pre.data

#print(df.tail)

X = df.iloc[:,1:-1] #from length to shell-w
y = df[['Sex','Rings']]

d_1=pre.tt(X,y,100) #first case
d_2=pre.tt(X,y,1000) #second case

print(d_1);
