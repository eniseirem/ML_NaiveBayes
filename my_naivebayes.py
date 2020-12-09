#Bayes relies on conditional probability

import numpy as np
import math
import pandas as pd

class NaiveBayes:
    def fit(self, X, y): #this will be fitting the training. Splitted dataset to X data matrice and y target column.
        i_val, i_fea =X.shape
        classes = y.unique()
        i_classes = len(classes) #find the lenght to fit in
        X = self.separate(X,y)

        means = np.zeros((i_classes, i_fea),dtype=np.float64) #for each feature we'll get mean
        var = np.zeros((i_classes, i_fea),dtype=np.float64) #variences
        pri = np.zeros(i_classes,dtype=np.float64) #priors
        std = np.zeros((i_classes, i_fea), dtype=np.float64) #standar deviations
        labels = {}
        keys = {}
        for c, val  in enumerate(classes): #foreach i_classes
            X_c = np.array(X[val]) #the class has labels
            means[c, :] = np.mean(X_c,axis=0) #means
            var[c, :] = np.var(X_c,axis=0) #variences
            pri[c] = X_c.shape[0] / float(i_val) #frequency of class, prior probability
            std[c, :] = np.std(X_c, axis=0) #standard deviation for gaussian
            labels[val] = c
            keys[c] = val
        # with that order 0 == middle-aged, 1 == old, 2 == young

       #which the values actually stated in the data files as ;

       # 	Length	Diam	Height	Whole	Shucked	Viscera	Shell	Rings
       # Min	0.075	0.055	0.000	0.002	0.001	0.001	0.002	    1
       # Max	0.815	0.650	1.130	2.826	1.488	0.760	1.005	   29
       # Mean	0.524	0.408	0.140	0.829	0.359	0.181	0.239	9.934
       # SD	0.120	0.099	0.042	0.490	0.222	0.110	0.139	3.224
       # Correl	0.557	0.575	0.557	0.540	0.421	0.504	0.628	  1.0

        #determining the variables which we will need outside of the fit function

        self._classes = classes
        self._pri = pri
        self._means = means
        self._var = var
        self._stdev = std
        self._labs = labels
        self._keys = keys

    def separate(self,X,y): #separate by class
        sep = dict()
        idx = 0
        for key, val in X.iterrows():
            class_idx = y.iloc[idx]
            if (class_idx not in sep):
                sep[class_idx] = list()
            sep[class_idx].append(val)
            idx = idx +1
        return sep

    def accuracy(self, prediction, y_valid):
        tot = 0
        for val, i in enumerate(y_valid):
            if i == prediction[val]:
                tot +=1
        return tot / float(len(y_valid))


    def gaussian_dist(self,class_val,x):
        class_val = self._labs[class_val]
        mean = self._means[class_val]
        stdev = self._stdev[class_val]
        res = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        den = np.sqrt(2 * np.pi * stdev)
        return 1 / den * res

    def train_gau(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for val, c in enumerate(self._classes):
            prior = np.log(self._pri[val])
            condt = np.sum(np.log(self.gaussian_dist(c, x))) #get log to prevent overflow
            posterior = prior + condt
            posteriors.append(posterior)
        return np.argmax(posteriors) #argmax to find highest posterior probability

    def predict_gau(self, X, fit_x, fit_y):
        self.fit(fit_x, fit_y)
        y_pred = [self.train_gau(x) for idx, x in X.iterrows()]
        y_pred_labelled = [self._keys.get(x) for x in y_pred] #labelling with strings
        return np.array(y_pred_labelled)

    def cf_matrix(self, y_valid, y_predicted):  # Precision = TP/(FP+TP

        y_actu = pd.Series(y_valid, name='Actual')
        y_pred = pd.Series(y_predicted, name='Pred')
        conf_matrix = pd.crosstab(y_actu, y_pred)

        return conf_matrix

