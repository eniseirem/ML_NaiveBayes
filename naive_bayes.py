#Bayes relies on conditional probability

import numpy as np
import math

class NaiveBayes:
    def fit(self, X, y): #this will be fitting the training. Splitted dataset to X data matrice and y target column.
        i_val, i_fea =X.shape
        classes = list(X)
        i_classes = len(classes) #find the lenght to fit in

        means = np.zeros((i_classes, i_fea),dtype=np.float64) #for each feature we'll get mean
        var = np.zeros((i_classes, i_fea),dtype=np.float64) #variences
        pri = np.zeros(i_classes,dtype=np.float64) #priors
        std = np.zeros((i_classes, i_fea), dtype=np.float64) #standar deviations

        for c, val  in enumerate(classes): #foreach i_classes
            X_c = X[val] #the class has labels
            means[c, :] = X_c.mean(axis=0) #means
            var[c, :] = X_c.var(axis=0) #variences
            pri[c] = X_c.shape[0] / float(i_val) #frequency of class, prior probability
            std[c, :] = X_c.std(axis=0) #standard deviation for gaussian

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

    def train(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for val, c in enumerate(self._classes):
            prior = np.log(self._pri[val])
            condt = np.sum(np.log(self._pdf(val, x))) #get log to prevent overflow
            posterior = prior + condt
            posteriors.append(posterior)
        return np.argmax(posteriors) #argmax to find highest posterior probability

    def _pdf(self, class_val, x): #class cond. probability second part
        mean = self._means[class_val][0] #get mean of the class
        var = self._var[class_val][0]
        numerator = np.exp(-(x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def separate(self,X): #separate by class
        sep = dict()
        for index, val in enumerate(X):
            i = X[val]
            if (index not in sep):
                sep[index] = list()
            sep[index].append(i)
        return sep

    def predict(self, X, fit_x, fit_y):
        #separated = self.separate(X)
        self.fit(fit_x, fit_y)
        y_pred = [self.train(x) for idx, x in X.iterrows()]
        y_pred = [self.labels(x) for x in y_pred]
        return np.array(y_pred)

    def labels(self,predict_df):
        data = predict_df
        conditions = [
            data < 8 - float(1.5),  # adding 1,5 will give the age so doing like that will work as well
            6.5 <= data & data <= (12 - 1.5),
            12 - 1.5 < data

        ]
        labels = ["young", "middle-aged", "old"]

        data = np.select(conditions, labels)

        return data


    def accuracy(self, prediction, y_valid):
        tot = 0
        y_valid=[self.labels(x) for x in y_valid] #labelling because we are trying to classify them, it is not important to have exact same result
        for val, i in enumerate(y_valid):
            print(val,i, prediction[val])
            if i == prediction[val]:
                tot +=1
        return tot / float(len(y_valid))


    def gaussian_dist(self,class_val,x):
        mean = self._means[class_val][0]
        stdev = self._stdev[class_val][0]
        res = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * res

    def train_gau(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for val, c in enumerate(self._classes):
            prior = np.log(self._pri[val])
            condt = np.sum(np.log(self.gaussian_dist(val, x))) #get log to prevent overflow
            posterior = prior + condt
            posteriors.append(posterior)
        return np.argmax(posteriors) #argmax to find highest posterior probability

    def predict_gau(self, X, fit_x, fit_y):
        #separated = self.separate(X)
        self.fit(fit_x, fit_y)
        y_pred = [self.train_gau(x) for idx, x in X.iterrows()]
        y_pred = [self.labels(x) for x in y_pred]
        return np.array(y_pred)