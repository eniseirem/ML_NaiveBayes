#Bayes relies on conditional probability

import numpy as np

class NaiveBayes:
    def fit(self, X, y): #this will be fitting the training. Splitted dataset to X data matrice and y target column.
        i_val, i_fea =X.shape
        classes = list(X)
        i_classes = len(classes) #find the lenght to fit in

        means = np.zeros((i_classes, i_fea),dtype=np.float64) #for each feature we'll get mean
        var = np.zeros((i_classes, i_fea),dtype=np.float64) #for each feature we'll get mean
        pri = np.zeros(i_classes,dtype=np.float64) #for each feature we'll get mean

        for c, val  in enumerate(classes): #foreach i_classes
            X_c = X[val] #the class has labels
            means[c, :] = X_c.mean(axis=0) #means
            var[c, :] = X_c.var(axis=0) #variences
            pri[c] = X_c.shape[0] / float(i_val) #frequency of class

       #which the values actually stated in the data files as ;

       # 	Length	Diam	Height	Whole	Shucked	Viscera	Shell	Rings
       # Min	0.075	0.055	0.000	0.002	0.001	0.001	0.002	    1
       # Max	0.815	0.650	1.130	2.826	1.488	0.760	1.005	   29
       # Mean	0.524	0.408	0.140	0.829	0.359	0.181	0.239	9.934
       # SD	0.120	0.099	0.042	0.490	0.222	0.110	0.139	3.224
       # Correl	0.557	0.575	0.557	0.540	0.421	0.504	0.628	  1.0

    def train(self): #usage of train
                    #magic happens ?
        pass

    def predict(self):
                    #last one
        pass

