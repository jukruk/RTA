import numpy as np


import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])
X = df.iloc[:100,[0,2]].values
y = df.iloc[0:100,4].values
y = np.where(y == 0, 0, 1)

class FirstPerceptron():
    def __init__(self, n_features):
         self.n_f = n_features

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        pred = np.where(z > 0, 1, 0)
        return pred

    def fit(self, X, y, n_epochs):
         
        self.n_epochs = n_epochs 
        self.w = np.zeros((self.n_f,), dtype = float)
        self.b = float(0)

        for i in range(self.n_epochs):
            for xi, yi in zip(X, y):
                pred_y = self.predict(xi)
                error = yi - pred_y
                self.w = self.w + error * xi
                self.b += error
        return self

per = FirstPerceptron(2)
per.fit(X,y,10)

with open('model.pkl', "wb") as picklefile:
    pickle.dump(per, picklefile)


