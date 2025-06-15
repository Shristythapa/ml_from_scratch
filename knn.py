import numpy as np
from collections import Counter
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self,k=3):
        self.k = k
    
    def fit(self, X,y):
        self.X_train = X
        self.Y_train = y
        
    def perdict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions
        
    def _predict(self,x):
        
        # this function is reponsible for computing distance of one point of x input
        
        #compute the distances for each value of x train
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
    
        #get the closest k
        k_indices =  np.argsort(distances)[:self.k]
        k_nearest_lables = [self.Y_train[i] for i in k_indices]
        
        #determine label with majority vote based on top k
        most_common = Counter(k_nearest_lables).most_common()[0][0]
        return most_common