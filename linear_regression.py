import numpy as np

class LinerRegression:
    def __init__(self, learning_rate = 0.01, n_iters = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            
            #caclulating y predict
            y_pred = np.dot(x, self.weights) +self.bias
            
            #caclulating the derivative of weight
            dw = (1/n_samples) * np.dot(x.T,(y_pred-y))
            #calculating the derivative of bias
            db = (1/n_samples) * np.sum(y_pred-y)
            
            #updating the weight and bias i.e. implementing gradient descent
            self.weights = self.weights - self.learning_rate *dw
            self.bias = self.bias - self.learning_rate * db
            
    def predict (self,x):
        #calculate prediction of y with final updated weights and biases
        y_pred = np.dot(x,self.weights)+self.bias
        return y_pred