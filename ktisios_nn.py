import numpy as np

#note: m = X.shape[1]

class NeuralNet():

    def __init__(self,n_input,n_output,n_hidden):
        self.W1 = np.random.randn(n_input,n_hidden)
        self.W2 = np.random.randn(n_hidden,n_output)
        self.b1 = np.zeros((1,n_hidden))
        self.b2 = np.zeros((1,n_output))
    
    def Sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def Forward(self,X):
        self.Z1 = np.dot(X,self.W1)+self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.Z1,self.W2) + self.b2
        self.A2 = self.Sigmoid(self.Z2)

        return self.A2

    def calCost(self,Y,m):
        self.cost = -(np.dot(Y,np.log(self.A2))+np.dot((1-Y),np.log(1-self.A2)))/m
        return self.cost

    def BackPropagation(self,X,Y):
        dZ2 = A2 - Y
