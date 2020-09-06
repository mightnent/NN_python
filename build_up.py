import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense():
    def __init__(self,n_input,n_neurons):
        self.weights = 0.1*np.random.randn(n_input,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,X):
        self.output = np.dot(X,self.weights) + self.biases

class Activation_ReLU():
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax():
    def forward(self,inputs):
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        prob = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = prob

class Loss():
    def calculate(self,output,y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples= len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        y_pred_clipped = y_pred_clipped[range(samples),y_true]
        neg_log_likelihood = -np.log(y_pred_clipped)
        #print(y_pred_clipped)
        #print(neg_log_likelihood)
        return neg_log_likelihood

X,y = spiral_data(100,3)

layer1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

#find loss
loss = loss_function.calculate(activation2.output,y)

#find accuracy
#argmax will just return 1 for the highest prob and 0 otherwise
pred = np.argmax(activation2.output,axis=1)
accuracy = np.mean(pred == y)

print(accuracy)



