import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

def lossFunction(y,y_pred):
  loss = (0.5) * np.mean((y - y_pred)**2)
  return loss

def train(nn,x,y,epochs,a):
  losses = []
  for i in range(epochs):
    y_pred = nn.forward(x)
    loss = lossFunction(y,y_pred)
    losses.append(loss)
    dL_dz = (y_pred - y)
    nn.backward(dL_dz)
    nn.update(a)
  return losses

nn = NeuralNetwork([2,4,5,1])
x = np.array([[0.25, 0.50], [0.42, 0.30]])

y = np.array([[2.1, 3.0]])

print(nn.forward(x))

train(nn,x,y,100,0.1)

print(nn.forward(x))
