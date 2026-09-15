import numpy as np
import matplotlib.pyplot as plt

def lossFunction(y,y_pred):
  avg = 0
  for i in range(len(y)):
    avg += (y[i] - y_pred[i])**2
  loss = (0.5) * avg/len(y)

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

n = NeuralNetwork([2,4,5,1])
x = np.array([0.25,0.42])
y = np.array([2.1])

print(nn.forward(x))

train(nn,x,y,100,0.1)

print(nn.forward(x))
