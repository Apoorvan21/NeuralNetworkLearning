import numpy as np

class Layer:
  def __init__(self,inp_size,out_size):
    self.W = np.random.randn(out_size,inp_size) *0.01
    self.b = np.zeros((out_size, 1))
    self.X = None
    self.dW = None
    self.db = None

  def forward(self,X):
    self.X = X
    return self.W @ X + self.b

  def backward(self,dL_dz):
    self.db = np.sum(dL_dz, axis=1, keepdims=True)
    self.dW = dL_dz @ self.X.T
    return self.W.T @ dL_dz

  def update(self,a):
    self.W -= a * self.dW
    self.b -= a * self.db
