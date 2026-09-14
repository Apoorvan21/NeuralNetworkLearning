class Layer:
  def __init__(self,inp_size,out_size):
    self.W = np.random.randn(out_size,inp_size)
    self.b = np.zeros(out_size)
    self.X = None
    self.dW = None
    self.db = None

  def forward(self,X):
    self.X = X
    return self.W @ X + self.b
