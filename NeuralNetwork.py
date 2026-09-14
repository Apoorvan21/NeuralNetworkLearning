class NeuralNetwork:
  def __init__(self,layer_size):
    self.layers = []
    for i in range(len(layer_size)-1):
      self.layers.append(Layer(layer_size[i],layer_size[i+1]))
  
  def forward(self,x):
    for layer in self.layers:
      x = layer.forward(x)
    return x
