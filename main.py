n = NeuralNetwork([2,4,5,1])
x = np.array([0.25,0.42])
y = np.array([2.1])

loss = (n.forward(x)-y)**2

