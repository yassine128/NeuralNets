import numpy as np 

class Activation:         
    def sgn(self): 
        return lambda sum_value: 1 if sum_value > 0 else -1 
        
    def sigmoid(self): 
        return lambda sum_value: 1 / (1 + np.exp(-np.clip(sum_value, -500, 500)))
    
    def sigmoid_derivative(self, x): 
        return x * (1 - x)
        

class Perceptron: 
    def __init__(self, size: int, activation) -> None:
        self.size = size
        self.weights = np.zeros(size)
        self.threshold = 0
        self.losses = []
        self.activation = activation
        
    def output(self, input: np.array) -> float: 
        if len(input) != self.size:
            raise ValueError("Input size does not match perceptron size.")
        return self.activation(self.weights.dot(input) - self.threshold)
    
    def updateWeights(self, pred: int, label: int, alpha: float, data: list) -> None: 
        e = (label - pred) / 2 
        self.weights = self.weights + alpha * e * data 
        self.threshold = self.threshold - alpha * e 

    def loss(self, label, data):
        return max(0, -label * (self.weights.dot(data) - self.threshold))
                
    def train(self, alpha: float, dataset: list, labels: list, epochs: int) -> None: 
        if len(dataset) != len(labels): 
            print(f"Dataset of shape: {len(dataset)} is not the same as shape of labels: {len(labels)}")
        else:
            for _ in range(epochs):
                total_loss = 0
                for data, label in zip(dataset, labels):
                    pred = self.output(data)
                    total_loss += self.loss(label, data)
                    self.updateWeights(pred, label, alpha, data)
                self.losses.append(total_loss / len(dataset))


class MLP: 
    # First Layer (0) : Input Layer 
    # Second Layer ... (n-1) : Hidden Layers 
    # Last Layer (n) : Output Layer 
    # Ex: [2, 3, 1]
    def __init__(self, layers_size: list[int]): 
        self.layers_size = layers_size 
        self.layers = []
        for i in range(1, len(layers_size)): 
            layer = []
            for j in range(layers_size[i]): 
                layer.append(Perceptron(layers_size[i - 1], Activation().sigmoid()))
            self.layers.append(layer.copy())

    def info(self): 
        for layer, i in zip(self.layers, range(len(self.layers))):
            print(f"\n Layer {i+1}: ")
            for perceptron in layer: 
                print(f"\t size_input={perceptron.size}, activation={perceptron.activation.__qualname__.split('.')[1]}", end=", ")
    
    def forward(self, input_data):
        self.inputs_forward = [input_data.copy()]
        for layer in self.layers:
            output_layer = []
            for perceptron in layer: 
                output_layer.append(perceptron.output(input_data)) 
            self.inputs_forward.append(output_layer.copy()) 
            input_data = output_layer.copy()
        self.inputs_forward = self.inputs_forward[:-1]
        return input_data
            
    def backward(self, input_data, predicted_output, true_label, learning_rate):
        # Calculate the initial error at the output layer
        output_layer = self.layers[-1]
        a = Activation()
        for perceptron, predicted_value in zip(output_layer, predicted_output):
            delta = (true_label - predicted_value) * a.sigmoid_derivative(predicted_value)
            for i in range(len(perceptron.weights)):
                perceptron.weights[i] += learning_rate * delta * self.inputs_forward[-1][i]
            perceptron.threshold -= learning_rate * delta
        
        # Backpropagate the error through hidden layers
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            next_deltas = [a.sigmoid_derivative(output) for output in self.inputs_forward[layer_idx + 1]]
            
            for perceptron_idx, perceptron in enumerate(layer):
                delta = sum([next_deltas[next_idx] * next_layer[next_idx].weights[perceptron_idx] for next_idx in range(len(next_layer))])
                delta *= a.sigmoid_derivative(self.inputs_forward[layer_idx][perceptron_idx])
                
                for i in range(len(perceptron.weights)):
                    perceptron.weights[i] += learning_rate * delta * self.inputs_forward[layer_idx][i]
                perceptron.threshold -= learning_rate * delta
        
    def train(self, input_data, labels, learning_rate, epochs):
        for _ in range(epochs):
            for data, label in zip(input_data, labels):
                predicted_output = self.forward(data)
                self.backward(data, predicted_output, label, learning_rate)

