import numpy as np 

class Activation:         
    def sgn(self): 
        return lambda sum_value: 1 if sum_value > 0 else -1 
        
    def sigmoid(self): 
        return lambda sum_value: 1 / (1 + np.exp(-np.clip(sum_value, -500, 500)))
    
    def sigmoid_derivative(self): 
        return lambda x:  x * (1 - x)
        

class Perceptron: 
    def __init__(self, size: int, activation) -> None:
        self.size = size
        self.weights = np.zeros(size)
        self.bias = 0
        self.losses = []
        self.activation = activation
        
    def output(self, input: np.array) -> float: 
        if len(input) != self.size:
            raise ValueError("Input size does not match perceptron size.")
        return self.activation(self.weights.dot(input) - self.bias)
    
    def updateWeights(self, pred: int, label: int, alpha: float, data: list) -> None: 
        e = (label - pred) / 2 
        self.weights = self.weights + alpha * e * data 
        self.bias = self.bias - alpha * e 

    def loss(self, label, data):
        return max(0, -label * (self.weights.dot(data) - self.bias))
                
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
    def __init__(self, layers_size: list[int], activation, derivative): 
        self.weights = [np.random.randn(layers_size[i], layers_size[i+1]) for i in range(len(layers_size) - 1)]
        self.bias = [np.random.randn(1, layers_size[i+1]) for i in range(len(layers_size) - 1)]
        self.activation = activation
        self.derivative = derivative

    def forward(self, input_data): 
        self.activations = [input_data]
        for w, b in zip(self.weights, self.bias): 
            z = np.dot(self.activations[-1], w) + b 
            self.activations.append(self.activation(z))
        return self.activations[-1]

    def backward(self, real_output, alpha): 
        output_error = real_output - self.activations[-1] 
        output_delta = output_error * self.derivative(self.activations[-1])

        for i in range(len(self.weights) - 1, -1, -1): 
            prev_activation = self.activations[i]
            weight_delta = np.dot(prev_activation.T, output_delta)
            self.weights[i] += alpha * weight_delta
            self.bias[i] += alpha * output_delta

            output_delta = np.dot(output_delta, self.weights[i].T) * self.derivative(prev_activation)
    
    def train(self, input_data, labels, learning_rate, epochs):
        loss_values = []  # List to store loss values

        for _ in range(epochs):
            epoch_loss = 0

            for data, label in zip(input_data, labels):
                data = np.array(data).reshape(1, -1)
                label = np.array(label).reshape(1, -1)
                predicted_output = self.forward(data)
                self.backward(label, learning_rate)

                # Calculate loss for this data point and add to epoch_loss
                epoch_loss += (predicted_output - label) ** 2

            # Calculate average loss for the epoch and store in loss_values
            avg_epoch_loss = epoch_loss / len(input_data)
            loss_values.append(avg_epoch_loss)

        return loss_values  # Return the list of loss values     



