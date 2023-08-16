import numpy as np 

class Activation():         
    def sgn(self): 
        return lambda sum_value: 1 if sum_value > 0 else -1 
        
    def sigmoid(self): 
        return lambda sum_value: 1 / (1 + np.exp(-np.clip(sum_value, -500, 500)))
        

class Perceptron: 
    def __init__(self, size: int, activation) -> None:
        self.size = size
        self.weights = np.zeros(size)
        self.threshold = 0
        self.losses = []
        self.activation = activation

    def normalize(self, data): 
        return np.linalg.norm(data)
        
    def output(self, input: np.array) -> float: 
        return self.activation(self.weights.dot(input.T) - self.threshold)
    
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
