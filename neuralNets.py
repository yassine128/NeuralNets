import numpy as np 

class Perceptron: 
    def __init__(self, size: int) -> None:
        self.size = size
        self.weights = np.zeros(size)
        self.threshold = 0
    
    def sgn(self, sum_value: float) -> int: 
        return 1 if sum_value > 0 else -1 

    def output(self, input: np.array) -> float: 
        return self.sgn(self.weights.dot(input.T) - self.threshold)
    
    def updateWeights(self, pred: int, label: int, alpha: float, data: list) -> None: 
        e = (label - pred) / 2 
        self.weights = self.weights + alpha * e * data 
        self.threshold = self.threshold - alpha * e 
                
    def train(self, alpha: float, dataset: list, labels: list, epochs: int) -> None: 
        if len(dataset) != len(labels): 
            print(f"Dataset of shape: {len(dataset)} is not the same as shape of labels: {len(labels)}")
        else: 
            for _ in range(epochs):
                for data, label in zip(dataset, labels):
                    pred = self.output(data)
                    self.updateWeights(pred, label, alpha, data)
