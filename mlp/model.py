"""
MLP Model for A/B Test Factor Optimization
Based on Farrell et al. (2021) architecture recommendations
"""

import numpy as np
from typing import List, Tuple, Optional


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class MLP:
    """
    Multilayer Perceptron for A/B Test factor combination optimization.
    
    Architecture follows Farrell et al. (2021):
    - Depth L: number of hidden layers
    - Width H: number of neurons per hidden layer
    - ReLU activation throughout
    - MSE loss for regression on continuous response variable
    """

    def __init__(self, input_dim: int, depth: int, width: int, output_dim: int = 1,
                 learning_rate: float = 0.01, seed: Optional[int] = None):
        """
        Args:
            input_dim:     Number of binary factors (A/B test variables)
            depth:         L — number of hidden layers
            width:         H — neurons per hidden layer
            output_dim:    Output dimension (1 for regression)
            learning_rate: Step size for gradient descent
            seed:          Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.depth = depth
        self.width = width
        self.lr = learning_rate
        self.relu = ReLU()

        # Build layer dimensions
        dims = [input_dim] + [width] * depth + [output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            # He initialization for ReLU networks
            fan_in = dims[i]
            w = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, dims[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass, returns predictions and cache for backprop."""
        self._cache = {'a': [X], 'z': []}
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            self._cache['z'].append(z)
            # No activation on final layer (regression)
            a = self.relu.forward(z) if i < len(self.weights) - 1 else z
            self._cache['a'].append(a)
        return a

    def backward(self, y_true: np.ndarray) -> float:
        """Backprop with MSE loss. Returns loss value."""
        n = y_true.shape[0]
        y_pred = self._cache['a'][-1]
        loss = np.mean((y_pred - y_true) ** 2)

        # Gradient of MSE
        delta = 2 * (y_pred - y_true) / n

        for i in reversed(range(len(self.weights))):
            a_prev = self._cache['a'][i]
            dw = a_prev.T @ delta
            db = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu.backward(self._cache['z'][i - 1])
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 500,
            verbose: bool = False) -> List[float]:
        """Train the MLP. Returns loss history."""
        y = y.reshape(-1, 1)
        losses = []
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(y)
            losses.append(loss)
            if verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.6f}")
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).flatten()

    def predict_best_combination(self, all_combinations: np.ndarray) -> int:
        """Returns the index of the predicted best factor combination."""
        preds = self.predict(all_combinations)
        return int(np.argmax(preds))
