import numpy as np
import matplotlib.pyplot as plt


# Linear model: y = ax + b
class LinearModel:
    def __init__(self, learning_rate=0.01):
        self.weight = np.random.randn()
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def forwardPropagation(self, X):
        return self.weight * X + self.bias

    def meanSquaredError(self, predictions, target):
        return np.mean((predictions - target) ** 2)

    def backwardPropagation(self, X, predictions, target):
        error = predictions - target
        weight_gradient = np.mean(error * X)
        bias_gradient = np.mean(error)

        # Update parameters
        self.weight -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

    def train(self, X, y, epochs=200):
        losses = []
        for epoch in range(epochs):

            predictions = self.forwardPropagation(X)
            loss = self.meanSquaredError(predictions, y)
            losses.append(loss)
            self.backwardPropagation(X, predictions, y)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}, Weight: {self.weight:.6f}, Bias: {self.bias:.6f}")

        return losses


# Example usage
if __name__ == "__main__":
    X = np.array([2, 4, 6, 8, 10])
    y = 2 * X + 30  # Fixed target

    model = LinearModel(learning_rate=0.01)
    epochs = 200
    losses = model.train(X, y, epochs=epochs)

    # Output final weight, bias, and predictions
    print(f"\nFinal weight: {model.weight}")
    print(f"Final bias: {model.bias}")

    # Plot learning progress (loss over epochs)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), losses, color='b', label='Loss')
    plt.title('Learning Progress V1 (Loss Over Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()

    # Plot data and model prediction
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    predictions = model.forwardPropagation(X)
    plt.plot(X, predictions, color='red', label='Model Prediction')
    plt.title('Linear Model: y = 2 * x + 30')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    #plt.show()

    X_ = np.array([2, 4, 6, 8, 10])
    y_ = 5 * X + 47  # Fixed target
    losses_ = model.train(X_, y_, epochs=epochs)

    # Plot learning progress (loss over epochs)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), losses_, color='b', label='Loss')
    plt.title('Learning Progress V2 (Loss Over Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()

    # Plot data and model prediction
    plt.figure(figsize=(8, 6))
    plt.scatter(X_, y_, color='blue', label='Data Points')
    predictions = model.forwardPropagation(X)
    plt.plot(X, predictions, color='red', label='Model Prediction')
    plt.title('Linear Model: y = 5 * x + 47')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()
