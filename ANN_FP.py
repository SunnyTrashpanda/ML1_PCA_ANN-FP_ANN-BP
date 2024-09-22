import numpy as np

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return np.where(x > 0, 1, 0)

def meanSquaredError(predicted, target):
    return np.mean((predicted - target) ** 2)


def initializeParameters(input_size, hidden_layer_1_size, hidden_layer_2_size, output_size):
    np.random.seed(1)

    # input to hidden layer 1
    w1 = np.random.randn(input_size, hidden_layer_1_size)
    b1 = np.random.randn(hidden_layer_1_size)

    # hidden layer 1 to hidden layer 2
    w2 = np.random.randn(hidden_layer_1_size, hidden_layer_2_size)
    b2 = np.random.randn(hidden_layer_2_size)

    # hidden layer 2 to output
    w3 = np.random.randn(hidden_layer_2_size, output_size)
    b3 = np.random.randn(output_size)

    return w1, b1, w2, b2, w3, b3


def forwardPropagation(x, w1, b1, w2, b2, w3, b3):
    # Input to hidden layer 1
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)

    # Input to hidden layer 2
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)

    # Input to output
    z3 = np.dot(a2, w3) + b3
    a3 = relu(z3)

    return a1, a2, a3


if __name__ == "__main__":
    x = np.array([[2, 8]])
    input_size = 2
    hiddenLayer1_size = 3
    hiddenLayer2_size = 4
    output_size = 1
    target = np.array([[0.65]]) # Fixed target

    w1, b1, w2, b2, w3, b3 = initializeParameters(input_size, hiddenLayer1_size, hiddenLayer2_size, output_size)

    bestLoss_mse = float('inf')
    bestParams = None

    for i in range(100):

        a1, a2, a3 = forwardPropagation(x, w1, b1, w2, b2, w3, b3)

        loss_mae = meanSquaredError(a3, target)

        if loss_mae < bestLoss_mse:
            bestLoss_mse = loss_mae
            bestParams = (w1, b1, w2, b2, w3, b3)

        # Reinitialize weights and biases for the next iteration
        w1, b1, w2, b2, w3, b3 = initializeParameters(input_size, hiddenLayer1_size, hiddenLayer2_size, output_size)

    print("Best Loss (mse) after 100 runs:", bestLoss_mse)
    print("Best Parameters (Weights and Biases):", bestParams)
