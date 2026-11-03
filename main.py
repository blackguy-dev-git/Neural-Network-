import numpy as np

np.random.seed(0)


# ---------------------------------------------------------------
# Data: spiral dataset (3 classes) so we have inputs AND labels
# ---------------------------------------------------------------
def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")
    for c in range(classes):
        ix = range(points * c, points * (c + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(c * 4, (c + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = c
    return X, y


# ---------------------------------------------------------------
# Dense layer
# ---------------------------------------------------------------
class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        # randn (not rand) so weights can be negative; small scale
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs                      # remember for backward
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # gradients of the loss w.r.t. weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


# ---------------------------------------------------------------
# ReLU activation
# ---------------------------------------------------------------
class activation_relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0        # gradient is 0 where input was <= 0


# ---------------------------------------------------------------
# Softmax activation (output layer -> probabilities)
# ---------------------------------------------------------------
class activation_softmax:
    def forward(self, inputs):
        # subtract row max to avoid overflow in exp()
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


# ---------------------------------------------------------------
# Loss: categorical cross-entropy
# ---------------------------------------------------------------
class loss_categorical_crossentropy:
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)   # avoid log(0)

        if y_true.ndim == 1:                                # sparse labels [0,2,1,...]
            correct_conf = y_pred_clipped[range(len(y_pred)), y_true]
        else:                                               # one-hot labels
            correct_conf = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_conf)                        # per-sample loss

    def calculate(self, y_pred, y_true):
        return np.mean(self.forward(y_pred, y_true))


# ---------------------------------------------------------------
# Softmax + cross-entropy combined: the backward pass simplifies
# to just (predicted - true), much faster and more stable.
# ---------------------------------------------------------------
class activation_softmax_loss_crossentropy:
    def __init__(self):
        self.activation = activation_softmax()
        self.loss = loss_categorical_crossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if y_true.ndim == 2:                       # one-hot -> sparse
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1  # softmax output minus 1 at true class
        self.dinputs = self.dinputs / samples      # average over batch


# ---------------------------------------------------------------
# Optimizer: SGD with learning-rate decay and momentum
# ---------------------------------------------------------------
class optimizer_sgd:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.initial_lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.initial_lr / (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            layer.weight_momentums = (self.momentum * layer.weight_momentums
                                      - self.current_lr * layer.dweights)
            layer.bias_momentums = (self.momentum * layer.bias_momentums
                                    - self.current_lr * layer.dbiases)
            layer.weights += layer.weight_momentums
            layer.biases += layer.bias_momentums
        else:
            layer.weights += -self.current_lr * layer.dweights
            layer.biases += -self.current_lr * layer.dbiases

    def post_update_params(self):
        self.iterations += 1


# ---------------------------------------------------------------
# Build the network: 2 -> 64 -> 3
# ---------------------------------------------------------------
X, y = spiral_data(points=100, classes=3)

dense1 = layer_dense(2, 64)
act1 = activation_relu()
dense2 = layer_dense(64, 3)
loss_activation = activation_softmax_loss_crossentropy()
optimizer = optimizer_sgd(learning_rate=1.0, decay=1e-3, momentum=0.9)

# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------
for epoch in range(10001):
    # ---- forward pass ----
    dense1.forward(X)
    act1.forward(dense1.output)
    dense2.forward(act1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 1000 == 0:
        print(f"epoch: {epoch:5d}  acc: {accuracy:.3f}  "
              f"loss: {loss:.3f}  lr: {optimizer.current_lr:.5f}")

    # ---- backward pass (reverse order of forward) ----
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    act1.backward(dense2.dinputs)
    dense1.backward(act1.dinputs)

    # ---- update weights ----
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# ---------------------------------------------------------------
# Validation on fresh data (checks for overfitting)
# ---------------------------------------------------------------
X_test, y_test = spiral_data(points=100, classes=3)
dense1.forward(X_test)
act1.forward(dense1.output)
dense2.forward(act1.output)
test_loss = loss_activation.forward(dense2.output, y_test)
test_acc = np.mean(np.argmax(loss_activation.output, axis=1) == y_test)
print(f"\nvalidation, acc: {test_acc:.3f}, loss: {test_loss:.3f}")