import numpy as np

class RNN:

    def __init__(self, input_dim, hidden_dim, output_dim):
        # x_n, z_n, h_n, y_hat_n, W, U, V, b, c
        # z_n = W * x_n + U * h_n + b
        # h_n = tanh(z_n)
        # o_n = V * h_n + c
        # y_hat_n = softmax(o_n)
        # loss = - sum(y_n * log(y_hat_n))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W = np.random.randn(self.input_dim, self.hidden_dim)
        self.U = np.random.randn(self.hidden_dim, self.hidden_dim)
        self.V = np.random.randn(self.hidden_dim, self.output_dim)
        self.b = np.zeros(self.hidden_dim)
        self.c = np.zeros(self.output_dim)

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def forward(self, x):
        # x: (seq_len, input_dim)
        # z: (seq_len, hidden_dim)
        # h: (seq_len, hidden_dim)
        # o: (seq_len, output_dim)
        # y_hat: (seq_len, output_dim)
        seq_len = x.shape[0]
        self.h = np.zeros((seq_len + 1, self.hidden_dim))
        self.y_hat = np.zeros((seq_len, self.output_dim))
        for t in range(seq_len):
            z = np.dot(x[t], self.W) + np.dot(h[t-1], self.U) + self.b
            self.h[t] = np.tanh(z)
            o = np.dot(h[t], self.V) + self.c
            self.y_hat[t] = self.softmax(o)
        return self.y_hat

    def predict(self, x):
        y_hat = self.forward(x)
        return np.argmax(y_hat, axis=1)

    def loss(self, y, y_hat):
        return - np.sum(y * np.log(y_hat))

    def total_loss(self, Y, labels):
        seq_len = Y.shape[0]
        total_loss = 0
        for t in range(seq_len):
            total_loss += self.loss(labels[t], Y[t])
        return total_loss

    def backwards(self, y, y_hat, learning_rate):
        T = y.shape[0]
        
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        dV = np.zeros_like(self.V)
        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)

        dz_next = np.zeros_like(self.b)

        for t in range(T-1, -1, -1):
            dy = y_hat[t] - y[t]
            # dV (hidden_dim, output_dim)
            # dy (output_dim,)
            # h[t] (hidden_dim,)
            dV += np.outer(h[t], dy)
            dc += dy

            dh = np.dot(self.V, dy) + dz_next * U
            dz

            dW += np.outer(x[t], dz)
            dU += np.outer(h[t-1], dz)
            db += dz

        self.V -= learning_rate * dV
        self.c -= learning_rate * dc
        self.W -= learning_rate * dW
        self.U -= learning_rate * dU
        self.b -= learning_rate * db

    def train(self, x, y, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            y_hat = self.forward(x)
            loss = self.total_loss(y, y_hat)
            print('Epoch: {}, Loss: {}'.format(epoch, loss))
            self.backwards()

if __name__ == '__main__':
    rnn = RNN(2, 3, 2)
    x = np.array([[1, 2], [3, 4], [5, 6]])
    labels = np.array([[1, 0], [1, 0], [1, 0]])
    y_hat = rnn.forward(x)
    print(y_hat)
    print(rnn.predict(x))
    print(rnn.total_loss(y_hat, labels))