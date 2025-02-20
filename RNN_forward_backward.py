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
        self.W = np.random.randn(self.input_dim, self.hidden_dim)*0.01
        self.U = np.random.randn(self.hidden_dim, self.hidden_dim)*0.01
        self.V = np.random.randn(self.hidden_dim, self.output_dim)*0.01
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
        self.x = x
        self.h = np.zeros((seq_len + 1, self.hidden_dim))
        self.y_hat = np.zeros((seq_len, self.output_dim))
        for t in range(seq_len):
            z = np.dot(x[t], self.W) + np.dot(self.h[t-1], self.U) + self.b
            self.h[t] = np.tanh(z)
            o = np.dot(self.h[t], self.V) + self.c
            self.y_hat[t] = self.softmax(o)
        return self.y_hat

    def predict(self, x):
        y_hat = self.forward(x)
        return np.argmax(y_hat, axis=1)

    def loss(self, y, y_hat):
        loss = - np.sum(np.log(y.T.dot(y_hat)))
        return loss

    def total_loss(self, Y, labels):
        seq_len = Y.shape[0]
        total_loss = 0
        for t in range(seq_len):
            total_loss += self.loss(labels[t], Y[t])
        return total_loss

    def backwards(self, y, y_hat):
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
            dV += np.outer(self.h[t], dy)
            dc += dy

            dh = np.dot(self.V, dy) + np.dot(self.U, dz_next)
            dz = (1 - self.h[t] ** 2) * dh

            dW += np.outer(self.x[t], dz)
            dU += np.outer(self.h[t-1], dz)
            db += dz

            dz_next = dz
        
        for dparam in [dW, dU, dV, db, dc]:
            np.clip(dparam, -5, 5, out=dparam)

        self.dW = dW
        self.dU = dU
        self.dV = dV
        self.db = db
        self.dc = dc

    def step(self, learning_rate):

        self.V -= learning_rate * self.dV
        self.c -= learning_rate * self.dc
        self.W -= learning_rate * self.dW
        self.U -= learning_rate * self.dU
        self.b -= learning_rate * self.db

    def train(self, x, y, epochs=10, learning_rate=0.03):
        for epoch in range(epochs):
            batch_size = x.shape[0]
            loss = 0
            for i in range(batch_size):
                x_batch = x[i]
                y_batch = y[i]
                y_hat = self.forward(x_batch)
                loss += self.total_loss(y_hat, y_batch)
                self.backwards(y_batch, y_hat)
                self.step(learning_rate)
                self.gradient_check(x_batch, y_batch)
            print('Epoch: {}, Loss: {}'.format(epoch, loss / batch_size))

    def gradient_check(self, x, y, epsilon=1e-5):
        W_orig = self.W.copy()
        U_orig = self.U.copy()
        V_orig = self.V.copy()
        b_orig = self.b.copy()
        c_orig = self.c.copy()

        y_hat = self.forward(x)
        loss_orig = self.total_loss(y_hat, y)
        self.backwards(y, y_hat)

        dW_analytic = self.dW
        dW_numeric = np.zeros_like(self.W)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i, j] = W_orig[i, j] + epsilon
                y_hat = self.forward(x)
                loss_plus = self.total_loss(y_hat, y)

                self.W[i, j] = W_orig[i, j] - epsilon
                y_hat = self.forward(x)
                loss_minus = self.total_loss(y_hat, y)

                dW_numeric[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                self.W[i, j] = W_orig[i, j]
        
        def rel_error(x, y):
            return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

        print('dW error: ', rel_error(dW_analytic, dW_numeric))

if __name__ == '__main__':
    s1 = '你 好 李 焕 英'
    s2 = '夏 洛 特 烦 恼'
    vocab_size= len(s1.split(' ')) + len(s2.split(' '))
    vocab = [[0] * vocab_size for _ in range(vocab_size)]
    for i in range(vocab_size): vocab[i][i] = 1
    x_sample = np.array([vocab[:5]] + [vocab[5:]])
    labels = np.array([vocab[1:6]] + [vocab[6:]+[vocab[0]]])
    rnn = RNN(10, 20, 10)
    rnn.train(x_sample, labels)
    # rnn.gradient_check(x_sample[0], labels[0])