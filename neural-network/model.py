import numpy as np
import matplotlib.pyplot as plt


# initialize parameters
class NeuralNet():
    def __init__(self, input_size, layer_size, class_size, is_relu, std=1e-1):
        # input_size = D (input dimension)
        # layer_size = H (hidden layer size)
        # class_size = C (the number of classes)

        # w_1 = (D, H), b_1 = (H, )
        # w_2 = (H, H), b_2 = (H, )
        # w_3 = (H, C), b_3 = (C, )

        # initialize parameters with small random variables

        self.params = {}
        self.params['w_1'] = std * np.random.randn(input_size, layer_size)
        self.params['b_1'] = np.zeros(layer_size)
        self.params['w_2'] = std * np.random.randn(layer_size, layer_size)
        self.params['b_2'] = np.zeros(layer_size)
        self.params['w_3'] = std * np.random.randn(layer_size, class_size)
        self.params['b_3'] = np.zeros(class_size)
        self.is_relu = is_relu

    def relu(self, x):
        return np.maximum(x, 0)

    def leaky_relu(self, x):
        return np.maximum(x, 0.01*x)

    def drelu(self, x, z):
        return x * (z > 0)

    def dleaky_relu(self, x, z):
        return np.where(z < 0, 0.01 * x, 1 * x)

    def loss(self, x, y, reg=0.0):
        # x: train image, shape (N, D), x[i] is a training sample
        # y: labels of x. y[i] is the label of x[i]
        # so far, N = the number of training images, D = input image dimension

        w_1, b_1 = self.params['w_1'], self.params['b_1']
        w_2, b_2 = self.params['w_2'], self.params['b_2']
        w_3, b_3 = self.params['w_3'], self.params['b_3']
        N, D = x.shape

        # forward pass
        z_1 = np.dot(x, w_1) + b_1
        if self.is_relu:
            # layer_1 = np.maximum(z_1, 0)
            layer_1 = self.relu(z_1)
        else:
            # layer_1 = np.maximum(z_1, 0.01 * z_1)
            layer_1 = self.leaky_relu(z_1)

        z_2 = np.dot(z_1, w_2) + b_2
        if self.is_relu:
            # layer_2 = np.maximum(z_2, 0)
            layer_2 = self.relu(z_2)
        else:
            # layer_2 = np.maximum(z_2, 0.01 * z_2)
            layer_2 = self.leaky_relu(z_2)

        scores = np.dot(layer_2, w_3) + b_3  # (N, C)

        # softmax probability
        loss = 0.0
        softmax = np.exp(scores)
        # (N, 1) is different to (N, )
        # softmax /= np.sum(softmax, axis=1).reshape(N, 1)
        softmax = (softmax.T) / (np.sum(softmax, axis=1))
        softmax = softmax.T

        # softmax loss
        temp = softmax[np.arange(N), y]  # label matching
        loss -= np.sum(np.log(temp))
        loss /= N
        # gamma is 0.5
        loss += 0.5 * reg * (np.sum(w_1**2) + np.sum(w_2**2) + np.sum(w_3**2))

        # backward
        grads = {}

        # back propagation
        dscores = np.copy(softmax)
        dscores[np.arange(N), y] -= 1
        dscores /= N

        dlayer_2 = np.dot(dscores, w_3.T)
        if self.is_relu:
            dz_2 = self.drelu(dlayer_2, z_2)
        else:
            dz_2 = self.dleaky_relu(dlayer_2, z_2)

        dlayer_1 = np.dot(dlayer_2, w_2.T)
        if self.is_relu:
            dz_1 = self.drelu(dlayer_1, z_1)
        else:
            dz_1 = self.dleaky_relu(dlayer_1, z_1)

        grads['w_3'] = np.dot(layer_2.T, dscores)  # (H, C)
        grads['b_3'] = np.sum(dscores, axis=0)  # (C, )
        grads['w_2'] = np.dot(layer_1.T, dz_2)  # (H, C)
        grads['b_2'] = np.sum(dz_2, axis=0)  # (C, )
        grads['w_1'] = np.dot(x.T, dz_1)  # (D, H)
        grads['b_1'] = np.sum(dz_1, axis=0)  # (H, )

        grads['w_3'] += reg * w_3
        grads['w_2'] += reg * w_2
        grads['w_1'] += reg * w_1

        return loss, grads

    def train(self, x, y, x_val, y_val, lr=1e-3, reg=1e-5, iters=100, batch_size=200):
        # x: train image, shape (N, D), x[i] is a training sample
        # y: labels of x. y[i] is the label of x[i]
        # so far, N = the number of training images, D = input image dimension
        # x_val: validation image, shape (N_val, D)
        # y_val: labels of x_val, shape (N_val, )

        num_train = x.shape[0]
        num_val = x_val.shape[0]

        train_loss_graph = []
        val_loss_graph = []
        train_acc_graph = []
        val_acc_graph = []

        for iter in range(iters):
            random_idxs = np.random.choice(num_train, batch_size)
            x_batch = x[random_idxs]
            y_batch = y[random_idxs]

            random_val_idxs = np.random.choice(num_val, batch_size)
            x_val_batch = x_val[random_val_idxs]
            y_val_batch = y_val[random_val_idxs]

            train_loss, grads = self.loss(x_batch, y_batch, reg)
            train_loss_graph.append(train_loss)
            val_loss, _ = self.loss(x_val_batch, y_val_batch, reg)
            val_loss_graph.append(val_loss)

            # SGD
            self.params['w_3'] -= lr * grads['w_3']
            self.params['b_3'] -= lr * grads['b_3']
            self.params['w_2'] -= lr * grads['w_2']
            self.params['b_2'] -= lr * grads['b_2']
            self.params['w_1'] -= lr * grads['w_1']
            self.params['b_1'] -= lr * grads['b_1']

            if iter % 10 == 0:
                print("iteration %d / %d: loss %f" % (iter, iters, train_loss))

            x_batch_predict, _ = self.predict(x_batch)
            train_acc = (x_batch_predict == y_batch).mean()

            x_val_batch_predict, _ = self.predict(x_val_batch)
            val_acc = (x_val_batch_predict == y_val_batch).mean()

            train_acc_graph.append(train_acc)
            val_acc_graph.append(val_acc)

        return {'train_loss_graph': train_loss_graph, 'val_loss_graph': val_loss_graph, 'train_acc_graph': train_acc_graph, 'val_acc_graph': val_acc_graph}

    def predict(self, x):
        params = self.params

        N = x.shape[0]

        # forward pass
        z_1 = np.dot(x, params['w_1']) + params['b_1']
        if self.is_relu:
            # layer_1 = np.maximum(z_1, 0)
            layer_1 = self.relu(z_1)
        else:
            layer_1 = self.leaky_relu(z_1)

        z_2 = np.dot(layer_1, params['w_2']) + params['b_2']
        if self.is_relu:
            layer_2 = self.relu(z_2)
        else:
            layer_2 = self.leaky_relu(z_2)

        scores = np.dot(layer_2, params['w_3']) + params['b_3']  # (N, C)
        y_pred = np.argmax(scores, axis=1)

        softmax = np.exp(scores)
        softmax /= np.sum(softmax, axis=1).reshape(N, 1)
        y_scores = np.max(softmax, axis=1)

        return y_pred, y_scores
