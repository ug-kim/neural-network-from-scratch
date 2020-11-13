import numpy as np
from layers import *

from tensorboardX import SummaryWriter


class TwoLayerNet():
    def __init__(self, input_dim=(1, 28, 28), filter_size=3, num_classes=10, scale=1e-1, reg=0.0):
        # input_size = D (input dimension)
        # layer_size = H (hidden layer size)
        # class_size = C (the number of classes)

        # w_1 = (D, H), b_1 = (H, )
        # w_2 = (H, C), b_2 = (C, )

        # initialize parameters with small random variables
        C, H, W = input_dim
        num_filters_1 = 5
        num_filters_2 = 10

        self.params = {}
        self.reg = reg
        # (1, 28, 28) -> (5, 28, 28) -> (5, 14, 14)
        self.params['w_1'] = scale * \
            np.random.randn(num_filters_1, C, filter_size, filter_size)
        self.params['b_1'] = np.zeros(num_filters_1)
        # (5, 14, 14) -> (10, 14, 14) -> (10, 7, 7)
        self.params['w_2'] = scale * \
            np.random.randn(num_filters_2, num_filters_1,
                            filter_size, filter_size)
        self.params['b_2'] = np.zeros(num_filters_2)
        self.params['w_3'] = scale * \
            np.random.randn(int(7 * 7 * num_filters_2), num_classes)
        self.params['b_3'] = np.zeros(num_classes)

        for key, value in self.params.items():
            self.params[key] = value.astype(np.float32)

        self.conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


    def loss(self, x, y):
        w_1, b_1 = self.params['w_1'], self.params['b_1']
        w_2, b_2 = self.params['w_2'], self.params['b_2']
        w_3, b_3 = self.params['w_3'], self.params['b_3']

        # conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        # pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        conv_param = self.conv_param
        pool_param = self.pool_param

        # conv_res, conv_cache = conv(x, w_1, b_1, conv_param)
        # relu_res, relu_cache  relu(conv_res)
        # out, pool_cache = max_pool(relu_res, pool_param)

        # conv_res_2, conv_cache_2 = conv(out, w_2, b_2, conv_param)
        # relu_res_2, relu_cache_2 = relu(conv_res_2)
        # out_2, pool_cache_2 = max_pool(relu_res_2, pool_param)

        conv_1, cache_1 = conv_relu_pool(
            x, w_1, b_1, conv_param, pool_param)
        conv_2, cache_2 = conv_relu_pool(
            conv_1, w_2, b_2, conv_param, pool_param)
        scores, linear_cache = linear(conv_2, w_3, b_3)

        grads = {}
        loss, dout = softmax(scores, y)

        loss += self.reg * 0.5 * \
            (np.sum(w_1 ** 2) + np.sum(w_2 ** 2) + np.sum(w_3 ** 2))

        dx_3, grads['w_3'], grads['b_3'] = dlinear(dout, linear_cache)
        dx_2, grads['w_2'], grads['b_2'] = dconv_relu_pool(dx_3, cache_2)
        dx_1, grads['w_1'], grads['b_1'] = dconv_relu_pool(dx_2, cache_1)

        grads['w_3'] += self.reg * self.params['w_3']
        grads['w_2'] += self.reg * self.params['w_2']
        grads['w_1'] += self.reg * self.params['w_1']

        return loss, grads

    def train(self, x, y, x_val, y_val, batch_size=200, lr=1e-3, reg=1e-5, epoch=3):
        num_train = x.shape[0]
        num_val = x_val.shape[0]

        iter_per_epoch = max(num_train // batch_size, 1)
        # iter_per_epoch = 100

        self.reg = reg
        writer = SummaryWriter(logdir='logs/scratch_new')

        train_loss_graph = []
        val_loss_graph = []
        train_acc_graph = []
        val_acc_graph = []

        for i in range(epoch):
            print("epoch", i + 1, "-------------------------------")
            for iter in range(iter_per_epoch):
                idxs = np.array(
                    range(iter * batch_size, (iter + 1) * batch_size))
                # idxs = np.random.choice(num_train, batch_size)
                x_batch = x[idxs]
                y_batch = y[idxs]

                val_idxs = np.random.choice(num_val, batch_size)
                x_val_batch = x_val[val_idxs]
                y_val_batch = y_val[val_idxs]

                train_loss, grads = self.loss(x_batch, y_batch)
                train_loss_graph.append(train_loss)

                val_loss, _ = self.loss(x_val_batch, y_val_batch)
                val_loss_graph.append(val_loss)

                writer.add_scalars('two_layer_loss', {
                                   'train': train_loss, 'val': val_loss}, iter)

                # SGD
                self.params['w_3'] -= lr * grads['w_3']
                self.params['b_3'] -= lr * grads['b_3']
                self.params['w_2'] -= lr * grads['w_2']
                self.params['b_2'] -= lr * grads['b_2']
                self.params['w_1'] -= lr * grads['w_1']
                self.params['b_1'] -= lr * grads['b_1']

                if iter % 10 == 0:
                    print("iteration %d / %d: loss %f" %
                          (iter, iter_per_epoch, train_loss))
                x_batch_predict, _ = self.predict(x_batch)
                train_acc = (x_batch_predict == y_batch).mean()

                x_val_batch_predict, _ = self.predict(x_val_batch)
                val_acc = (x_val_batch_predict == y_val_batch).mean()

                train_acc_graph.append(train_acc)
                val_acc_graph.append(val_acc)

                writer.add_scalars('two_layer_accuracy', {
                                   'train': train_acc, 'val': val_acc}, iter)
        writer.close()

        return {'train_loss_graph': train_loss_graph, 'val_loss_graph': val_loss_graph, 'train_acc_graph': train_acc_graph, 'val_acc_graph': val_acc_graph}

    def predict(self, x):
        params = self.params

        N = x.shape[0]
        conv_param = self.conv_param
        pool_param = self.pool_param

        # forward pass
        conv_1, _ = conv_relu_pool(
            x, params['w_1'], params['b_1'], conv_param, pool_param)
        conv_2, _ = conv_relu_pool(
            conv_1, params['w_2'], params['b_2'], conv_param, pool_param)
        scores, _ = linear(conv_2, params['w_3'], params['b_3'])

        y_pred = np.argmax(scores, axis=1)

        softmax = np.exp(scores)
        softmax /= np.sum(softmax, axis=1).reshape(N, 1)
        y_scores = np.max(softmax, axis=1)

        return y_pred, y_scores
